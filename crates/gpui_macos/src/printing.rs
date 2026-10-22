use std::{ffi::c_void, sync::{Arc, atomic::{AtomicBool, Ordering}}};

use anyhow::{Context as _, Result, anyhow, ensure};
use futures::{channel::oneshot, future::{Either, select}};
use gpui::{ForegroundExecutor, PrintOutcome, PrintPdfOptions};
use objc2::{AnyThread as _, DefinedClass as _, MainThreadMarker, define_class, msg_send, rc::Retained, sel};
use objc2_app_kit::{NSModalResponseCancel, NSPrintCancelJob, NSPrintInfo, NSPrintOperation, NSWindow};
use objc2_foundation::{NSData, NSObject, NSObjectProtocol, NSString};
use objc2_pdf_kit::{PDFDocument, PDFPrintScalingMode};
use parking_lot::Mutex;

// These signatures and status values are from PrintCore's PMCore.h, PMDefinitions.h and PMErrors.h.
#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn PMSessionSetError(session: *mut c_void, error: i32) -> i32;
    fn PMSessionError(session: *mut c_void) -> i32;
}
const PRINT_CANCELLED: i32 = 0x0080;
const PRINT_JOB_CANCELLED: i32 = -9643;

#[derive(Clone, Default)]
pub(crate) struct PrintOwner(Arc<OwnerState>);
#[derive(Default)]
struct OwnerState {
    closed: AtomicBool,
    cancellation: Mutex<Option<oneshot::Sender<()>>>,
}

impl PrintOwner {
    pub(crate) fn close(&self) {
        self.0.closed.store(true, Ordering::Release);
        if let Some(sender) = self.0.cancellation.lock().take()
            && sender.send(()).is_err() { log::debug!("Print operation already completed before its owner closed"); }
    }

    fn begin(&self) -> Result<oneshot::Receiver<()>> {
        let mut cancellation = self.0.cancellation.lock();
        ensure!(!self.0.closed.load(Ordering::Acquire), "The print window has closed");
        ensure!(cancellation.is_none(), "A print operation is already open for this window");
        let (sender, receiver) = oneshot::channel();
        *cancellation = Some(sender);
        Ok(receiver)
    }
}

struct CompletionIvars { sender: Mutex<Option<oneshot::Sender<bool>>> }

define_class!(
    // AppKit may invoke this selector on its printing thread. Only the channel is touched there.
    #[unsafe(super(NSObject))]
    #[ivars = CompletionIvars]
    struct GPUIPrintCompletionDelegate;

    unsafe impl NSObjectProtocol for GPUIPrintCompletionDelegate {}

    impl GPUIPrintCompletionDelegate {
        #[unsafe(method(printOperationDidRun:success:contextInfo:))]
        fn finished(&self, _operation: &NSPrintOperation, success: bool, _context: *mut c_void) {
            // The foreground may release its delegate as soon as the channel wakes it.
            let _keep_alive = unsafe { Retained::retain(self as *const Self as *mut Self) };
            if let Some(sender) = self.ivars().sender.lock().take()
                && sender.send(success).is_err() { log::debug!("Print completion receiver closed"); }
        }
    }
);

impl GPUIPrintCompletionDelegate {
    fn new(sender: oneshot::Sender<bool>) -> Retained<Self> {
        let allocated = Self::alloc().set_ivars(CompletionIvars { sender: Mutex::new(Some(sender)) });
        // SAFETY: NSObject's init is the superclass designated initializer.
        unsafe { msg_send![super(allocated), init] }
    }
}

struct NativePrint {
    _document: Retained<PDFDocument>,
    operation: Retained<NSPrintOperation>,
    parent: Retained<NSWindow>,
    _delegate: Retained<GPUIPrintCompletionDelegate>,
    initial_sheet: Option<Retained<NSWindow>>,
    marker: MainThreadMarker,
}

impl NativePrint {
    fn begin(parent: Retained<NSWindow>, options: &PrintPdfOptions, marker: MainThreadMarker) -> Result<(Self, oneshot::Receiver<bool>)> {
        options.validate()?;
        ensure!(parent.attachedSheet().is_none(), "Finish the current window dialog before printing");
        let data = NSData::with_bytes(&options.data);
        // SAFETY: NSData owns the complete immutable PDF snapshot through initialization.
        let document = unsafe { PDFDocument::initWithData(PDFDocument::alloc(), &data) }.context("The operating system could not read this PDF")?;
        let pages = unsafe { document.pageCount() };
        ensure!(pages > 0 && pages <= 4096, "The PDF must contain between 1 and 4096 pages to print");
        let info = NSPrintInfo::new();
        // SAFETY: PDFKit/AppKit printing is initialized on the owning foreground thread.
        let operation = unsafe { document.printOperationForPrintInfo_scalingMode_autoRotate(Some(&info), PDFPrintScalingMode::PageScaleDownToFit, true, marker) }.context("The operating system could not create a PDF print operation")?;
        operation.setJobTitle(Some(&NSString::from_str(&options.title)));
        operation.setShowsPrintPanel(true);
        operation.setShowsProgressPanel(true);
        operation.setCanSpawnSeparateThread(true);
        let (sender, receiver) = oneshot::channel();
        let delegate = GPUIPrintCompletionDelegate::new(sender);
        // SAFETY: The selector has the documented operation/bool/context signature, and
        // NativePrint retains the owner, operation and delegate until its callback finishes.
        unsafe { operation.runOperationModalForWindow_delegate_didRunSelector_contextInfo(&parent, Some(&delegate), Some(sel!(printOperationDidRun:success:contextInfo:)), std::ptr::null_mut()); }
        let initial_sheet = parent.attachedSheet();
        Ok((Self { _document: document, operation, parent, _delegate: delegate, initial_sheet, marker }, receiver))
    }

    fn cancel(&self) {
        let info = self.operation.printInfo();
        info.setJobDisposition(unsafe { NSPrintCancelJob });
        // The print worker observes the session error even after the settings sheet closes.
        let status = unsafe { PMSessionSetError(info.PMPrintSession().as_ptr(), PRINT_CANCELLED) };
        if status != 0 { log::warn!("Could not cancel native print session: {status}"); }
        if let Some(sheet) = self.parent.attachedSheet() {
            let captured_sheet = self.initial_sheet.as_deref() == Some(&*sheet);
            let current_operation = NSPrintOperation::currentOperation(self.marker).as_deref() == Some(&*self.operation);
            if captured_sheet || (self.initial_sheet.is_none() && current_operation) {
                self.parent.endSheet_returnCode(&sheet, NSModalResponseCancel);
            }
        }
    }

    fn outcome(&self, success: bool, cancellation_requested: bool) -> Result<PrintOutcome> {
        if success { return Ok(PrintOutcome::Submitted); }
        let info = self.operation.printInfo();
        let status = unsafe { PMSessionError(info.PMPrintSession().as_ptr()) };
        if cancellation_requested || &*info.jobDisposition() == unsafe { NSPrintCancelJob } || matches!(status, PRINT_CANCELLED | PRINT_JOB_CANCELLED) {
            Ok(PrintOutcome::Cancelled)
        } else { Err(anyhow!("The operating system could not complete printing (status {status})")) }
    }
}

pub(crate) fn print_pdf(parent: Retained<NSWindow>, marker: MainThreadMarker, options: PrintPdfOptions, owner: PrintOwner, executor: &ForegroundExecutor) -> oneshot::Receiver<Result<PrintOutcome>> {
    let (mut sender, receiver) = oneshot::channel();
    let cancellation = match owner.begin() {
        Ok(cancellation) => cancellation,
        Err(error) => { if sender.send(Err(error)).is_err() { log::debug!("Print receiver closed before initialization"); } return receiver; }
    };
    executor.spawn(async move {
        let result = async {
            if sender.is_canceled() || owner.0.closed.load(Ordering::Acquire) { return Ok(PrintOutcome::Cancelled); }
            let (native, completion) = NativePrint::begin(parent, &options, marker)?;
            drop(options);
            let cancelled = async { match select(sender.cancellation(), cancellation).await { Either::Left(_) | Either::Right(_) => {} } };
            futures::pin_mut!(cancelled);
            match select(completion, cancelled).await {
                Either::Left((success, _)) => native.outcome(success.context("Native print completion was interrupted")?, false),
                Either::Right(((), completion)) => {
                    native.cancel();
                    native.outcome(completion.await.context("Native print cancellation was interrupted")?, true)
                }
            }
        }.await;
        drop(owner.0.cancellation.lock().take());
        if sender.send(result).is_err() { log::debug!("Native print finished after its caller closed"); }
    }).detach();
    receiver
}
