use std::{fmt,mem::size_of,sync::{Arc,atomic::{AtomicBool,Ordering}},time::Duration};

use anyhow::{Context as _,Result,anyhow,bail,ensure};
use gpui::{PrintOutcome,PrintPdfOptions};
use gpui_util::ResultExt as _;
use windows::{
    Data::Pdf::{PdfDocument,PdfPageRenderOptions},
    Foundation::IClosable,
    Graphics::Imaging::{BitmapAlphaMode,BitmapDecoder,BitmapEncoder,BitmapPixelFormat,BitmapTransform,ColorManagementMode,ExifOrientationMode},
    Storage::Streams::{DataWriter,InMemoryRandomAccessStream},
    Win32::{
        Foundation::{HGLOBAL,HWND,WAIT_FAILED},
        Graphics::Gdi::{self,BITMAPINFO,BITMAPINFOHEADER,BI_RGB,DIB_RGB_COLORS,GDI_ERROR,HDC,HALFTONE,HORZRES,LOGPIXELSX,LOGPIXELSY,PHYSICALHEIGHT,PHYSICALOFFSETX,PHYSICALOFFSETY,PHYSICALWIDTH,SRCCOPY,VERTRES},
        Storage::Xps::{AbortDoc,DOCINFOW,EndDoc,EndPage,SetAbortProc,StartDocW,StartPage},
        System::WinRT::{RO_INIT_SINGLETHREADED,RoInitialize,RoUninitialize},
        UI::{Controls::Dialogs::{PD_HIDEPRINTTOFILE,PD_NOCURRENTPAGE,PD_NOSELECTION,PD_PAGENUMS,PD_RESULT_PRINT,PD_RETURNDC,PD_USEDEVMODECOPIESANDCOLLATE,PRINTDLGEXW,PRINTPAGERANGE,PrintDlgExW,START_PAGE_GENERAL},
            WindowsAndMessaging::{DispatchMessageW,MSG,MWMO_INPUTAVAILABLE,MsgWaitForMultipleObjectsEx,PM_REMOVE,PeekMessageW,QS_ALLINPUT,TranslateMessage,WM_QUIT}},
    },
    core::{BOOL,Interface,PCWSTR},
};

use crate::{dialog::cancellation_requested,print_geometry::{MAX_PAGES,MAX_RASTER_PIXELS,PageFit,PrintableArea,fit_page,selected_pages}};

const MAX_PDF_BYTES:usize=64*1024*1024;
const MAX_ENCODED_PAGE_BYTES:u64=MAX_RASTER_PIXELS*4+4096;

#[derive(Debug)]
struct Cancelled;
impl fmt::Display for Cancelled {fn fmt(&self,formatter:&mut fmt::Formatter<'_>)->fmt::Result {formatter.write_str("printing cancelled")}}
impl std::error::Error for Cancelled {}

fn check_cancelled()->Result<()> {if cancellation_requested(){Err(Cancelled.into())}else{Ok(())}}

// WinRT's synchronous get is not an STA wait. Keep COM and the owner-cancellation
// timer serviced while the system PDF renderer or bitmap decoder is working.
macro_rules! complete {
    ($operation:expr) => {{
        let operation=$operation?;
        let terminal=Arc::new(AtomicBool::new(false));
        let registered=operation.when({
            let terminal=Arc::clone(&terminal);
            move |_|terminal.store(true,Ordering::Release)
        });
        wait_for(&terminal,registered,||operation.Status().map(|status|status.0!=0),||operation.Cancel(),||operation.GetResults(),||operation.Close(),cancellation_requested,wait_for_messages)
    }};
}

pub(crate) fn print_pdf(owner:HWND,options:PrintPdfOptions)->Result<PrintOutcome> {
    match print(owner,options) {
        Err(error) if error.downcast_ref::<Cancelled>().is_some()=>Ok(PrintOutcome::Cancelled),
        result=>result,
    }
}

fn print(owner:HWND,options:PrintPdfOptions)->Result<PrintOutcome> {
    check_cancelled()?;
    ensure!(!owner.0.is_null(),"Printing requires an owning window.");
    ensure!(options.data.len()<=MAX_PDF_BYTES,"Windows printing supports PDF files up to 64 MiB.");
    ensure!(options.data.starts_with(b"%PDF-"),"Printing requires a PDF document.");
    ensure!(!options.title.trim().is_empty()&&options.title.len()<=1024&&!options.title.contains('\0'),"Printing requires a document title without NUL characters, up to 1,024 bytes.");
    let _apartment=WinRtApartment::new()?;
    let input=Closable::new(InMemoryRandomAccessStream::new()?,"PDF input stream")?;
    let writer=Closable::new(DataWriter::CreateDataWriter(&input.value.GetOutputStreamAt(0)?)?,"PDF input writer")?;
    writer.value.WriteBytes(&options.data)?;
    let stored=complete!(writer.value.StoreAsync()).context("copying PDF data to Windows")?;
    ensure!(stored as usize==options.data.len(),"Windows did not receive the complete PDF document.");
    drop(writer.value.DetachStream().context("detaching PDF input writer")?);writer.close()?;input.value.Seek(0)?;
    let document=complete!(PdfDocument::LoadFromStreamAsync(&input.value)).context("opening the PDF for Windows printing")?;
    let page_count=document.PageCount()?;ensure!(page_count>0&&page_count<=MAX_PAGES,"Windows printing supports between 1 and 10,000 PDF pages.");
    check_cancelled()?;
    let mut settings=PrintSettings::new(owner,page_count);
    unsafe {PrintDlgExW(&mut settings.dialog)}.context("opening the Windows print dialog")?;
    check_cancelled()?;
    if settings.dialog.dwResultAction!=PD_RESULT_PRINT {settings.close()?;drop(document);input.close()?;return Ok(PrintOutcome::Cancelled);}
    ensure!(!settings.dialog.hDC.0.is_null(),"The selected printer did not provide a device context.");
    let selected=settings.selected_pages(page_count)?;let hdc=settings.dialog.hDC;let area=printer_area(hdc)?;
    let title:Vec<u16>=options.title.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {ensure!(SetAbortProc(hdc,Some(abort_print))>0,"Installing the printer cancellation callback failed.");}
    let job=PrintJob::start(&settings,&title)?;
    // With PD_USEDEVMODECOPIESANDCOLLATE the returned HDC carries copies,
    // collation, duplex, paper size and orientation. Emit each selected page once.
    for index in selected {
        check_cancelled()?;pump_messages()?;
        let page=Closable::new(document.GetPage(index).with_context(||format!("opening PDF page {}",index+1))?,"PDF page")?;
        let size=page.value.Size()?;let fit=fit_page(size.Width as f64,size.Height as f64,area).map_err(anyhow::Error::msg)?;
        log::debug!("Windows PDF print page {}: {}x{} raster, {:.1} effective DPI, fitted to {}x{} device pixels",index+1,fit.raster_width,fit.raster_height,fit.effective_dpi,fit.width,fit.height);
        let output=Closable::new(InMemoryRandomAccessStream::new()?,"PDF page raster stream")?;
        let render=PdfPageRenderOptions::new()?;render.SetDestinationWidth(fit.raster_width)?;render.SetDestinationHeight(fit.raster_height)?;
        render.SetIsIgnoringHighContrast(true)?;render.SetBitmapEncoderId(BitmapEncoder::BmpEncoderId()?)?;
        complete!(page.value.RenderWithOptionsToStreamAsync(&output.value,&render)).with_context(||format!("rendering PDF page {}",index+1))?;
        ensure!(output.value.Size()?<=MAX_ENCODED_PAGE_BYTES,"The rendered PDF page exceeds the Windows print buffer limit.");
        output.value.Seek(0)?;
        let decoder=complete!(BitmapDecoder::CreateAsync(&output.value)).context("decoding the PDF print raster")?;
        ensure!(decoder.PixelWidth()?==fit.raster_width&&decoder.PixelHeight()?==fit.raster_height,"Windows returned an unexpected PDF raster size.");
        let transform=BitmapTransform::new()?;
        let pixels=complete!(decoder.GetPixelDataTransformedAsync(BitmapPixelFormat::Bgra8,BitmapAlphaMode::Ignore,&transform,ExifOrientationMode::IgnoreExifOrientation,ColorManagementMode::DoNotColorManage)).context("reading the PDF print pixels")?.DetachPixelData()?;
        let byte_count=fit.raster_width as usize*fit.raster_height as usize*4;
        ensure!(pixels.len()==byte_count,"Windows returned an incomplete PDF print raster.");
        check_cancelled()?;
        gdi_result(unsafe {StartPage(hdc)},"starting a printer page")?;
        draw_page(hdc,fit,&pixels)?;
        gdi_result(unsafe {EndPage(hdc)},"finishing a printer page")?;
        drop(pixels);drop(decoder);output.close()?;page.close()?;
    }
    check_cancelled()?;job.finish()?;drop(document);
    input.close().context("Windows accepted the print job, but closing its PDF input failed")?;
    settings.close().context("Windows accepted the print job, but releasing its printer resources failed")?;
    Ok(PrintOutcome::Submitted)
}

fn wait_for<T>(terminal:&AtomicBool,registered:windows::core::Result<()>,completed:impl Fn()->windows::core::Result<bool>,cancel:impl Fn()->windows::core::Result<()>,result:impl FnOnce()->windows::core::Result<T>,close:impl FnOnce()->windows::core::Result<()>,cancelled:impl Fn()->bool,mut wait:impl FnMut()->Result<()>)->Result<T> {
    let mut error=None;
    if let Err(registration_error)=registered {
        record_error(&mut error,anyhow::Error::from(registration_error).context("registering Windows PDF completion; resources will remain held until terminal completion is observed"));
    }
    let mut status_error_seen=false;
    let mut wait_error_seen=false;
    let mut cancel_attempted=false;
    loop {
        if terminal.load(Ordering::Acquire){break;}
        match completed(){
            Ok(true)=>break,
            Ok(false)=>{},
            Err(status_error)=>{
                if !status_error_seen {record_error(&mut error,anyhow::Error::from(status_error).context("reading the Windows PDF operation status"));status_error_seen=true;}
            },
        }
        if (cancelled()||error.is_some())&&!cancel_attempted {
            cancel_attempted=true;
            if let Err(cancel_error)=cancel(){record_error(&mut error,anyhow::Error::from(cancel_error).context("cancelling the Windows PDF operation"));}
        }
        if let Err(wait_error)=wait()&& !wait_error_seen {record_error(&mut error,wait_error);wait_error_seen=true;}
    }
    // A failed registration/status/cancel/wait does not make a hot operation safe
    // to close. The callback or a valid terminal status must establish that first.
    let outcome=result().map_err(anyhow::Error::from);
    let closed=close().context("closing the Windows PDF operation");
    if let Some(error)=error {
        if let Err(result_error)=outcome {log::debug!("Windows PDF operation finished after an earlier failure: {result_error:#}");}
        return match closed {
            Ok(())=>Err(error),
            Err(close_error) if error.downcast_ref::<Cancelled>().is_some()=>Err(close_error),
            Err(close_error)=>Err(error.context(format!("Closing the Windows PDF operation also failed: {close_error:#}"))),
        };
    }
    if cancelled(){closed?;return Err(Cancelled.into());}
    match (outcome,closed) {
        (Ok(value),Ok(()))=>Ok(value),
        (Err(error),Ok(()))|(Ok(_),Err(error))=>Err(error),
        (Err(error),Err(close_error))=>Err(error.context(format!("Closing the Windows PDF operation also failed: {close_error:#}"))),
    }
}

fn record_error(slot:&mut Option<anyhow::Error>,error:anyhow::Error) {
    let cancellation=error.downcast_ref::<Cancelled>().is_some();
    if !cancellation{log::error!("{error:#}");}
    if slot.is_none()||(!cancellation&&slot.as_ref().is_some_and(|error|error.downcast_ref::<Cancelled>().is_some())){*slot=Some(error);}
}

fn wait_for_messages()->Result<()> {
    let wait=unsafe {MsgWaitForMultipleObjectsEx(None,50,QS_ALLINPUT,MWMO_INPUTAVAILABLE)};
    let wait_error=(wait==WAIT_FAILED).then(windows::core::Error::from_thread);
    let message_result=pump_messages();
    if let Some(error)=wait_error {
        // A broken wait API must not spin while completion/resources remain held.
        std::thread::sleep(Duration::from_millis(50));
        let error=anyhow::Error::from(error).context("waiting for the Windows PDF renderer");
        return match message_result {
            Ok(())=>Err(error),
            Err(message_error)=>Err(error.context(format!("Processing Windows print messages also failed: {message_error:#}"))),
        };
    }
    message_result
}

fn pump_messages()->Result<()> {
    let mut message=MSG::default();
    for _ in 0..128 {
        if !unsafe {PeekMessageW(&mut message,None,0,0,PM_REMOVE)}.as_bool(){break;}
        if message.message==WM_QUIT {return Err(Cancelled.into());}
        unsafe {TranslateMessage(&message);DispatchMessageW(&message);}
    }
    Ok(())
}

unsafe extern "system" fn abort_print(_:HDC,_:i32)->BOOL {
    if let Err(error)=pump_messages(){if error.downcast_ref::<Cancelled>().is_none(){log::error!("Windows print message processing failed: {error:#}");}return false.into();}
    (!cancellation_requested()).into()
}

fn gdi_result(value:i32,operation:&str)->Result<()> {
    if value>0{return Ok(());}
    check_cancelled()?;Err(windows::core::Error::from_thread()).with_context(||format!("{operation} failed"))
}

fn printer_area(hdc:HDC)->Result<PrintableArea> {
    let caps=|index|unsafe {gdi::GetDeviceCaps(Some(hdc),index)};
    let area=PrintableArea{width:caps(HORZRES),height:caps(VERTRES),dpi_x:caps(LOGPIXELSX),dpi_y:caps(LOGPIXELSY)};
    ensure!(area.width>0&&area.height>0&&area.dpi_x>0&&area.dpi_y>0,"The printer reported an invalid printable area.");
    log::debug!("Windows PDF printer: physical {}x{}, printable {}x{}, hardware offsets {},{}, DPI {}x{}",caps(PHYSICALWIDTH),caps(PHYSICALHEIGHT),area.width,area.height,caps(PHYSICALOFFSETX),caps(PHYSICALOFFSETY),area.dpi_x,area.dpi_y);
    Ok(area)
}

fn draw_page(hdc:HDC,fit:PageFit,pixels:&[u8])->Result<()> {
    let mut bitmap=BITMAPINFO::default();bitmap.bmiHeader=BITMAPINFOHEADER{biSize:size_of::<BITMAPINFOHEADER>() as u32,biWidth:fit.raster_width as i32,biHeight:-(fit.raster_height as i32),biPlanes:1,biBitCount:32,biCompression:BI_RGB.0,..Default::default()};
    unsafe {
        ensure!(gdi::SetStretchBltMode(hdc,HALFTONE)!=0,"The printer does not support raster scaling.");
        gdi::SetBrushOrgEx(hdc,0,0,None).ok().context("setting printer raster alignment")?;
        let lines=gdi::StretchDIBits(hdc,fit.left,fit.top,fit.width,fit.height,0,0,fit.raster_width as i32,fit.raster_height as i32,Some(pixels.as_ptr().cast()),&bitmap,DIB_RGB_COLORS,SRCCOPY);
        if lines==0||lines==GDI_ERROR{check_cancelled()?;bail!("The printer could not accept the PDF page raster.");}
    }
    Ok(())
}

struct WinRtApartment;
impl WinRtApartment {fn new()->Result<Self>{unsafe {RoInitialize(RO_INIT_SINGLETHREADED)}.context("initializing Windows PDF printing")?;Ok(Self)}}
impl Drop for WinRtApartment {fn drop(&mut self){unsafe {RoUninitialize()};}}

struct Closable<T:Interface> {value:T,closable:IClosable,closed:bool,label:&'static str}
impl<T:Interface> Closable<T> {
    fn new(value:T,label:&'static str)->Result<Self>{let closable=value.cast::<IClosable>()?;Ok(Self{value,closable,closed:false,label})}
    fn close(mut self)->Result<()>{let result=self.closable.Close();self.closed=result.is_ok();result.with_context(||format!("closing {}",self.label))}
}
impl<T:Interface> Drop for Closable<T> {fn drop(&mut self){if !self.closed{self.closable.Close().with_context(||format!("closing {}",self.label)).log_err();}}}

struct PrintSettings {dialog:PRINTDLGEXW,ranges:Vec<PRINTPAGERANGE>}
impl PrintSettings {
    fn new(owner:HWND,count:u32)->Self {
        let mut ranges=vec![PRINTPAGERANGE{nFromPage:1,nToPage:count};32];
        let dialog=PRINTDLGEXW{lStructSize:size_of::<PRINTDLGEXW>() as u32,hwndOwner:owner,Flags:PD_RETURNDC|PD_USEDEVMODECOPIESANDCOLLATE|PD_NOSELECTION|PD_NOCURRENTPAGE|PD_HIDEPRINTTOFILE,nPageRanges:1,nMaxPageRanges:ranges.len() as u32,lpPageRanges:ranges.as_mut_ptr(),nMinPage:1,nMaxPage:count,nCopies:1,nStartPage:START_PAGE_GENERAL,..Default::default()};
        Self{dialog,ranges}
    }
    fn selected_pages(&self,count:u32)->Result<Vec<u32>> {
        if !self.dialog.Flags.contains(PD_PAGENUMS){return selected_pages(count,None).map_err(anyhow::Error::msg);}
        let count_ranges=self.dialog.nPageRanges as usize;ensure!(count_ranges>0&&count_ranges<=self.ranges.len(),"The print dialog returned an invalid page selection.");
        let ranges:Vec<_>=self.ranges[..count_ranges].iter().map(|range|(range.nFromPage,range.nToPage)).collect();selected_pages(count,Some(&ranges)).map_err(anyhow::Error::msg)
    }
}
impl PrintSettings {
    fn release(&mut self)->Result<()> {
        let mut errors=Vec::new();
        if !self.dialog.hDC.0.is_null(){match unsafe {gdi::DeleteDC(self.dialog.hDC)}.ok(){Ok(())=>self.dialog.hDC=HDC::default(),Err(error)=>errors.push(format!("releasing printer device context: {error}"))}}
        for (index,handle) in [self.dialog.hDevMode,self.dialog.hDevNames].into_iter().enumerate(){
            if handle.0.is_null(){continue;}
            if unsafe {free_global(handle)}.0.is_null(){if index==0{self.dialog.hDevMode=HGLOBAL::default();}else{self.dialog.hDevNames=HGLOBAL::default();}}
            else{errors.push(format!("releasing Windows print settings: {}",windows::core::Error::from_thread()));}
        }
        if errors.is_empty(){Ok(())}else{Err(anyhow!(errors.join("; ")))}
    }
    fn close(mut self)->Result<()>{self.release()}
}
impl Drop for PrintSettings {fn drop(&mut self){self.release().log_err();}}

// windows 0.62 projects GlobalFree's NULL-success convention as an invalid-handle
// error and loses the raw return. Retain it so successful frees are not errors.
#[link(name="kernel32")]
unsafe extern "system" {#[link_name="GlobalFree"]fn free_global(memory:HGLOBAL)->HGLOBAL;}

struct PrintJob<'a> {settings:&'a PrintSettings,active:bool}
impl<'a> PrintJob<'a> {
    fn start(settings:&'a PrintSettings,title:&[u16])->Result<Self> {
        check_cancelled()?;let info=DOCINFOW{cbSize:size_of::<DOCINFOW>() as i32,lpszDocName:PCWSTR(title.as_ptr()),..Default::default()};
        gdi_result(unsafe {StartDocW(settings.dialog.hDC,&info)},"starting the print job")?;Ok(Self{settings,active:true})
    }
    fn finish(mut self)->Result<()> {gdi_result(unsafe {EndDoc(self.settings.dialog.hDC)},"submitting the print job")?;self.active=false;Ok(())}
}
impl Drop for PrintJob<'_> {fn drop(&mut self){if self.active&&unsafe {AbortDoc(self.settings.dialog.hDC)}<=0{log::error!("Could not abort the Windows PDF print job: {}",windows::core::Error::from_thread());}}}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn windows_pdf_async_failures_drain_before_closing() {
        for fault in ["registration","status","cancel","wait"] {
            let terminal=AtomicBool::new(false);
            let steps=Cell::new(0);
            let cancel_calls=Cell::new(0);
            let close_calls=Cell::new(0);
            let failure=||windows::core::Error::new(windows::core::HRESULT(0x80004005u32 as i32),"injected async failure");
            let registered=if fault=="registration"{Err(failure())}else{Ok(())};
            let outcome=wait_for(&terminal,registered,||{
                if fault=="status"&&steps.get()==0{Err(failure())}
                else{Ok(fault=="registration"&&steps.get()>=3)}
            },||{
                cancel_calls.set(cancel_calls.get()+1);
                if matches!(fault,"cancel"|"status"){Err(failure())}else{Ok(())}
            },||{
                assert!(steps.get()>=3,"GetResults must wait for terminal completion after {fault} failure");Ok(7)
            },||{
                assert!(steps.get()>=3,"Close must wait for terminal completion after {fault} failure");
                close_calls.set(close_calls.get()+1);Ok(())
            },||fault=="cancel",||{
                assert_eq!(close_calls.get(),0,"the owner/resources must remain held while {fault} is pending");
                steps.set(steps.get()+1);
                if steps.get()>=3&&fault!="registration"{terminal.store(true,Ordering::Release);}
                if fault=="wait"&&steps.get()==1{Err(anyhow!("waiting for the Windows PDF renderer"))}else{Ok(())}
            });
            let Err(error)=outcome else{panic!("{fault} failure must remain visible after native completion");};
            let context=match fault {"registration"=>"registering", "status"=>"reading", "cancel"=>"cancelling", _=>"waiting"};
            assert!(error.to_string().starts_with(context),"the first error must survive draining: {error:#}");
            assert_eq!(steps.get(),3);assert_eq!(cancel_calls.get(),1);assert_eq!(close_calls.get(),1);
        }
    }
}
