use std::{cell::{Cell, RefCell}, collections::HashMap, future::Future, io::Write as _, os::fd::AsFd as _, rc::Rc};

use anyhow::{Context as _, Result, anyhow, ensure};
use ashpd::{WindowIdentifier, zbus::{self, MatchRule, MessageStream, Proxy, message::Type, zvariant::{Fd, OwnedObjectPath, OwnedValue, Value}}};
use futures::{FutureExt as _, StreamExt as _, channel::oneshot, future::{Either, FusedFuture as _, select}};
use gpui::{BackgroundExecutor, ForegroundExecutor, PrintOutcome, PrintPdfOptions};

#[derive(Clone, Default)]
pub(crate) struct PrintOwner(Rc<OwnerState>);
#[derive(Default)]
struct OwnerState {
    closed: Cell<bool>,
    cancellation: RefCell<Option<oneshot::Sender<()>>>,
}

impl PrintOwner {
    pub(crate) fn close(&self) {
        self.0.closed.set(true);
        if let Some(sender) = self.0.cancellation.borrow_mut().take()
            && sender.send(()).is_err() { log::debug!("Portal print already completed before owner closure"); }
    }

    fn begin(&self) -> Result<oneshot::Receiver<()>> {
        ensure!(!self.0.closed.get(), "The print window has closed");
        ensure!(self.0.cancellation.borrow().is_none(), "A print dialog is already open for this window");
        let (sender, receiver) = oneshot::channel();
        drop(self.0.cancellation.replace(Some(sender)));
        Ok(receiver)
    }
}

async fn close_request(connection: &zbus::Connection, path: &OwnedObjectPath) -> Result<()> {
    let request = Proxy::new(connection, "org.freedesktop.portal.Desktop", path.as_str(), "org.freedesktop.portal.Request").await?;
    request.call::<_, _, ()>("Close", &()).await.context("The desktop could not close its print dialog")
}

async fn run_request(file: &std::fs::File, identifier: &WindowIdentifier, title: &str, cancellation: impl Future<Output = ()>) -> Result<PrintOutcome> {
    let connection = zbus::Connection::session().await.context("The desktop print service is unavailable")?;
    let portal = Proxy::new(&connection, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.Print").await?;
    // Subscribe before Print. Matching the request namespace also handles backends that
    // return a different object path without racing an immediate Response signal.
    let rule = MatchRule::builder().msg_type(Type::Signal).sender("org.freedesktop.portal.Desktop")?
        .interface("org.freedesktop.portal.Request")?.member("Response")?
        .path_namespace("/org/freedesktop/portal/desktop/request")?.build();
    let mut responses = MessageStream::for_match_rule(rule, &connection, Some(64)).await?;
    let sender_name = connection.unique_name().context("The print connection has no bus identity")?.as_str().trim_start_matches(':').replace('.', "_");
    let request_prefix = format!("/org/freedesktop/portal/desktop/request/{sender_name}/");
    let token = format!("gpui_print_{}", ashpd::desktop::HandleToken::default());
    let options = HashMap::from([
        ("handle_token", Value::from(token.as_str())),
        ("modal", Value::from(true)),
        ("supported_output_file_formats", Value::from(vec!["pdf"])),
    ]);
    let body = (identifier.to_string(), title, Fd::from(file.as_fd()), options);
    let call = portal.call::<_, _, OwnedObjectPath>("Print", &body);
    futures::pin_mut!(call);
    let cancellation = cancellation.fuse();
    futures::pin_mut!(cancellation);
    if cancellation.as_mut().now_or_never().is_some() { return Ok(PrintOutcome::Cancelled); }
    let handle = match select(call, &mut cancellation).await {
        Either::Left((handle, _)) => handle.context("The desktop print service could not open its dialog")?,
        Either::Right(((), call)) => {
            // Print's method reply is the handle, not the user's response. Keep the RPC
            // alive long enough to close the exact request the desktop created.
            let handle = call.await.context("The desktop print service did not acknowledge cancellation")?;
            ensure!(handle.as_str().starts_with(&request_prefix), "The desktop returned a print request belonging to a different connection");
            close_request(&connection, &handle).await?;
            return Ok(PrintOutcome::Cancelled);
        }
    };
    ensure!(handle.as_str().starts_with(&request_prefix), "The desktop returned a print request belonging to a different connection");
    loop {
        futures::select_biased! {
            message = responses.next().fuse() => {
                let message = message.context("The desktop print service closed before replying")??;
                if message.header().path().map(|path| path.as_str()) != Some(handle.as_str()) { continue; }
                let (status, _results): (u32, HashMap<String, OwnedValue>) = message.body().deserialize()?;
                return match status {
                    0 => Ok(PrintOutcome::Submitted),
                    1 => Ok(PrintOutcome::Cancelled),
                    _ => Err(anyhow!("The desktop could not complete printing (response {status})")),
                };
            }
            () = cancellation => {
                close_request(&connection, &handle).await?;
                return Ok(PrintOutcome::Cancelled);
            }
        }
    }
}

pub(crate) fn print_pdf(options: PrintPdfOptions, owner: PrintOwner, identifier: impl Future<Output = Option<WindowIdentifier>> + 'static, foreground: &ForegroundExecutor, background: &BackgroundExecutor) -> oneshot::Receiver<Result<PrintOutcome>> {
    let (mut sender, receiver) = oneshot::channel();
    let owner_cancellation = match owner.begin() {
        Ok(cancellation) => cancellation,
        Err(error) => { if sender.send(Err(error)).is_err() { log::debug!("Portal print receiver closed before initialization"); } return receiver; }
    };
    let (identifier_sender, identifier_receiver) = oneshot::channel();
    let (identifier_finished, identifier_cleanup) = oneshot::channel();
    // ashpd owns its Wayland export worker. Let that export finish even after the
    // caller cancels so dropping its private receiver cannot panic the worker.
    foreground.spawn(async move {
        if identifier_sender.send(identifier.await).is_err() { log::debug!("Print parent export completed after cancellation"); }
        if identifier_finished.send(()).is_err() { log::debug!("Print parent export owner was released"); }
    }).detach();
    let preparation = background.spawn(async move {
        options.validate()?;
        let mut temporary = tempfile::NamedTempFile::new().context("Could not create private print storage")?;
        temporary.write_all(&options.data).context("Could not prepare the PDF for printing")?;
        let file = std::fs::File::open(temporary.path()).context("Could not open private print storage for reading")?;
        temporary.close().context("Could not remove the private print filename")?;
        Ok::<_, anyhow::Error>((file, options.title))
    });
    foreground.spawn(async move {
        let mut preparation = preparation.fuse();
        let result = async {
            if sender.is_canceled() || owner.0.closed.get() { return Ok(PrintOutcome::Cancelled); }
            let cancellation = async { match select(sender.cancellation(), owner_cancellation).await { Either::Left(_) | Either::Right(_) => {} } }.fuse();
            futures::pin_mut!(cancellation);
            let prepared = async { Ok::<_, anyhow::Error>(((&mut preparation).await?, identifier_receiver.await.context("The print parent export was interrupted")?.context("The desktop could not identify the print window")?)) };
            futures::pin_mut!(prepared);
            let ((file, title), identifier) = match select(prepared, &mut cancellation).await {
                Either::Left((prepared, _)) => prepared?,
                Either::Right(((), _)) => return Ok(PrintOutcome::Cancelled),
            };
            let request = run_request(&file, &identifier, &title, cancellation);
            request.await
        }.await;
        if sender.send(result).is_err() { log::debug!("Portal print finished after its caller closed"); }
        // A cancelled prepared future only borrows this task. Retain and join it
        // before releasing the lease because write_all cannot be preempted.
        if !preparation.is_terminated()
            && let Err(error) = preparation.await { log::debug!("PDF preparation failed after cancellation: {error:#}"); }
        // Keep the per-window lease until ashpd's export worker also exits.
        if identifier_cleanup.await.is_err() { log::warn!("Print parent export cleanup was interrupted"); }
        drop(owner.0.cancellation.borrow_mut().take());
    }).detach();
    receiver
}
