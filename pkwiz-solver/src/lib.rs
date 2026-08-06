//! # pkwiz-solver — the engine, behind a pipe
//!
//! A standalone GTO solver, spoken to over newline-delimited JSON on stdin/stdout.
//!
//! ## Why a separate program
//!
//! `postflop-solver` is AGPL-3.0. An application that wants to solve a spot but is not itself
//! AGPL cannot link it — so it does not. This binary links it, and lives in the same repository
//! as the engine so the two move together; the application holds a URL, downloads the binary, and
//! talks to it over a pipe. No AGPL code, and no type from the engine, ever crosses that pipe.
//!
//! The arrangement is not only a licensing one, and would be worth having anyway: a solve is
//! minutes of numerical code, and running it in its own address space means a crash costs one job
//! rather than the host.
//!
//! ```text
//! -> {"id":1,"cmd":"solve","spot":{"oop":"QQ+","ip":"JJ+","board":"Td9d6h","pot":100,
//!                                  "effectiveStack":300}}
//! <- {"id":1,"ok":true,"result":{"jobId":1,"phase":"queued",…}}
//! <- {"event":"job","job":{"jobId":1,"phase":"running","iterations":40,"exploitability":2.1,…}}
//! ```
//!
//! ## The session contract
//!
//! Deliberately plain, because the interesting part of this protocol is the job lifecycle and
//! not the envelope:
//!
//! **A command can fail; a session cannot.** Bad JSON, an unknown command, an unknown job, a
//! panic inside the engine — each comes back as one JSON response and the loop continues. The
//! process exits non-zero only if stdout itself breaks.
//!
//! **One line in, one line out** — plus pushed `{"event":"job",…}` frames at any time, which is
//! the one thing here a request/response protocol does not have. See [`protocol`].
//!
//! ## What is different, and why
//!
//! A parse is milliseconds and a solve is minutes. Everything below follows from that:
//!
//! - `solve` **queues** and returns; it does not block the session.
//! - Progress is **pushed** while the solve runs, throttled to a frame every 200 ms, and is also
//!   pollable via `progress` for a host that would rather ask.
//! - `cancel` sets a flag the CFR loop checks once per iteration, and returns without waiting.
//! - A cancelled solve is still **finalized**, so stopping early gives you a rough answer rather
//!   than nothing.
//! - Finished solutions can be written to disk and reopened as a job, so a host can offer a
//!   library of past solves rather than only the one in flight.
//! - A finished job the host is done with can be `forget`ten — the only way to free a tree that
//!   was never saved, since `release` refuses to discard the sole copy of a solve.
//!
//! ## A host should treat this binary as optional
//!
//! It is published as a release artifact rather than shipped inside anything, so on any given
//! machine it may simply not be there. That is an ordinary state, not a failure: a host is
//! expected to say "no solver installed" and carry on, and to offer `--version` as the check that
//! a binary it found is one it can actually talk to.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod convert;
pub mod engine;
pub mod jobs;
pub mod protocol;
pub mod spot;

use std::io::{BufRead, Read, Write};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

pub use engine::{EngineError, MemoryEstimate, Sample, Solved, Stopped, DEFAULT_MEMORY_LIMIT};
pub use jobs::{Emit, JobError, JobId, JobStatus, Jobs, NodeView, Phase, Silent};
pub use protocol::{execute, Command, OpError, PROTOCOL_VERSION};
pub use spot::{BoardSpec, RangeSpec, Sizing, Spot, SpotError, Stop, StreetSizing};

/// One request line.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Request {
    /// Echoed back verbatim so a pipelining host can match responses to requests.
    #[serde(default)]
    pub id: Option<serde_json::Value>,
    #[serde(flatten)]
    pub command: Command,
}

/// What went wrong, in a form the host can branch on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ErrorBody {
    /// Stable discriminant: `bad_request`, `no_such_job`, `not_readable`, `bad_node`, `engine`,
    /// `nothing_to_save`, `not_recoverable`, `not_finished`, `serialize`, or `panic`.
    pub code: String,
    pub message: String,
}

/// One response line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Response {
    /// The request's `id`, or `null` when the request was too malformed to have one.
    pub id: Option<serde_json::Value>,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBody>,
}

impl Response {
    #[must_use]
    pub fn ok(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    #[must_use]
    pub fn err(id: Option<serde_json::Value>, code: &str, message: impl Into<String>) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(ErrorBody {
                code: code.to_owned(),
                message: message.into(),
            }),
        }
    }
}

/// A response plus whether the session should end after sending it.
#[derive(Debug, Clone, PartialEq)]
pub struct Handled {
    pub response: Response,
    pub stop: bool,
}

/// A live session: a job queue plus the frame sink both it and the command loop write to.
#[derive(Debug)]
pub struct Session {
    jobs: Jobs,
    emit: Arc<dyn Emit>,
}

impl Session {
    /// Start a session, with its worker thread.
    #[must_use]
    pub fn new(emit: Arc<dyn Emit>) -> Self {
        Self {
            jobs: Jobs::new(Arc::clone(&emit)),
            emit,
        }
    }

    /// The queue behind this session, for a host embedding it rather than piping to it.
    #[must_use]
    pub const fn jobs(&self) -> &Jobs {
        &self.jobs
    }

    /// Handle one request line. Never panics and never returns `Err`.
    #[must_use]
    pub fn handle_line(&self, line: &str) -> Handled {
        self.handle_line_with(line, |cmd| protocol::execute(&self.jobs, cmd))
    }

    /// [`Session::handle_line`], with the executor injected.
    ///
    /// Public because it is the seam: it lets the panic-containment path be tested without
    /// shipping a command whose only purpose is to panic, and it lets a host that embeds the
    /// session wrap or extend the command set without reimplementing the framing, the id
    /// echoing, or the catch-unwind.
    #[must_use]
    pub fn handle_line_with<F>(&self, line: &str, exec: F) -> Handled
    where
        F: FnOnce(Command) -> Result<serde_json::Value, OpError>,
    {
        let request: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return Handled {
                    response: Response::err(None, "bad_request", e.to_string()),
                    stop: false,
                }
            }
        };

        let id = request.id.clone();
        let stop = matches!(request.command, Command::Shutdown);

        // The engine is full of `unsafe` in its hot loops and panics rather than erroring on
        // several kinds of misuse. A panic that killed the session would take the queue and every
        // solved tree in memory with it.
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| exec(request.command)));

        let response = match outcome {
            Ok(Ok(result)) => Response::ok(id, result),
            Ok(Err(e)) => Response::err(id, e.code(), e.to_string()),
            Err(payload) => Response::err(id, "panic", describe_panic(&*payload)),
        };

        if stop {
            self.jobs.shutdown();
        }

        Handled { response, stop }
    }

    /// Read requests until EOF or `shutdown`, emitting every response through the sink.
    ///
    /// Blank lines are ignored, so a host that terminates its writes with an extra newline does
    /// not get a spurious error back.
    ///
    /// Framing failures are commands failing, not the session failing, per the contract above:
    /// a line that is not UTF-8 is answered with `bad_request` (lossy-decoded JSON does not
    /// parse) rather than ending the loop the way `BufRead::lines` would, and a line longer
    /// than [`MAX_LINE_BYTES`] is discarded and answered with `bad_request` rather than growing
    /// a buffer without bound while a broken host never sends its newline.
    ///
    /// # Errors
    ///
    /// Only for I/O on the input stream itself.
    pub fn run<R: BufRead>(&self, mut input: R) -> std::io::Result<()> {
        let mut buf = Vec::new();
        loop {
            match read_frame(&mut input, &mut buf)? {
                Frame::Eof => break,
                Frame::Oversized => {
                    let response = Response::err(
                        None,
                        "bad_request",
                        format!("request line exceeds {MAX_LINE_BYTES} bytes and was discarded"),
                    );
                    self.emit.emit(encode(&response));
                }
                Frame::Line => {
                    let line = String::from_utf8_lossy(&buf);
                    if line.trim().is_empty() {
                        continue;
                    }
                    let handled = self.handle_line(&line);
                    self.emit.emit(encode(&handled.response));
                    if handled.stop {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The longest request line [`Session::run`] will buffer.
///
/// Generous — the largest legitimate command is a `solve` whose ranges are spelled out combo by
/// combo, which is kilobytes — while still bounding what a host that never sends a newline can
/// make this process hold.
pub const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

/// One framing step of [`Session::run`].
enum Frame {
    /// `buf` holds a complete line, without its trailing newline.
    Line,
    /// The line outgrew [`MAX_LINE_BYTES`]; it has been consumed and discarded.
    Oversized,
    Eof,
}

/// Read one newline-terminated frame into `buf`, enforcing [`MAX_LINE_BYTES`].
fn read_frame<R: BufRead>(input: &mut R, buf: &mut Vec<u8>) -> std::io::Result<Frame> {
    buf.clear();
    let read = Read::take(&mut *input, MAX_LINE_BYTES as u64 + 1).read_until(b'\n', buf)?;
    if read == 0 {
        return Ok(Frame::Eof);
    }

    if buf.last() == Some(&b'\n') {
        // The take limit is one over the cap, so a line whose newline arrived holds at most
        // `MAX_LINE_BYTES` of content.
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
        return Ok(Frame::Line);
    }

    if buf.len() <= MAX_LINE_BYTES {
        // EOF fell in the middle of a line; treat what arrived as the final frame, the same
        // way `BufRead::lines` does.
        return Ok(Frame::Line);
    }

    // Too long and still no newline: swallow the rest of the line in bounded chunks.
    let mut chunk = Vec::with_capacity(8 * 1024);
    loop {
        chunk.clear();
        let read = Read::take(&mut *input, 8 * 1024).read_until(b'\n', &mut chunk)?;
        if read == 0 || chunk.last() == Some(&b'\n') {
            return Ok(Frame::Oversized);
        }
    }
}

pub(crate) fn describe_panic(payload: &(dyn std::any::Any + Send)) -> String {
    let detail = payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown payload".to_owned());
    format!("internal panic: {detail}")
}

/// Serialise a response to a single line.
///
/// Infallible in practice — the payload came out of `serde_json` in the first place — but if it
/// somehow is not, the host still gets well-formed JSON rather than a truncated line.
#[must_use]
pub fn encode(response: &Response) -> String {
    serde_json::to_string(response).unwrap_or_else(|e| {
        let fallback = Response::err(
            response.id.clone(),
            "serialize",
            format!("response could not be encoded: {e}"),
        );
        serde_json::to_string(&fallback)
            .unwrap_or_else(|_| r#"{"id":null,"ok":false,"error":{"code":"serialize","message":"response could not be encoded"}}"#.to_owned())
    })
}

/// An [`Emit`] that writes each line to a stream, flushing every time.
///
/// Flushed per frame because the whole point of pushed progress is that it arrives while the
/// solve is still running; a buffered frame is a frame that has not happened.
///
/// The mutex is what makes it safe for the worker thread and the command loop to share one
/// stdout. Contention is nil: frames are small and rare next to the work between them.
pub struct StreamEmit<W: Write + Send> {
    out: Mutex<W>,
}

impl<W: Write + Send> std::fmt::Debug for StreamEmit<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamEmit").finish_non_exhaustive()
    }
}

impl<W: Write + Send> StreamEmit<W> {
    pub const fn new(out: W) -> Self {
        Self {
            out: Mutex::new(out),
        }
    }
}

impl<W: Write + Send> Emit for StreamEmit<W> {
    fn emit(&self, line: String) {
        // A broken pipe means the host is gone; there is nowhere to report that to, and the
        // command loop will find out on its next read.
        let Ok(mut out) = self.out.lock() else { return };
        let _ = out.write_all(line.as_bytes());
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}

/// An [`Emit`] that collects lines in memory. For tests.
#[derive(Debug, Default)]
pub struct Collector {
    lines: Mutex<Vec<String>>,
}

impl Collector {
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().unwrap().clone()
    }

    /// Every collected line parsed as JSON.
    #[must_use]
    pub fn frames(&self) -> Vec<serde_json::Value> {
        self.lines()
            .iter()
            .map(|l| serde_json::from_str(l).expect("every emitted line is one JSON object"))
            .collect()
    }

    /// Only the pushed job events, in order.
    #[must_use]
    pub fn events(&self) -> Vec<serde_json::Value> {
        self.frames()
            .into_iter()
            .filter(|f| f.get("event").is_some())
            .collect()
    }
}

impl Emit for Collector {
    fn emit(&self, line: String) {
        self.lines.lock().unwrap().push(line);
    }
}
