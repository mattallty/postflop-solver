//! `pkwiz-solver` — the GTO solver sidecar.
//!
//! Reads newline-delimited JSON commands on stdin and writes newline-delimited JSON on stdout:
//! one response per command, plus `{"event":"job",…}` frames pushed while a solve runs. See
//! [`pkwiz_solver`] for the protocol.
//!
//! stdout carries the protocol and nothing else. Diagnostics go to stderr, so a host can leave
//! stderr attached for debugging without corrupting a single frame.

#![forbid(unsafe_code)]

use std::io::{self, BufReader};
use std::process::ExitCode;
use std::sync::Arc;

use pkwiz_solver::{Session, StreamEmit};

fn main() -> ExitCode {
    // `--version` exists so a host can probe a binary it found on disk without opening a session
    // — and, because this binary is optional, so it can tell "not built" from "wrong build".
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        let v = pkwiz_solver::protocol::version();
        println!(
            "pkwiz-solver {} (protocol {}, engine {})",
            v.version, v.protocol_version, v.engine_rev
        );
        return ExitCode::SUCCESS;
    }

    // Unbuffered on purpose: `StreamEmit` flushes every frame anyway, and a `BufWriter` between
    // the worker thread and the pipe would only add a place for a progress frame to sit.
    let emit = Arc::new(StreamEmit::new(io::stdout()));
    let session = Session::new(emit);

    match session.run(BufReader::new(io::stdin().lock())) {
        Ok(()) => ExitCode::SUCCESS,
        // The host closed the pipe — usually because it is shutting down and did not wait for us.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pkwiz-solver: fatal stream error: {e}");
            ExitCode::FAILURE
        }
    }
}
