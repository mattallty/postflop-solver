//! The wire protocol: commands in, one response each, plus pushed job events.
//!
//! A deliberately ordinary envelope — `{"id":…,"ok":…,"result":…}` — with one addition that is
//! not ordinary: this session also pushes **unsolicited frames**, because a solve runs for
//! minutes and a protocol that could only answer questions would force the host to poll:
//!
//! ```text
//! -> {"id":1,"cmd":"solve","spot":{"oop":"QQ+","ip":"QQ+","board":"Td9d6h","pot":100,
//!                                  "effectiveStack":300}}
//! <- {"id":1,"ok":true,"result":{"jobId":1,"phase":"queued",…}}
//! <- {"event":"job","job":{"jobId":1,"phase":"running","iterations":40,…}}
//! <- {"event":"job","job":{"jobId":1,"phase":"running","iterations":120,…}}
//! -> {"id":2,"cmd":"cancel","jobId":1}
//! <- {"id":2,"ok":true,"result":{"jobId":1,…}}
//! <- {"event":"job","job":{"jobId":1,"phase":"cancelled","iterations":134,…}}
//! ```
//!
//! **Events are told apart from responses by the `event` key, never by the absence of `id`.** A
//! host that ignores frames carrying `event` degrades to polling `progress` and still works.
//!
//! Field names are `camelCase` throughout, because they are read by TypeScript.

use serde::{Deserialize, Serialize};

use crate::engine::MemoryEstimate;
use crate::jobs::{JobError, JobId, JobStatus, Jobs, NodeView};
use crate::spot::Spot;

/// Bumped when a change to the request or response shape is not backward compatible.
pub const PROTOCOL_VERSION: u32 = 1;

/// The commit of the engine this binary links.
///
/// Reported by `version` so a saved solution can be traced to the engine that produced it, and
/// checked by a host against [`ENGINE_COMPATIBLE_REVS`] before it offers to reopen a stored tree.
///
/// # How this is kept honest
///
/// It used to be honest by construction: the sidecar lived in another repository and named the
/// engine as a git dependency, so this string and the `rev =` in its manifest were the same
/// commit or the build did not resolve. Sharing a repository with the engine removed that check —
/// a path dependency has no revision to disagree with — and left a bare `&'static str` that any
/// change to the engine could quietly falsify.
///
/// `tests/engine_rev.rs` puts the check back: it fails when the engine's `src/` has moved since
/// the commit named here. So changing the engine forces a decision rather than allowing an
/// oversight — update this to the new commit, and add the old value to
/// [`ENGINE_COMPATIBLE_REVS`] if solutions written by it are still readable.
///
/// The test deliberately ignores changes to *this* directory. The sidecar is expected to move on
/// its own, and bumping the engine revision for a protocol change would be a lie in the other
/// direction — a host that keys "can I open this file?" on an exact match would mark a whole
/// library of saved solves unreadable for a change that never touched the engine.
pub const ENGINE_REV: &str = "5e3de32ad2cf848b6a33db4a2a806e5f8dca7f51";

/// The version string the engine stamps into every solution it writes.
///
/// The engine keeps this private (`VERSION_STR` in its `game/serialization.rs`) and refuses to
/// decode a file carrying any other value, so it — not [`ENGINE_REV`] — is what actually decides
/// whether a stored tree can be opened. It is duplicated here because it has to be, and
/// `solution_files_carry_the_format_version_we_claim` reads it back out of a real file so the
/// duplicate cannot drift in silence.
pub const ENGINE_FORMAT: &str = "2023-03-19";

/// Engine revisions whose saved solutions this build can open.
///
/// Keying "can I still read this file?" on an exact revision match is wrong, and expensively so:
/// the pin moves for reasons that have nothing to do with the file format — a lint fix, a new
/// method — and every such bump would otherwise mark a library of stored trees unreadable while
/// they in fact load perfectly. So the claim is made explicitly, per revision, and each entry has
/// to be *earned*: same [`ENGINE_FORMAT`], same storage layout, verified by writing the same spot
/// with both builds and comparing the files.
///
/// This list is why the pin can move at all — it has moved three times in two days, never once for
/// a reason that touched the format. Every earlier revision stays on it, so a tree saved before any
/// of those bumps is still offered rather than greyed out:
///
/// - `b97e0bd` — [PR #3](https://github.com/mattallty/postflop-solver/pull/3): docs, CI and the
///   MSRV declaration. `5e3de32` ([PR #6](https://github.com/mattallty/postflop-solver/pull/6))
///   sits on top of it and touches only `visit` and its tests.
/// - `7c64831` — [PR #2](https://github.com/mattallty/postflop-solver/pull/2): tree visitor,
///   `free_memory`, CI lints.
/// - `6f485ef` — [PR #1](https://github.com/mattallty/postflop-solver/pull/1): the bincode 2.0
///   migration. What every solution saved before 2026-08-02 was written by.
///
/// Verified 2026-08-02, for each of the four: the same spot saved by every pair of builds is
/// byte-identical, and each opens the others' files with identical node output.
pub const ENGINE_COMPATIBLE_REVS: &[&str] = &[
    ENGINE_REV,
    "b97e0bd464a8297c6476cad63b1ddb792d69bc34",
    "7c64831363519d9e34db7589ee2d8f20367801e8",
    "6f485efcbc08744c3875748d7e3750a773e1075d",
];

/// A request from the host. Internally tagged on `cmd`, so a request is one flat object.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "cmd")]
pub enum Command {
    /// Queue a spot. Answers immediately with the queued job; the solving happens after.
    #[serde(rename = "solve", rename_all = "camelCase")]
    Solve { spot: Box<Spot> },
    /// How big would this tree be? Builds it, does not allocate or solve it.
    #[serde(rename = "estimate", rename_all = "camelCase")]
    Estimate { spot: Box<Spot> },
    /// Current status of one job — the poll-shaped twin of the pushed `job` event.
    #[serde(rename = "progress", rename_all = "camelCase")]
    Progress { job_id: JobId },
    /// Ask a job to stop. Returns without waiting for it to notice.
    #[serde(rename = "cancel", rename_all = "camelCase")]
    Cancel { job_id: JobId },
    /// Every job this session has seen.
    #[serde(rename = "jobs")]
    JobList,
    /// Read one node of a finished job's strategy.
    #[serde(rename = "node", rename_all = "camelCase")]
    Node {
        job_id: JobId,
        /// Action indices from the root; at a chance node the index is the dealt card's id.
        #[serde(default)]
        history: Vec<usize>,
    },
    /// Write a finished job's solution to disk.
    #[serde(rename = "save", rename_all = "camelCase")]
    Save { job_id: JobId, path: String },
    /// Hand back the memory a finished job's tree is holding, keeping the row.
    #[serde(rename = "release", rename_all = "camelCase")]
    Release { job_id: JobId },
    /// Remove a finished job entirely — its tree and its row. The deliberate-discard
    /// counterpart to `release`, and the only way to free a tree that was never saved.
    /// Responds with the removed job's final status; afterwards the id answers `no_such_job`.
    #[serde(rename = "forget", rename_all = "camelCase")]
    Forget { job_id: JobId },
    /// Read a solution back as a new, already-finished job.
    #[serde(rename = "open", rename_all = "camelCase")]
    Open {
        path: String,
        /// Refuse to load a file whose tree needs more than this. Absent means
        /// [`crate::DEFAULT_MEMORY_LIMIT`] — the same refusal-over-OOM-kill contract the
        /// `solve` command's `maxMemoryBytes` provides.
        #[serde(default)]
        max_memory_bytes: Option<u64>,
    },
    /// Liveness check.
    #[serde(rename = "ping")]
    Ping,
    /// Crate, protocol and engine versions.
    #[serde(rename = "version")]
    Version,
    /// End the session. The response is sent before the process exits.
    #[serde(rename = "shutdown")]
    Shutdown,
}

/// Version information, for diagnosing a stale sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionResult {
    pub version: String,
    pub protocol_version: u32,
    /// Commit of the engine this binary links.
    pub engine_rev: String,
    /// The engine's own solution-format stamp. See [`ENGINE_FORMAT`].
    pub engine_format: String,
    /// Every revision whose saved solutions this build can open, including [`Self::engine_rev`].
    ///
    /// A host deciding whether a stored `savedTo` path is still worth offering asks whether the
    /// revision recorded against it is in *this* list, not whether it equals `engineRev`.
    pub engine_compatible_revs: Vec<String>,
}

/// Why an operation failed.
///
/// Every variant is reportable to the host as JSON. Nothing here ends the session.
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error(transparent)]
    Job(#[from] JobError),
    #[error("could not serialise the result: {0}")]
    Serialize(#[from] serde_json::Error),
}

impl OpError {
    /// Stable, machine-readable discriminant. The host switches on this, never on the message.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Job(e) => e.code(),
            Self::Serialize(_) => "serialize",
        }
    }
}

/// Run one command against a job queue.
///
/// # Errors
///
/// If the command names an unknown job, describes an invalid spot, or touches the filesystem and
/// fails. Never for a reason that should end the session.
pub fn execute(jobs: &Jobs, command: Command) -> Result<serde_json::Value, OpError> {
    let value = match command {
        Command::Solve { spot } => json(&jobs.submit(*spot)?)?,
        Command::Estimate { spot } => {
            let estimate: MemoryEstimate =
                crate::engine::estimate(&spot).map_err(JobError::Engine)?;
            json(&estimate)?
        }
        Command::Progress { job_id } => json(&jobs.status(job_id)?)?,
        Command::Cancel { job_id } => json(&jobs.cancel(job_id)?)?,
        Command::JobList => {
            let list: Vec<JobStatus> = jobs.list();
            json(&list)?
        }
        Command::Node { job_id, history } => {
            let view: NodeView = jobs.node(job_id, &history)?;
            json(&view)?
        }
        Command::Save { job_id, path } => json(&jobs.save(job_id, &path)?)?,
        Command::Release { job_id } => json(&jobs.release(job_id)?)?,
        Command::Forget { job_id } => json(&jobs.forget(job_id)?)?,
        Command::Open {
            path,
            max_memory_bytes,
        } => json(&jobs.open(&path, max_memory_bytes)?)?,
        Command::Ping => serde_json::json!({ "pong": true }),
        Command::Version | Command::Shutdown => json(&version())?,
    };
    Ok(value)
}

fn json<T: Serialize>(value: &T) -> Result<serde_json::Value, OpError> {
    Ok(serde_json::to_value(value)?)
}

/// Version of this sidecar.
#[must_use]
pub fn version() -> VersionResult {
    VersionResult {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        protocol_version: PROTOCOL_VERSION,
        engine_rev: ENGINE_REV.to_owned(),
        engine_format: ENGINE_FORMAT.to_owned(),
        engine_compatible_revs: ENGINE_COMPATIBLE_REVS
            .iter()
            .map(|&r| r.to_owned())
            .collect(),
    }
}
