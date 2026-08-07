//! `ENGINE_REV` has to name the engine this binary actually links.
//!
//! # Why this test exists
//!
//! It used to be impossible to get wrong. The sidecar lived in another repository and named the
//! engine as a git dependency, so `ENGINE_REV` and the `rev =` in its manifest were the same
//! commit or the build did not resolve. Moving the sidecar in here removed that check — a path
//! dependency has no revision to disagree with — and left a bare `&'static str` that any change
//! to the engine could quietly make false.
//!
//! What makes it worth checking rather than shrugging at: the revision is written into every
//! saved solution and reported over the wire, and a host uses it to decide whether a stored tree
//! was produced by an engine it can still trust. A stale value does not fail, it *misattributes* —
//! the worst kind of wrong, because everything keeps working.
//!
//! # What is checked, and what is deliberately not
//!
//! Only whether `src/` — the engine — has moved since the named commit. Not whether *this*
//! directory has: the sidecar is expected to change on its own, and bumping the engine revision
//! for a change to the protocol would be a lie in the other direction.
//!
//! The check needs git history, and there are honest reasons it may be unavailable: a source
//! tarball with no repository, or a shallow clone that never fetched the named commit. Those skip
//! loudly rather than pass quietly. But an absent commit is only excusable when the history is
//! incomplete — in a clone with *full* history there is nowhere left for the commit to be hiding,
//! and the pin itself is wrong (a typo, or a hash that was never real). That case fails. It used
//! to skip, and a fabricated hash nearly shipped that way in August 2026.

use std::path::{Path, PathBuf};
use std::process::Command;

use pkwiz_solver::protocol::{ENGINE_COMPATIBLE_REVS, ENGINE_REV};

/// The repository root: this crate's directory, one up.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the sidecar crate always has a parent directory")
        .to_path_buf()
}

fn git_in(dir: &Path, args: &[&str]) -> Option<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()
}

fn git(args: &[&str]) -> Option<std::process::Output> {
    git_in(&repo_root(), args)
}

/// Whether a commit exists in a repository — and, when it does not, whose fault that is.
///
/// The distinction matters because the two absences deserve opposite responses: a shallow clone
/// is a property of how the checkout was fetched and says nothing about the pin, while a full
/// clone that lacks the commit can only mean the pin names something that never existed here.
#[derive(Debug, PartialEq, Eq)]
enum RevPresence {
    /// The commit is in the object store; the real check can proceed.
    Present,
    /// Absent, but the clone is shallow: the commit may well exist upstream and simply was
    /// never fetched. Excusable.
    MissingFromShallowClone,
    /// Absent from a clone with full history: a wrong pin, not a fetch problem.
    MissingFromFullClone,
    /// Absent, and git could not say whether the clone is shallow (`--is-shallow-repository`
    /// needs git ≥ 2.15). Excused, reluctantly — better a loud skip than a false alarm.
    MissingShallownessUnknown,
}

fn rev_presence(repo: &Path, rev: &str) -> RevPresence {
    let present = git_in(repo, &["cat-file", "-e", &format!("{rev}^{{commit}}")])
        .is_some_and(|o| o.status.success());
    if present {
        return RevPresence::Present;
    }

    let shallow = git_in(repo, &["rev-parse", "--is-shallow-repository"])
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    match shallow.as_deref() {
        Some("true") => RevPresence::MissingFromShallowClone,
        Some("false") => RevPresence::MissingFromFullClone,
        _ => RevPresence::MissingShallownessUnknown,
    }
}

#[test]
fn engine_rev_is_a_full_commit_hash() {
    // A short hash would still compare equal to itself and still be written into files, so the
    // shape is worth pinning independently of whether the commit exists.
    assert_eq!(
        ENGINE_REV.len(),
        40,
        "ENGINE_REV should be a full 40-character hash"
    );
    assert!(
        ENGINE_REV
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "ENGINE_REV should be lowercase hex: {ENGINE_REV}"
    );
}

#[test]
fn engine_rev_is_the_first_compatible_rev() {
    // A build must always be able to read what it writes. If this ever failed, `readableNow` on
    // the host side would call a freshly written solution unopenable.
    assert_eq!(
        ENGINE_COMPATIBLE_REVS.first(),
        Some(&ENGINE_REV),
        "the running revision must head the compatibility list"
    );
}

#[test]
fn engine_rev_names_the_engine_source_this_binary_links() {
    let Some(inside) = git(&["rev-parse", "--is-inside-work-tree"]) else {
        eprintln!("SKIPPED: git is not available");
        return;
    };
    if !inside.status.success() {
        eprintln!("SKIPPED: not a git checkout (a source tarball, most likely)");
        return;
    }

    match rev_presence(&repo_root(), ENGINE_REV) {
        RevPresence::Present => {}
        RevPresence::MissingFromShallowClone => {
            // `git diff` against a missing object reports a difference, which would fail this
            // test for a reason that has nothing to do with the engine. CI fetches full history
            // for exactly this step; a shallow clone anywhere else gets a loud skip.
            eprintln!(
                "SKIPPED: {ENGINE_REV} is not in this shallow clone \
                 (`git fetch --unshallow` to run this check)"
            );
            return;
        }
        RevPresence::MissingShallownessUnknown => {
            eprintln!(
                "SKIPPED: {ENGINE_REV} is not in this clone, and this git is too old to say \
                 whether the clone is shallow (git ≥ 2.15 to run this check)"
            );
            return;
        }
        RevPresence::MissingFromFullClone => panic!(
            "ENGINE_REV ({ENGINE_REV}) does not exist in this repository.\n\n\
             This clone has full history, so the commit is not merely unfetched — the pin \
             itself is wrong: a typo, or a hash that was never real. ENGINE_REV is written \
             into every saved solution and reported over the wire; a pin that names nothing \
             misattributes every one of them. Set it to a commit that actually exists and \
             names the engine source this binary links."
        ),
    }

    let changed = git(&["diff", "--quiet", ENGINE_REV, "HEAD", "--", "src/"])
        .is_some_and(|o| !o.status.success());

    if changed {
        let files = git(&["diff", "--name-only", ENGINE_REV, "HEAD", "--", "src/"])
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_default();
        let head = git(&["rev-parse", "HEAD"])
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_default();

        panic!(
            "the engine's `src/` has changed since ENGINE_REV ({ENGINE_REV}).\n\n\
             Changed:\n{files}\n\n\
             ENGINE_REV is written into every saved solution and reported over the wire, so \
             leaving it behind misattributes those files to an engine that did not produce them.\n\n\
             Set it to {head} — and then decide, deliberately, whether solutions written by \
             {ENGINE_REV} can still be opened by this build. If they can, add it to \
             ENGINE_COMPATIBLE_REVS; earn that entry by writing the same spot with both builds and \
             comparing the bytes, which is what every existing entry in that list did."
        );
    }
}

// ---------------------------------------------------------------------------
// Tests of the test: `rev_presence` is the sensor that decides between "excusable absence" and
// "wrong pin", and it is exactly the part that silently mis-fired before — a fabricated hash in
// a full clone used to be indistinguishable from a shallow checkout. These exercise both sides
// of that line against scratch repositories, so the distinction cannot rot unnoticed.
// ---------------------------------------------------------------------------

/// Forty hex characters that no repository has ever hashed to.
const FABRICATED_REV: &str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

/// A throwaway repository with two commits, so a `--depth 1` clone of it is genuinely shallow
/// (a single-commit clone cuts no parents and git does not mark it shallow at all).
/// Returns `None` when git itself is unavailable — the same skip the real check takes.
fn scratch_repo(name: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("pkwiz-engine-rev-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok()?;

    git_in(&dir, &["init", "-q"]).filter(|o| o.status.success())?;
    for msg in ["one", "two"] {
        git_in(
            &dir,
            &[
                "-c",
                "user.name=engine-rev-test",
                "-c",
                "user.email=engine-rev-test@invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                msg,
            ],
        )
        .filter(|o| o.status.success())?;
    }
    Some(dir)
}

#[test]
fn a_fabricated_rev_in_a_full_clone_is_a_wrong_pin() {
    let Some(repo) = scratch_repo("full") else {
        eprintln!("SKIPPED: git is not available");
        return;
    };
    // This is the case that nearly shipped: full history, made-up hash. It must read as a wrong
    // pin — anything else and the main test would print SKIPPED and pass again.
    assert_eq!(
        rev_presence(&repo, FABRICATED_REV),
        RevPresence::MissingFromFullClone
    );
    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn a_missing_rev_in_a_shallow_clone_stays_excused() {
    let Some(origin) = scratch_repo("shallow-origin") else {
        eprintln!("SKIPPED: git is not available");
        return;
    };
    let clone = std::env::temp_dir().join(format!(
        "pkwiz-engine-rev-shallow-clone-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&clone);
    // `--depth` is silently ignored for plain local paths; the file:// scheme forces a real
    // fetch negotiation, which is what leaves the clone marked shallow.
    let cloned = git_in(
        std::env::temp_dir().as_path(),
        &[
            "clone",
            "-q",
            "--depth",
            "1",
            &format!("file://{}", origin.display()),
            clone.to_str().expect("temp paths are valid UTF-8"),
        ],
    )
    .is_some_and(|o| o.status.success());
    assert!(cloned, "shallow-cloning a local scratch repository failed");

    assert_eq!(
        rev_presence(&clone, FABRICATED_REV),
        RevPresence::MissingFromShallowClone
    );

    let _ = std::fs::remove_dir_all(&origin);
    let _ = std::fs::remove_dir_all(&clone);
}

#[test]
fn a_commit_that_exists_reads_as_present() {
    let Some(repo) = scratch_repo("present") else {
        eprintln!("SKIPPED: git is not available");
        return;
    };
    let head = git_in(&repo, &["rev-parse", "HEAD"])
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .expect("a freshly committed scratch repository has a HEAD");
    assert_eq!(rev_presence(&repo, &head), RevPresence::Present);
    let _ = std::fs::remove_dir_all(&repo);
}
