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
//! The check needs git history, and there are two honest reasons it may be unavailable: a source
//! tarball with no repository, and a shallow clone that does not contain the named commit. Both
//! skip loudly rather than pass quietly — a test that silently becomes a no-op is how this class
//! of guarantee rots.

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

fn git(args: &[&str]) -> Option<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(repo_root())
        .output()
        .ok()
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

    // A shallow clone can lack the named commit entirely, and `git diff` against a missing object
    // reports a difference — which would fail this test for a reason that has nothing to do with
    // the engine. CI fetches full history for exactly this step; anywhere else, skip.
    let known = git(&["cat-file", "-e", &format!("{ENGINE_REV}^{{commit}}")])
        .is_some_and(|o| o.status.success());
    if !known {
        eprintln!(
            "SKIPPED: {ENGINE_REV} is not in this clone — shallow checkout? \
             (`git fetch --unshallow` to run this check)"
        );
        return;
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
