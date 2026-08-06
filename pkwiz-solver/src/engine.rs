//! Driving `postflop-solver`: build a tree, run Discounted CFR, stop when told to.
//!
//! The engine ships a `solve()` that does the whole job in one call, prints to stdout and cannot
//! be interrupted. None of those three are acceptable here — a host wants progress on a
//! stream and a cancel button — so this module runs the same loop by hand out of `solve_step`,
//! `compute_exploitability` and `finalize`, which is exactly the decomposition the engine's own
//! `basic.rs` example documents as the supported way to do it.
//!
//! The loop is otherwise faithful to `solve()`: alternating updates, exploitability measured
//! every `check_interval` iterations rather than every iteration, and `finalize()` at the end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use postflop_solver::{
    compute_current_ev, compute_exploitability, finalize, solve_step, ActionTree, CardConfig,
    PostFlopGame, NOT_DEALT,
};
use serde::{Deserialize, Serialize};

use crate::convert::{to_engine_card, to_engine_range};
use crate::spot::{Spot, SpotError, Validated};

/// Refuse a tree bigger than this unless the caller raises it.
///
/// Four gibibytes is a big flop tree with several bet sizes — enough that hitting the limit means
/// the configuration is genuinely ambitious rather than that the limit is mean. The point is that
/// an ambitious tree gets a message instead of an OOM kill.
pub const DEFAULT_MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

/// Why a solve could not be built or run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Spot(#[from] SpotError),
    #[error("the action tree is invalid: {0}")]
    Tree(String),
    #[error("the game could not be built: {0}")]
    Build(String),
    #[error("this tree needs {needed} bytes{compressed} and the limit is {limit}; raise `maxMemoryBytes`, shrink the ranges, or drop a bet size")]
    TooBig {
        needed: u64,
        limit: u64,
        compressed: &'static str,
    },
    #[error("could not write the solution to `{path}`: {reason}")]
    Save { path: String, reason: String },
    #[error("could not read a solution from `{path}`: {reason}")]
    Load { path: String, reason: String },
}

/// How much memory a tree would take, both ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEstimate {
    /// 32-bit floats.
    pub uncompressed: u64,
    /// 16-bit integers with a shared scale.
    pub compressed: u64,
    /// What the built game actually allocated.
    pub allocated: u64,
}

/// Build the game for a spot and allocate its storage.
///
/// # Errors
///
/// If the spot is invalid, the tree cannot be built, or it would need more memory than the spot
/// allows.
pub fn build(spot: &Spot) -> Result<(PostFlopGame, MemoryEstimate), EngineError> {
    let (mut game, memory) = construct(spot)?;
    game.allocate_memory(spot.compress);
    Ok((game, memory))
}

/// How big this spot's tree would be, without allocating it.
///
/// Constructing the tree is cheap next to solving it, so a UI can ask this on every change to the
/// bet sizing and show the memory cost before anyone commits to a solve.
///
/// # Errors
///
/// As [`build`], except that the memory limit is *not* enforced: the whole point is to report a
/// number that may be too large.
pub fn estimate(spot: &Spot) -> Result<MemoryEstimate, EngineError> {
    let mut relaxed = spot.clone();
    relaxed.max_memory_bytes = Some(u64::MAX);
    Ok(construct(&relaxed)?.1)
}

/// Everything up to but not including `allocate_memory`.
fn construct(spot: &Spot) -> Result<(PostFlopGame, MemoryEstimate), EngineError> {
    let validated = spot.validate()?;
    construct_validated(spot, &validated)
}

fn construct_validated(
    spot: &Spot,
    validated: &Validated,
) -> Result<(PostFlopGame, MemoryEstimate), EngineError> {
    let board = &validated.board;
    let card_config = CardConfig {
        range: [
            to_engine_range(&validated.oop).map_err(EngineError::Build)?,
            to_engine_range(&validated.ip).map_err(EngineError::Build)?,
        ],
        // The engine wants the flop sorted; it derives suit isomorphism from the order.
        flop: {
            let mut flop = [
                to_engine_card(board[0]),
                to_engine_card(board[1]),
                to_engine_card(board[2]),
            ];
            flop.sort_unstable();
            flop
        },
        turn: board.get(3).copied().map_or(NOT_DEALT, to_engine_card),
        river: board.get(4).copied().map_or(NOT_DEALT, to_engine_card),
    };

    let action_tree = ActionTree::new(validated.tree_config.clone()).map_err(EngineError::Tree)?;
    let game = PostFlopGame::with_config(card_config, action_tree).map_err(EngineError::Build)?;

    let (uncompressed, compressed) = game.memory_usage();
    let needed = if spot.compress {
        compressed
    } else {
        uncompressed
    };
    let limit = spot.max_memory_bytes.unwrap_or(DEFAULT_MEMORY_LIMIT);
    if needed > limit {
        return Err(EngineError::TooBig {
            needed,
            limit,
            compressed: if spot.compress { " compressed" } else { "" },
        });
    }

    Ok((
        game,
        MemoryEstimate {
            uncompressed,
            compressed,
            allocated: needed,
        },
    ))
}

/// One exploitability measurement, kept so a UI can draw the convergence curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sample {
    pub iteration: u32,
    pub exploitability: f32,
}

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Stopped {
    /// Exploitability reached the target.
    Converged,
    /// The iteration cap was hit first.
    IterationCap,
    /// Someone pressed cancel.
    Cancelled,
}

/// What a finished (or abandoned) solve produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Solved {
    pub stopped: Stopped,
    pub iterations: u32,
    pub exploitability: f32,
    pub target_exploitability: f32,
    /// Expected value of each player at the root, bias-subtracted so it is zero-sum without rake.
    pub ev: [f32; 2],
    pub elapsed_ms: u64,
    pub history: Vec<Sample>,
}

/// Run the CFR loop.
///
/// `on_progress` is called after every exploitability measurement with the iteration count and
/// the value; it is the caller's job to throttle what it does with that.
///
/// Cancellation is checked once per iteration. That is the finest granularity available without
/// forking the engine's inner loop, and it is fine in practice: one iteration is milliseconds on
/// a river tree and a second or two on a big flop tree, so cancel is "prompt" at the scale of a
/// solve that runs for minutes.
///
/// The game is finalized on every exit path, including cancellation. A cancelled solve therefore
/// still has a readable — merely worse — strategy, which is what you want from a stop button:
/// you stopped because you had seen enough, not to throw the work away.
pub fn run(
    game: &mut PostFlopGame,
    stop: &crate::spot::Stop,
    starting_pot: i32,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(u32, f32),
) -> Solved {
    let started = Instant::now();
    let target = stop.target_for(starting_pot);

    let mut exploitability = compute_exploitability(&*game);
    let mut history = vec![Sample {
        iteration: 0,
        exploitability,
    }];
    on_progress(0, exploitability);

    let mut iterations = 0u32;
    let mut stopped = Stopped::IterationCap;

    for t in 0..stop.max_iterations {
        if cancel.load(Ordering::Relaxed) {
            stopped = Stopped::Cancelled;
            break;
        }
        if exploitability <= target {
            stopped = Stopped::Converged;
            break;
        }

        solve_step(&*game, t);
        iterations = t + 1;

        if iterations.is_multiple_of(stop.check_interval) || iterations == stop.max_iterations {
            exploitability = compute_exploitability(&*game);
            push_sample(
                &mut history,
                Sample {
                    iteration: iterations,
                    exploitability,
                },
            );
            on_progress(iterations, exploitability);
        }
    }

    // A loop that fell out of the `for` having reached the target on its last measurement did
    // converge; the `break` above only fires on the following pass, which never happens at the cap.
    if stopped == Stopped::IterationCap && exploitability <= target {
        stopped = Stopped::Converged;
    }

    finalize(game);
    let ev = compute_current_ev(&*game);

    Solved {
        stopped,
        iterations,
        exploitability,
        target_exploitability: target,
        ev,
        elapsed_ms: started.elapsed().as_millis() as u64,
        history,
    }
}

/// How many samples a convergence curve keeps before it starts thinning.
const MAX_SAMPLES: usize = 512;

/// Append a sample, halving the resolution of the whole series whenever it gets too long.
///
/// A 100k-iteration solve measured every ten iterations would otherwise accumulate ten thousand
/// samples, all of which ride on every `progress` response. Decimating keeps the shape of the
/// curve — which is all anyone plots it for — at a bounded cost.
fn push_sample(history: &mut Vec<Sample>, sample: Sample) {
    if history.len() >= MAX_SAMPLES {
        let mut keep = false;
        history.retain(|_| {
            keep = !keep;
            keep
        });
    }
    history.push(sample);
}

/// zstd level used when a caller does not ask for one.
///
/// Three is zstd's own default and the right end of the curve for this shape of data: a solved
/// tree is float arrays that compress well, the encoder in the engine's `file.rs` is already
/// multithreaded, and the levels above three buy single-digit percentages for several times the
/// CPU. The choice that matters is compressing at all — these files are the largest thing the
/// project writes, and until now they were written raw.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Write a solved game to disk, compressed unless told otherwise.
///
/// `compression_level` of `None` writes the tree raw. The format records which was used, so both
/// kinds open with the same `load` and files written before compression existed stay readable.
///
/// # Errors
///
/// If the game is not solved, or the file cannot be written.
pub fn save(
    game: &PostFlopGame,
    path: &str,
    memo: &str,
    compression_level: Option<i32>,
) -> Result<(), EngineError> {
    postflop_solver::save_data_to_file(game, memo, path, compression_level).map_err(|reason| {
        EngineError::Save {
            path: path.to_owned(),
            reason,
        }
    })
}

/// Read a solved game back, refusing one that needs more memory than the cap allows.
///
/// `max_memory_bytes` of `None` means [`DEFAULT_MEMORY_LIMIT`]. An uncapped load would be the
/// one path on which an oversized — or corrupt-header — file gets this process OOM-killed, the
/// exact fate the solve path's `TooBig` refusal exists to prevent.
///
/// # Errors
///
/// If the file is missing, is not a solution, was written by an incompatible engine build, or
/// needs more memory than the cap allows.
pub fn load(
    path: &str,
    max_memory_bytes: Option<u64>,
) -> Result<(PostFlopGame, String), EngineError> {
    let limit = max_memory_bytes.unwrap_or(DEFAULT_MEMORY_LIMIT);
    postflop_solver::load_data_from_file(path, Some(limit)).map_err(|reason| EngineError::Load {
        path: path.to_owned(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimation_keeps_the_ends_and_bounds_the_middle() {
        let mut history = Vec::new();
        for i in 0..5000u32 {
            push_sample(
                &mut history,
                Sample {
                    iteration: i,
                    exploitability: 1.0 / f32::from(u16::try_from(i + 1).unwrap()),
                },
            );
        }
        assert!(history.len() <= MAX_SAMPLES, "{}", history.len());
        assert!(!history.is_empty());
        assert_eq!(history.last().unwrap().iteration, 4999);
        // Still ordered, so a chart drawn from it is not a scribble.
        assert!(history.windows(2).all(|w| w[0].iteration < w[1].iteration));
    }
}
