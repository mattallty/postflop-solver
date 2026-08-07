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
    compute_current_ev, compute_exploitability, finalize, solve_step, ActionTree, BunchingData,
    CardConfig, IcmConfig, PostFlopGame, NOT_DEALT,
};
use serde::{Deserialize, Serialize};

use crate::convert::{to_engine_card, to_engine_range};
use crate::spot::{BunchingSpec, Spot, SpotError, Validated};

/// Refuse a tree bigger than this unless the caller raises it.
///
/// Four gibibytes is a big flop tree with several bet sizes — enough that hitting the limit means
/// the configuration is genuinely ambitious rather than that the limit is mean. The point is that
/// an ambitious tree gets a message instead of an OOM kill.
pub const DEFAULT_MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

/// Peak live bytes of a bunching preparation, indexed by fold-player count minus one.
///
/// The engine allocates nothing until the phases run, so the only way to refuse a preparation
/// that would not fit is to know its peak in advance. Hand-derived from the `COMB_49_k`
/// constants at the top of the engine's `src/bunching.rs` (choose k of the 49 cards outside the
/// flop) and its documented phase schedule:
///
/// - The results, allocated by `phase3_prepare` and kept for the life of the data:
///   4·(C(49,4) + C(49,5) + C(49,6)) = 4·(211 876 + 1 906 884 + 13 983 816) = 64 410 304 bytes
///   of `f32`.
/// - The `f64` sum tables, allocated by `phase2_prepare` for dead-card counts `0..=2n` and
///   freed only when phase 3 completes, so they are live alongside the results:
///   8·Σ C(49,k) for k `0..=2n` — 9 808 (n=1), 1 852 208 (n=2), 128 977 808 (n=3).
/// - Four fold players peak earlier, during phase 2: `temp_table3` at 8·C(49,8) =
///   3 607 824 528 bytes, plus the sum tables for `0..=6` — the table above C(49,6) is never
///   allocated as a sum.
/// - Plus an 8 MiB allowance for what the derivation deliberately ignores: the struct itself,
///   the fold ranges, and the small phase-1 temporaries (about 6 KB observed for one fold
///   player) — generous against those, immaterial against the 62 MB result.
///
/// If the engine ever reschedules its phases these go stale, which is why the worker logs (in
/// debug builds, and never fails a job) whenever live usage exceeds the claim.
pub const PEAK_BUNCHING_BYTES: [u64; 4] = {
    const ALLOWANCE: u64 = 8 * 1024 * 1024;
    [
        64_410_304 + 9_808 + ALLOWANCE, // 1 fold player: results + sums 0..=2
        64_410_304 + 1_852_208 + ALLOWANCE, // 2: results + sums 0..=4
        64_410_304 + 128_977_808 + ALLOWANCE, // 3: results + sums 0..=6
        3_607_824_528 + 128_977_808 + ALLOWANCE, // 4: phase-2 temp table + sums 0..=6
    ]
};

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
    #[error("locks[{index}] cannot be applied: {reason}")]
    Lock { index: usize, reason: String },
    #[error("preparing bunching data for {fold_players} fold players peaks at {peak} bytes and the limit is {limit}; raise `maxMemoryBytes` or drop a fold range")]
    BunchingTooBig {
        fold_players: usize,
        peak: u64,
        limit: u64,
    },
}

/// How much memory a tree would take, both ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEstimate {
    /// 32-bit floats.
    pub uncompressed: u64,
    /// 16-bit integers with a shared scale.
    pub compressed: u64,
    /// What the built game actually allocated. Includes [`Self::bunching_extra`] when present.
    pub allocated: u64,
    /// Game-side arena bytes the bunching effect adds, on top of the tree itself. Present
    /// exactly when the spot asks for bunching; computable from the tree alone, so `estimate`
    /// reports it without ever touching the referenced data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bunching_extra: Option<u64>,
}

/// Build the game for a spot, allocate its storage, and pin any requested strategies.
///
/// # Errors
///
/// If the spot is invalid, the tree cannot be built, it would need more memory than the spot
/// allows, or a lock cannot be applied to the tree that was built.
pub fn build(spot: &Spot) -> Result<(PostFlopGame, MemoryEstimate), EngineError> {
    build_with_bunching(spot, None)
}

/// [`build`], with bunching data to apply between the tree build and the allocation — the
/// engine's canonical ordering (`with_config` → `set_bunching_effect` → `allocate_memory`).
///
/// The caller resolves the spot's [`crate::spot::BunchingRef`] to actual data; this function
/// only applies it. The `TooBig` check inside `construct` already counted the bunching arena
/// (the spot says whether bunching is coming, and the arena size needs no data), so a game that
/// gets here fits, arena included.
///
/// # Errors
///
/// As [`build`], plus the engine's refusals of the data itself: not ready, a flop that does not
/// match the spot's, or fold ranges that leave no valid combination against this board.
pub fn build_with_bunching(
    spot: &Spot,
    bunching: Option<&BunchingData>,
) -> Result<(PostFlopGame, MemoryEstimate), EngineError> {
    let (mut game, memory) = construct(spot)?;
    if let Some(data) = bunching {
        game.set_bunching_effect(data).map_err(EngineError::Build)?;
    }
    game.allocate_memory(spot.compress);
    // After allocation (the engine refuses locks before it) and before the CFR loop (it
    // refuses them after `finalize` marks the game solved).
    apply_locks(&mut game, &spot.locks)?;
    Ok((game, memory))
}

/// Build unprocessed bunching data for a spec, refusing one whose preparation would not fit.
///
/// The peak gate has to run *here*, before any phase allocates: the four-fold-player case
/// mallocs 3.6 GB in one go during phase 2, well past the point of refusing politely.
///
/// # Errors
///
/// If the spec is invalid (the engine's own message, wrapped in
/// [`SpotError::Bunching`]) or the preparation's peak exceeds the spec's memory cap.
pub fn validate_bunching(spec: &BunchingSpec) -> Result<BunchingData, EngineError> {
    let data = spec.validate()?;
    let fold_players = data.fold_ranges().len();
    let peak = PEAK_BUNCHING_BYTES[fold_players - 1];
    let limit = spec.max_memory_bytes.unwrap_or(DEFAULT_MEMORY_LIMIT);
    if peak > limit {
        return Err(EngineError::BunchingTooBig {
            fold_players,
            peak,
            limit,
        });
    }
    Ok(data)
}

/// One failed step of a history replay: `history[depth]` asked for `index` where the node
/// offers `available` (for a chance node, "offers" means dealable cards).
pub(crate) struct WalkStep {
    pub depth: usize,
    pub index: usize,
    pub available: usize,
    pub is_chance: bool,
}

/// Replays `history` from the root, validating every step the way the `node` command does:
/// at a chance node the index is the dealt card's id and must be dealable; at a player node it
/// must be within `available_actions`. On success the game is positioned at the node.
pub(crate) fn walk(game: &mut PostFlopGame, history: &[usize]) -> Result<(), WalkStep> {
    game.back_to_root();
    for (depth, &index) in history.iter().enumerate() {
        if game.is_terminal_node() {
            return Err(WalkStep {
                depth,
                index,
                available: 0,
                is_chance: false,
            });
        }
        if game.is_chance_node() {
            let mask = game.possible_cards();
            if index >= 52 || mask & (1u64 << index) == 0 {
                return Err(WalkStep {
                    depth,
                    index,
                    available: mask.count_ones() as usize,
                    is_chance: true,
                });
            }
        } else {
            let available = game.available_actions().len();
            if index >= available {
                return Err(WalkStep {
                    depth,
                    index,
                    available,
                    is_chance: false,
                });
            }
        }
        game.play(index);
    }
    Ok(())
}

/// Pins each lock's strategy at its node. Runs on a built, allocated, not-yet-solved game.
///
/// # Errors
///
/// [`EngineError::Lock`] naming the lock and the reason: a history step that does not exist, a
/// target that is not a decision node, dimensions that do not match the node, or a `hands`
/// guard that disagrees with the acting player's hand list.
fn apply_locks(game: &mut PostFlopGame, locks: &[crate::spot::Lock]) -> Result<(), EngineError> {
    for (index, lock) in locks.iter().enumerate() {
        let err = |reason: String| EngineError::Lock { index, reason };

        walk(game, &lock.history).map_err(|step| {
            err(format!(
                "step {} names {} {}, but that node offers {}",
                step.depth,
                if step.is_chance { "card" } else { "action" },
                step.index,
                step.available,
            ))
        })?;

        if game.is_terminal_node() || game.is_chance_node() {
            return Err(err(
                "the node it reaches is not a decision node; only a player's strategy can be \
                 pinned"
                    .to_owned(),
            ));
        }

        let player = game.current_player();
        let num_actions = game.available_actions().len();
        let num_hands = game.private_cards(player).len();
        if lock.strategy.len() != num_actions || lock.strategy[0].len() != num_hands {
            return Err(err(format!(
                "`strategy` is {}x{} where the node needs {num_actions} actions x {num_hands} \
                 hands",
                lock.strategy.len(),
                lock.strategy.first().map_or(0, Vec::len),
            )));
        }

        if let Some(hands) = &lock.hands {
            for (h, expected) in hands.iter().enumerate() {
                let actual = crate::convert::hole_to_string(game.private_cards(player)[h])
                    .unwrap_or_else(|_| "??".to_owned());
                if *expected != actual {
                    return Err(err(format!(
                        "`hands[{h}]` is `{expected}` but the acting player's hand there is \
                         `{actual}`",
                    )));
                }
            }
        }

        let flat: Vec<f32> = lock.strategy.iter().flatten().copied().collect();
        game.lock_current_strategy(&flat);
    }

    if !locks.is_empty() {
        game.back_to_root();
    }
    Ok(())
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

    let mut action_tree =
        ActionTree::new(validated.tree_config.clone()).map_err(EngineError::Tree)?;
    // Edits before `with_config`, so everything downstream — the memory estimate, the TooBig
    // check, the solve — sees the edited tree without knowing edits exist.
    crate::spot::apply_edits(
        &mut action_tree,
        &validated.added_lines,
        &validated.removed_lines,
    )?;
    let mut game =
        PostFlopGame::with_config(card_config, action_tree).map_err(EngineError::Build)?;

    // Right after the build, before the memory estimate: the effect adds only a small
    // per-terminal-amount table, so the estimate is untouched, and an engine-side refusal
    // (the sidecar's synchronous mirror should have caught it) fails the job with the
    // engine's own message.
    if let Some(icm) = &validated.icm {
        game.set_icm_effect(icm).map_err(EngineError::Build)?;
    }

    let (uncompressed, compressed) = game.memory_usage();
    // The bunching arena's size depends only on the tree, not on the data, so the budget check
    // covers it even though the data has not been resolved yet — and `estimate` can report it
    // for a spot whose preparation is still queued.
    let bunching_extra = spot
        .bunching
        .is_some()
        .then(|| game.memory_usage_bunching());
    let needed = if spot.compress {
        compressed
    } else {
        uncompressed
    } + bunching_extra.unwrap_or(0);
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
            bunching_extra,
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
    /// Expected value of each player at the root, bias-subtracted so it is zero-sum without
    /// rake. On an ICM job this is each player's $ delta versus the street-start ICM baseline
    /// (not zero-sum — see `Spot::icm`).
    pub ev: [f32; 2],
    pub elapsed_ms: u64,
    pub history: Vec<Sample>,
}

/// Run the CFR loop.
///
/// `target` is the absolute exploitability to stop at, precomputed by the caller because its
/// scale is mode-dependent: `stop.target_for(starting_pot)` in chip mode,
/// `stop.target_in(icm_pot_value)` in ICM mode (where exploitability is measured in $).
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
    target: f32,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(u32, f32),
) -> Solved {
    let started = Instant::now();

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
pub(crate) fn push_sample(history: &mut Vec<Sample>, sample: Sample) {
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

/// Write prepared bunching data to disk, in the engine's own container format.
///
/// Same container as a solution (the header's data-type byte tells them apart), so a bunching
/// file refuses to open as a game and vice versa, cleanly.
///
/// # Errors
///
/// If the data is not fully processed, or the file cannot be written.
pub fn save_bunching(
    data: &BunchingData,
    path: &str,
    memo: &str,
    compression_level: Option<i32>,
) -> Result<(), EngineError> {
    postflop_solver::save_data_to_file(data, memo, path, compression_level).map_err(|reason| {
        EngineError::Save {
            path: path.to_owned(),
            reason,
        }
    })
}

/// Re-establish the runtime effects (ICM and/or bunching) on a game loaded from disk —
/// weights, tables *and* EVs.
///
/// `set_icm_effect`/`set_bunching_effect` alone are not enough after a load: the file format
/// omits the stored counterfactual values, and the decoder restores them by re-running
/// `finalize` — necessarily *before* any runtime effect can be applied, so every stored EV on
/// a freshly loaded game is a plain chip-space evaluation of the saved strategy. (Weights and
/// equity are computed live at read time, which is why they alone would look correct — the
/// discrepancy is exactly the kind that survives a casual check.) So after applying the
/// effects, `finalize` runs **once** more with the effect-aware evaluation in place. It only
/// *reads* the strategy — never rewrites it — so the second pass reproduces the pre-save EVs
/// bit for bit; the private `Refinalize` wrapper below is what gets it past the
/// already-solved refusal, the same state dance the decoder itself performs.
///
/// With both `None` this is a no-op — no wasted finalize pass.
///
/// # Errors
///
/// The engine's `set_icm_effect` refusals (a configuration that does not match the loaded
/// tree) and `set_bunching_effect` refusals (data not ready, a flop that does not match, or
/// fold ranges leaving no valid combination).
pub fn reapply_effects(
    game: &mut PostFlopGame,
    icm: Option<&IcmConfig>,
    bunching: Option<&BunchingData>,
) -> Result<(), String> {
    if icm.is_none() && bunching.is_none() {
        return Ok(());
    }
    if let Some(icm) = icm {
        game.set_icm_effect(icm)?;
    }
    if let Some(data) = bunching {
        game.set_bunching_effect(data)?;
    }
    postflop_solver::finalize(&mut Refinalize(game));
    Ok(())
}

/// [`reapply_effects`] for the bunching-only case, kept for call sites and tests that predate
/// ICM.
pub fn reapply_bunching(game: &mut PostFlopGame, data: &BunchingData) -> Result<(), String> {
    reapply_effects(game, None, Some(data))
}

/// A view of a solved [`PostFlopGame`] that [`postflop_solver::finalize`] will accept again.
///
/// Delegates the whole [`Game`] trait except the two solved-state methods: `is_solved` answers
/// `false` so `finalize` runs, and `set_solved` is a no-op because the inner game already is.
struct Refinalize<'a>(&'a mut PostFlopGame);

impl postflop_solver::Game for Refinalize<'_> {
    type Node = postflop_solver::PostFlopNode;

    fn root(&self) -> postflop_solver::MutexGuardLike<'_, Self::Node> {
        self.0.root()
    }

    fn num_private_hands(&self, player: usize) -> usize {
        self.0.num_private_hands(player)
    }

    fn initial_weights(&self, player: usize) -> &[f32] {
        self.0.initial_weights(player)
    }

    fn evaluate(
        &self,
        result: &mut [std::mem::MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        self.0.evaluate(result, node, player, cfreach);
    }

    fn chance_factor(&self, node: &Self::Node) -> usize {
        self.0.chance_factor(node)
    }

    fn is_solved(&self) -> bool {
        false
    }

    fn set_solved(&mut self) {}

    fn is_ready(&self) -> bool {
        true
    }

    fn is_raked(&self) -> bool {
        self.0.is_raked()
    }

    fn is_icm(&self) -> bool {
        self.0.is_icm()
    }

    fn isomorphic_chances(&self, node: &Self::Node) -> &[u8] {
        self.0.isomorphic_chances(node)
    }

    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2] {
        self.0.isomorphic_swap(node, index)
    }

    fn locking_strategy(&self, node: &Self::Node) -> &[f32] {
        self.0.locking_strategy(node)
    }

    fn is_compression_enabled(&self) -> bool {
        self.0.is_compression_enabled()
    }
}

/// Read prepared bunching data back, under the same refusal-over-OOM-kill cap as [`load`].
///
/// # Errors
///
/// If the file is missing, is not bunching data, fails the engine's decode-time validation, or
/// claims more memory than the cap allows.
pub fn load_bunching(
    path: &str,
    max_memory_bytes: Option<u64>,
) -> Result<(BunchingData, String), EngineError> {
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
