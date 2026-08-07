//! Aggregation reports, the full-tree dump, and the strategy simplification — the traversal
//! core all three share.
//!
//! All three are worker jobs ([`crate::jobs::JobKind::Report`] / `Dump` / `Simplify`) over a
//! *finished* solve's game, built entirely on the engine's public interpreter API:
//! `apply_history`, `play`, `possible_cards`, `strategy`, and the cached-weight readers. The
//! engine, its revision pin and its file format are untouched.
//!
//! # Reports never traverse the full tree
//!
//! Each report kind is a bounded-node-count design: `runouts` is one row per dealable card at a
//! chance node (≤ 48 nodes, one `play` each), `lines` is the decision nodes of a single street,
//! and `categories` is one node. They still run as jobs because a bunching source multiplies the
//! per-node cost of `cache_normalized_weights`/`equity` from *O*(#hands) to *O*(#OOP × #IP) —
//! tens of seconds on a big preparation — and a synchronous command would block `cancel` and
//! `progress` for the duration.
//!
//! # The dump's constraint is output size, not traversal time
//!
//! A mid-size flop tree holds on the order of 10⁸ strategy floats — a gigabyte of JSON before a
//! single big sizing is added — so everything about [`run_dump`] is shaped to bound output:
//! per-array `include` flags, [`DumpSpec::max_board_cards`] street bounding, optional zstd
//! framing, a [`DumpSpec::max_bytes`] abort, and no terminal lines at all (they are ~60% of all
//! nodes and carry nothing a reader cannot infer). The summary trailer is the completeness
//! marker: a file without one was cancelled, failed, or truncated by a crash.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use postflop_solver::{BoardState, PostFlopGame};
use serde::{Deserialize, Serialize};

use crate::classify::{classify, Draws, MadeHand};
use crate::engine::{self, WalkStep};
use crate::jobs::JobId;

/// Stamped into every report result and dump header, so a host can tell shapes apart later.
pub const FORMAT_VERSION: u32 = 1;

/// What a `report` job computes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportSpec {
    /// Source solve job (a `solve` or `open` job).
    pub job_id: JobId,
    /// Base node, node-command convention: chance step = dealt card id 0–51.
    #[serde(default)]
    pub history: Vec<usize>,
    pub kind: ReportKind,
    /// `runouts` only: decision-action indices replayed after each dealt card before the row is
    /// read. May not contain chance steps.
    #[serde(default)]
    pub line: Vec<usize>,
    /// `runouts`/`lines`: add a per-category breakdown to every row.
    #[serde(default)]
    pub categories: bool,
}

/// The three report shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReportKind {
    /// One row per dealable card at a chance node.
    Runouts,
    /// One row per same-street decision node under a base node.
    Lines,
    /// One decision node, partitioned by hand category.
    Categories,
}

/// Which per-hand arrays a dump writes at each decision node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DumpInclude {
    pub strategy: bool,
    /// Per-hand EVs *and* the per-action `evDetail` matrix.
    pub ev: bool,
    pub equity: bool,
    pub weights: bool,
}

impl Default for DumpInclude {
    fn default() -> Self {
        Self {
            strategy: true,
            ev: false,
            equity: false,
            weights: false,
        }
    }
}

/// What a `dump` job writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DumpSpec {
    /// Source solve job.
    pub job_id: JobId,
    /// Output file. Created/truncated. JSON Lines; one zstd frame when `compress`.
    pub path: String,
    /// Subtree root; empty = whole tree.
    #[serde(default)]
    pub history: Vec<usize>,
    /// Do not descend a chance node once the board holds this many cards.
    /// 3 = flop actions only, 4 = +turn, 5 = everything. Default 5.
    #[serde(default = "default_max_board_cards")]
    pub max_board_cards: u8,
    #[serde(default)]
    pub include: DumpInclude,
    /// Abort (the job fails, a partial file is left) past this many output bytes, counted
    /// before compression. `None` = unbounded.
    #[serde(default)]
    pub max_bytes: Option<u64>,
    #[serde(default)]
    pub compress: bool,
    /// zstd level when `compress`; defaults like `Spot::compressionLevel` (3).
    #[serde(default = "crate::spot::default_compression_level")]
    pub compression_level: Option<i32>,
}

fn default_max_board_cards() -> u8 {
    5
}

/// What a `simplify` job computes: a grid-rounded, optionally purified copy of one player's
/// strategy over a bounded region, expressed as a `locks` array in `Spot::locks`' exact shape.
///
/// Feed the locks to `resolve` to measure what the simplification costs — the resolved job's
/// `ev`/`exploitability` are the honest numbers (`ev[player]` minus the source's `ev[player]`
/// is the EV loss against an opponent free to exploit the pinned strategy; drive the resolve's
/// `stop` tight, e.g. `targetExploitabilityPct: 0.1`, or the measurement is only as honest as
/// its convergence). This job's own deviation stats are a free proxy, nothing more: a locked
/// hand's zero-frequency actions are genuinely never taken, so the simplified strategy is
/// exploitable *by design*, and its output must not be read as "still near-equilibrium"
/// without the resolve's numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimplifySpec {
    /// Source solve job (a `solve`, `open`, or `resolve`-created job).
    pub job_id: JobId,
    /// Whose strategy to simplify: 0 = OOP, 1 = IP. Only this player's decision nodes are
    /// locked — the cost measurement needs the other side free to exploit, and a human follows
    /// one seat's strategy.
    pub player: usize,
    /// Subtree root, node-command convention (chance step = dealt card id). Empty = the whole
    /// tree, street-bounded below.
    #[serde(default)]
    pub history: Vec<usize>,
    /// Do not descend a chance node once the board holds this many cards. Default 3, not
    /// dump's 5: the locks output scales with every visited decision node, and "make this
    /// street's strategy followable" is the primary use.
    #[serde(default = "default_simplify_board_cards")]
    pub max_board_cards: u8,
    /// Frequencies become multiples of `1/grid`. 2 = halves, 4 = quarters; 1 = pure only (the
    /// degenerate grid is full purification, deliberately).
    #[serde(default = "default_grid")]
    pub grid: u32,
    /// A hand whose top action's frequency is at least this becomes pure — judged against the
    /// *original* probabilities, before grid rounding moves them. `null` disables
    /// purification.
    #[serde(default = "default_purify")]
    pub purify_threshold: Option<f32>,
    /// Emit the `hands` guard on every lock. Doubles the size, and the same server against
    /// the same spot cannot misalign, so default off.
    #[serde(default)]
    pub include_hands: bool,
}

fn default_simplify_board_cards() -> u8 {
    3
}

fn default_grid() -> u32 {
    2
}

fn default_purify() -> Option<f32> {
    Some(0.8)
}

/// A finished simplification, in the exact shape `reportResult` serves for a simplify job.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SimplifyResult {
    pub format_version: u32,
    pub source_job_id: JobId,
    pub player: usize,
    pub history: Vec<usize>,
    pub grid: u32,
    pub purify_threshold: Option<f32>,
    pub max_board_cards: u8,
    /// One entry per locked decision node, `Spot::locks` shape — feed to `resolve` verbatim.
    /// An already-locked source node is rounded like any other (its `strategy()` shows the
    /// pinned mix); a host wanting the source's locks verbatim splices them back in by
    /// history. A node whose every hand is zero-weight emits no entry at all — an all-zero
    /// lock would set `is_locked` while pinning nothing.
    pub locks: Vec<crate::spot::Lock>,
    pub nodes_visited: u64,
    pub nodes_locked: u64,
    /// Hands whose column was pinned (any positive entry).
    pub hands_locked: u64,
    /// Hands the purify threshold made pure, before rounding.
    pub hands_purified: u64,
    /// Zero-weight hands, emitted as all-zero columns — the engine's "free hand" encoding;
    /// pinning a zero-reach hand would freeze solver noise.
    pub hands_free: u64,
    /// `max |simplified − original|` over locked entries.
    pub max_deviation: f32,
    /// Reach-weighted mean of per-hand L1 strategy distance, over locked hands.
    pub mean_deviation: f32,
    /// A street bound or storage boundary stopped descent somewhere.
    pub truncated: bool,
}

/// Why an analysis could not run to completion. The worker maps [`Self::Cancelled`] to a
/// cancelled phase and everything else to a failed one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalyzeError {
    /// The base history does not describe a node.
    #[error("history step {0} does not describe a node: {1}")]
    Walk(usize, String),
    /// The base reaches the wrong kind of node for this report.
    #[error("the base history reaches a {actual} node, but this report needs a {needed} node")]
    BadBase {
        needed: &'static str,
        actual: &'static str,
    },
    /// A `runouts` line contains a chance step.
    #[error("`line` reaches a chance node; lines may only contain decision-action indices")]
    LineCrossesChance,
    /// The source file was saved with reduced storage and the street below is not present.
    #[error("the node sits at this file's storage boundary; the street below was not saved")]
    StorageBoundary,
    /// The dump outgrew its `maxBytes`.
    #[error("the dump reached {bytes} bytes, past the maxBytes limit of {limit}; a partial file was left")]
    TooBig { bytes: u64, limit: u64 },
    /// The output file could not be written.
    #[error("could not write the dump: {0}")]
    Io(String),
    /// The job's cancel flag was set.
    #[error("cancelled")]
    Cancelled,
}

impl From<WalkStep> for AnalyzeError {
    fn from(step: WalkStep) -> Self {
        let what = if step.is_chance {
            "card"
        } else {
            "action index"
        };
        Self::Walk(
            step.depth,
            format!(
                "{what} {} is not available at that node, which offers {}",
                step.index, step.available
            ),
        )
    }
}

/// A finished report, in the exact shape `reportResult` serves.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ReportResult {
    Runouts(RunoutsReport),
    Lines(LinesReport),
    Categories(CategoriesReport),
}

impl ReportResult {
    /// How many rows the result carries — `AnalysisStatus::rows` at `done`.
    #[must_use]
    pub fn rows(&self) -> u64 {
        match self {
            Self::Runouts(r) => r.rows.len() as u64,
            Self::Lines(r) => r.rows.len() as u64,
            Self::Categories(r) => r.rows.len() as u64,
        }
    }
}

/// The `runouts` report: one row per dealable card at the base chance node.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunoutsReport {
    pub format_version: u32,
    pub kind: ReportKind,
    pub source_job_id: JobId,
    pub history: Vec<usize>,
    pub line: Vec<usize>,
    /// The street the base node deals: `"turn"` or `"river"`.
    pub street: &'static str,
    /// Hoisted from the rows — the subtree template is identical for every card. `null` when
    /// the row node is a terminal or chance node.
    pub player: Option<usize>,
    /// Hoisted like [`Self::player`]; empty at terminal row nodes.
    pub actions: Vec<String>,
    pub rows: Vec<RunoutRow>,
}

/// One dealt card's row.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunoutRow {
    /// The dealt card, rendered (`"2c"`).
    pub card: String,
    /// Total matchup count of the acting player's range at the row node (the engine's
    /// normalized-weight semantics: actual combination counts).
    pub matchups: f64,
    /// Frequency of each action across the range. `null` when the row node is terminal or
    /// chance, or when `matchups` is zero.
    pub frequencies: Option<Vec<f64>>,
    /// Range-average equity of `[OOP, IP]`. `null` when no weight remains.
    pub average_equity: Option<[f32; 2]>,
    /// Range-average EV of `[OOP, IP]`, same units as `NodeView.ev`.
    pub average_ev: Option<[f32; 2]>,
    /// Per-category breakdown, present iff the spec asked for it and the row node is a
    /// decision node. Every category is emitted (empty ones with nulls) for a stable row set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<CategoryRow>>,
}

/// The `lines` report: every decision node reachable from the base without crossing a chance
/// node, depth-first in action order, base first.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinesReport {
    pub format_version: u32,
    pub kind: ReportKind,
    pub source_job_id: JobId,
    pub history: Vec<usize>,
    pub rows: Vec<LineRow>,
}

/// One decision node's row.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LineRow {
    /// Action indices relative to the report's base history.
    pub line: Vec<usize>,
    /// The line rendered the way tree-edit lines are written; `"(root)"` at the base.
    pub line_text: String,
    pub player: usize,
    pub actions: Vec<String>,
    /// `matchups / matchups(base)` — 1.0 at the base, 0-guarded.
    pub reach: f64,
    pub matchups: f64,
    pub frequencies: Option<Vec<f64>>,
    pub average_equity: Option<[f32; 2]>,
    pub average_ev: Option<[f32; 2]>,
    /// Present iff the spec asked for categories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<CategoryRow>>,
}

/// The `categories` report: one decision node, partitioned by made-hand category.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoriesReport {
    pub format_version: u32,
    pub kind: ReportKind,
    pub source_job_id: JobId,
    pub history: Vec<usize>,
    pub player: usize,
    pub actions: Vec<String>,
    pub matchups: f64,
    pub rows: Vec<CategoryRow>,
}

/// One category's slice of the acting player's range.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryRow {
    pub category: MadeHand,
    /// Member combos (`"AsAh"`), only in the `categories` kind where one node's list is small.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hands: Option<Vec<String>>,
    pub matchups: f64,
    /// `100 · matchups / node matchups`.
    pub range_pct: f64,
    /// The row formulas restricted to member hands; `null` when the category is empty.
    pub frequencies: Option<Vec<f64>>,
    pub average_equity: Option<f32>,
    pub average_ev: Option<f32>,
    /// Matchup-weighted fraction of the category holding each draw; `null` when empty.
    pub draws: Option<DrawShares>,
}

/// Matchup-weighted draw fractions of one category.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DrawShares {
    pub flush_draw: f64,
    pub open_ended: f64,
    pub gutshot: f64,
}

/// What a finished dump reports back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DumpSummary {
    pub nodes: u64,
    pub decision_nodes: u64,
    pub chance_nodes: u64,
    pub terminal_nodes: u64,
    /// Whether any chance node was left undescended (street bound or storage boundary).
    pub truncated: bool,
    /// Bytes that reached the file, compression included.
    pub bytes_written: u64,
}

/// Run one report over a finalized game. `progress` receives the running node count.
///
/// The game may be positioned anywhere; the report positions itself with a validated replay and
/// leaves the position unspecified afterwards (every other reader replays from the root too).
///
/// # Errors
///
/// [`AnalyzeError`] — a base that does not describe a node or is the wrong node kind, a line
/// crossing a chance node, the storage boundary of a reduced-storage file, or cancellation.
pub fn run_report(
    game: &mut PostFlopGame,
    spec: &ReportSpec,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<ReportResult, AnalyzeError> {
    engine::walk(game, &spec.history)?;
    match spec.kind {
        ReportKind::Runouts => run_runouts(game, spec, cancel, progress),
        ReportKind::Lines => run_lines(game, spec, cancel, progress),
        ReportKind::Categories => run_categories(game, spec, progress),
    }
}

/// The node kind of the current position, for error messages.
fn node_kind(game: &PostFlopGame) -> &'static str {
    if game.is_terminal_node() {
        "terminal"
    } else if game.is_chance_node() {
        "chance"
    } else {
        "decision"
    }
}

/// Whether the current *chance* node may be descended, mirroring the engine's private
/// `can_play_chance_node` with public getters — `play` panics past the boundary of a
/// reduced-storage file.
fn may_descend(game: &PostFlopGame) -> bool {
    let mode = game.storage_mode();
    let is_turn = game.current_board().len() == 3;
    mode != BoardState::Flop && (is_turn || mode != BoardState::Turn)
}

fn run_runouts(
    game: &mut PostFlopGame,
    spec: &ReportSpec,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<ReportResult, AnalyzeError> {
    if !game.is_chance_node() {
        return Err(AnalyzeError::BadBase {
            needed: "chance",
            actual: node_kind(game),
        });
    }
    if !may_descend(game) {
        return Err(AnalyzeError::StorageBoundary);
    }

    let street = if game.current_board().len() == 3 {
        "turn"
    } else {
        "river"
    };
    let base = game.history().to_vec();
    let mask = game.possible_cards();

    let mut rows = Vec::with_capacity(mask.count_ones() as usize);
    let mut player = None;
    let mut actions = Vec::new();
    let mut nodes = 0u64;

    for card in 0..52u8 {
        if mask & (1u64 << card) == 0 {
            continue;
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(AnalyzeError::Cancelled);
        }
        game.apply_history(&base);
        game.play(usize::from(card));
        replay_line(game, &spec.line)?;

        let stats = row_stats(game, spec.categories, false);
        if rows.is_empty() {
            player = stats.player;
            actions.clone_from(&stats.actions);
        }
        rows.push(RunoutRow {
            card: render_card(card),
            matchups: stats.matchups,
            frequencies: stats.frequencies,
            average_equity: stats.average_equity,
            average_ev: stats.average_ev,
            categories: stats.categories,
        });
        nodes += 1;
        progress(nodes);
    }

    Ok(ReportResult::Runouts(RunoutsReport {
        format_version: FORMAT_VERSION,
        kind: ReportKind::Runouts,
        source_job_id: spec.job_id,
        history: base,
        line: spec.line.clone(),
        street,
        player,
        actions,
        rows,
    }))
}

/// Replay a `runouts` line below a freshly dealt card, validating each step.
fn replay_line(game: &mut PostFlopGame, line: &[usize]) -> Result<(), AnalyzeError> {
    for (depth, &step) in line.iter().enumerate() {
        if game.is_chance_node() {
            return Err(AnalyzeError::LineCrossesChance);
        }
        if game.is_terminal_node() {
            return Err(AnalyzeError::Walk(
                depth,
                format!("action index {step} plays past a terminal node"),
            ));
        }
        let available = game.available_actions().len();
        if step >= available {
            return Err(AnalyzeError::Walk(
                depth,
                format!(
                    "action index {step} is not available at that node, which offers {available}"
                ),
            ));
        }
        game.play(step);
    }
    Ok(())
}

fn run_lines(
    game: &mut PostFlopGame,
    spec: &ReportSpec,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<ReportResult, AnalyzeError> {
    if game.is_terminal_node() || game.is_chance_node() {
        return Err(AnalyzeError::BadBase {
            needed: "decision",
            actual: node_kind(game),
        });
    }

    let base = game.history().to_vec();
    let mut rows: Vec<LineRow> = Vec::new();
    let mut nodes = 0u64;

    // An explicit stack of relative lines, visited depth-first in action order. Pushing the
    // children in reverse keeps the pop order equal to the action order.
    let mut stack: Vec<(Vec<usize>, Vec<String>)> = vec![(Vec::new(), Vec::new())];
    while let Some((rel, texts)) = stack.pop() {
        if cancel.load(Ordering::Relaxed) {
            return Err(AnalyzeError::Cancelled);
        }
        let mut abs = base.clone();
        abs.extend_from_slice(&rel);
        game.apply_history(&abs);
        if game.is_terminal_node() || game.is_chance_node() {
            continue;
        }

        let stats = row_stats(game, spec.categories, false);
        let base_matchups = rows.first().map_or(stats.matchups, |first| first.matchups);
        let reach = if rel.is_empty() {
            1.0
        } else if base_matchups > 0.0 {
            stats.matchups / base_matchups
        } else {
            0.0
        };
        let line_text = if texts.is_empty() {
            "(root)".to_owned()
        } else {
            texts.join(", ")
        };
        rows.push(LineRow {
            line: rel.clone(),
            line_text,
            player: stats.player.expect("a decision node has a player"),
            actions: stats.actions.clone(),
            reach,
            matchups: stats.matchups,
            frequencies: stats.frequencies,
            average_equity: stats.average_equity,
            average_ev: stats.average_ev,
            categories: stats.categories,
        });
        nodes += 1;
        progress(nodes);

        for (index, action) in stats.actions.iter().enumerate().rev() {
            let mut child = rel.clone();
            child.push(index);
            let mut child_texts = texts.clone();
            child_texts.push(action.clone());
            stack.push((child, child_texts));
        }
    }

    Ok(ReportResult::Lines(LinesReport {
        format_version: FORMAT_VERSION,
        kind: ReportKind::Lines,
        source_job_id: spec.job_id,
        history: base,
        rows,
    }))
}

fn run_categories(
    game: &mut PostFlopGame,
    spec: &ReportSpec,
    progress: &mut dyn FnMut(u64),
) -> Result<ReportResult, AnalyzeError> {
    if game.is_terminal_node() || game.is_chance_node() {
        return Err(AnalyzeError::BadBase {
            needed: "decision",
            actual: node_kind(game),
        });
    }
    let stats = row_stats(game, true, true);
    progress(1);
    Ok(ReportResult::Categories(CategoriesReport {
        format_version: FORMAT_VERSION,
        kind: ReportKind::Categories,
        source_job_id: spec.job_id,
        history: game.history().to_vec(),
        player: stats.player.expect("a decision node has a player"),
        actions: stats.actions,
        matchups: stats.matchups,
        rows: stats.categories.expect("categories were requested"),
    }))
}

/// Everything one row needs, read from the game's current node.
struct RowStats {
    player: Option<usize>,
    actions: Vec<String>,
    matchups: f64,
    frequencies: Option<Vec<f64>>,
    average_equity: Option<[f32; 2]>,
    average_ev: Option<[f32; 2]>,
    categories: Option<Vec<CategoryRow>>,
}

/// Read one node's aggregates. At a terminal or chance node there is no acting player:
/// `frequencies` and `categories` are `null`, `matchups` counts OOP's range (both players'
/// matchup totals are equal by construction), and the per-player averages are still computed.
fn row_stats(game: &mut PostFlopGame, want_categories: bool, want_hands: bool) -> RowStats {
    game.cache_normalized_weights();
    let is_decision = !game.is_terminal_node() && !game.is_chance_node();

    // Per-player averages, 0-guarded so an emptied range yields null rather than NaN.
    let mut average_equity = [None; 2];
    let mut average_ev = [None; 2];
    for p in 0..2 {
        let w = game.normalized_weights(p);
        let sum: f64 = w.iter().map(|&x| f64::from(x)).sum();
        if sum > 0.0 {
            let eq = game.equity(p);
            let ev = game.expected_values(p);
            average_equity[p] = Some((weighted_sum(&eq, w) / sum) as f32);
            average_ev[p] = Some((weighted_sum(&ev, w) / sum) as f32);
        }
    }
    let average_equity = average_equity[0]
        .zip(average_equity[1])
        .map(|(a, b)| [a, b]);
    let average_ev = average_ev[0].zip(average_ev[1]).map(|(a, b)| [a, b]);

    if !is_decision {
        let matchups: f64 = game
            .normalized_weights(0)
            .iter()
            .map(|&x| f64::from(x))
            .sum();
        return RowStats {
            player: None,
            actions: Vec::new(),
            matchups,
            frequencies: None,
            average_equity,
            average_ev,
            categories: None,
        };
    }

    let player = game.current_player();
    let actions: Vec<String> = game
        .available_actions()
        .iter()
        .map(crate::convert::action_to_string)
        .collect();
    let num_actions = actions.len();
    let weights = game.normalized_weights(player).to_vec();
    let num_hands = weights.len();
    let matchups: f64 = weights.iter().map(|&x| f64::from(x)).sum();
    let strategy = game.strategy();

    let frequencies = (matchups > 0.0).then(|| {
        (0..num_actions)
            .map(|a| {
                let row = &strategy[a * num_hands..(a + 1) * num_hands];
                weighted_sum(row, &weights) / matchups
            })
            .collect()
    });

    let categories = want_categories.then(|| {
        let board = game.current_board();
        let holes = game.private_cards(player).to_vec();
        let equity = game.equity(player);
        let ev = game.expected_values(player);
        let classified: Vec<(MadeHand, Draws)> =
            holes.iter().map(|&h| classify(h, &board)).collect();

        MadeHand::ALL
            .iter()
            .map(|&category| {
                let members: Vec<usize> = (0..num_hands)
                    .filter(|&h| classified[h].0 == category)
                    .collect();
                let cat_matchups: f64 = members.iter().map(|&h| f64::from(weights[h])).sum();
                let occupied = cat_matchups > 0.0;

                let frequencies = occupied.then(|| {
                    (0..num_actions)
                        .map(|a| {
                            members
                                .iter()
                                .map(|&h| {
                                    f64::from(strategy[a * num_hands + h]) * f64::from(weights[h])
                                })
                                .sum::<f64>()
                                / cat_matchups
                        })
                        .collect()
                });
                let avg = |values: &[f32]| {
                    occupied.then(|| {
                        (members
                            .iter()
                            .map(|&h| f64::from(values[h]) * f64::from(weights[h]))
                            .sum::<f64>()
                            / cat_matchups) as f32
                    })
                };
                let draws = occupied.then(|| {
                    let share = |pick: fn(&Draws) -> bool| {
                        members
                            .iter()
                            .filter(|&&h| pick(&classified[h].1))
                            .map(|&h| f64::from(weights[h]))
                            .sum::<f64>()
                            / cat_matchups
                    };
                    DrawShares {
                        flush_draw: share(|d| d.flush_draw),
                        open_ended: share(|d| d.open_ended),
                        gutshot: share(|d| d.gutshot),
                    }
                });
                let hands = want_hands.then(|| {
                    members
                        .iter()
                        .map(|&h| {
                            crate::convert::hole_to_string(holes[h])
                                .unwrap_or_else(|_| "??".to_owned())
                        })
                        .collect()
                });

                CategoryRow {
                    category,
                    hands,
                    matchups: cat_matchups,
                    range_pct: if matchups > 0.0 {
                        100.0 * cat_matchups / matchups
                    } else {
                        0.0
                    },
                    frequencies,
                    average_equity: avg(&equity),
                    average_ev: avg(&ev),
                    draws,
                }
            })
            .collect()
    });

    RowStats {
        player: Some(player),
        actions,
        matchups,
        frequencies,
        average_equity,
        average_ev,
        categories,
    }
}

fn weighted_sum(values: &[f32], weights: &[f32]) -> f64 {
    values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| f64::from(v) * f64::from(w))
        .sum()
}

fn render_card(card: u8) -> String {
    crate::convert::from_engine_card(card).map_or_else(|_| "??".to_owned(), |c| c.to_string())
}

// ---------------------------------------------------------------------------------------------
// The dump.
// ---------------------------------------------------------------------------------------------

/// One JSON Lines file: header, then every reachable node in depth-first order (decision-node
/// children in action order, chance children in ascending card id, parents before children,
/// terminals counted but not written), then a summary trailer.
///
/// `progress` receives (nodes visited, pre-compression bytes written).
///
/// # Errors
///
/// [`AnalyzeError`] — a bad base history, I/O, the `maxBytes` abort, or cancellation. On every
/// error path the partial file is left on disk; the missing summary line marks it incomplete.
pub fn run_dump(
    game: &mut PostFlopGame,
    spec: &DumpSpec,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<DumpSummary, AnalyzeError> {
    engine::walk(game, &spec.history)?;

    let file = File::create(&spec.path).map_err(|e| AnalyzeError::Io(e.to_string()))?;
    let buffered = BufWriter::with_capacity(
        64 * 1024,
        CountingWriter {
            inner: file,
            count: 0,
        },
    );
    let mut sink = if spec.compress {
        let level = spec
            .compression_level
            .unwrap_or(crate::engine::DEFAULT_COMPRESSION_LEVEL);
        Sink::Zstd(
            zstd::stream::write::Encoder::new(buffered, level)
                .map_err(|e| AnalyzeError::Io(e.to_string()))?,
        )
    } else {
        Sink::Plain(buffered)
    };

    let tree = game.tree_config();
    let (pot, stack) = (tree.starting_pot, tree.effective_stack);
    let hands = |p: usize| -> Vec<String> {
        game.private_cards(p)
            .iter()
            .map(|&h| crate::convert::hole_to_string(h).unwrap_or_else(|_| "??".to_owned()))
            .collect()
    };
    let header = HeaderLine {
        t: "header",
        format_version: FORMAT_VERSION,
        generator: concat!("pkwiz-solver ", env!("CARGO_PKG_VERSION")),
        engine_rev: crate::protocol::ENGINE_REV,
        source_job_id: spec.job_id,
        board: game
            .current_board()
            .iter()
            .map(|&c| render_card(c))
            .collect(),
        pot,
        effective_stack: stack,
        history: spec.history.clone(),
        max_board_cards: spec.max_board_cards,
        include: spec.include,
        oop_hands: hands(0),
        ip_hands: hands(1),
    };

    let mut ctx = DumpCtx {
        spec,
        cancel,
        progress,
        sink: &mut sink,
        written: 0,
        counts: DumpCounts::default(),
        truncated: false,
    };
    ctx.write_line(&header)?;
    dump_node(game, &mut ctx)?;

    let summary = SummaryLine {
        t: "summary",
        complete: true,
        nodes: ctx.counts.nodes,
        decision_nodes: ctx.counts.decision,
        chance_nodes: ctx.counts.chance,
        terminal_nodes: ctx.counts.terminal,
        truncated: ctx.truncated,
        bytes_written: ctx.written,
    };
    ctx.write_line(&summary)?;
    let (counts, truncated) = (ctx.counts, ctx.truncated);

    let bytes_written = sink.finish().map_err(|e| AnalyzeError::Io(e.to_string()))?;
    Ok(DumpSummary {
        nodes: counts.nodes,
        decision_nodes: counts.decision,
        chance_nodes: counts.chance,
        terminal_nodes: counts.terminal,
        truncated,
        bytes_written,
    })
}

/// Counts the bytes that actually reach the file — post-compression, which is what
/// [`DumpSummary::bytes_written`] reports once the stream is flushed. The `maxBytes` meter is
/// the *pre*-compression [`DumpCtx::written`] instead, checked per line so the abort cannot
/// hide behind the 64 KiB buffer or the zstd frame.
struct CountingWriter {
    inner: File,
    count: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// The output stack: file ← counter ← buffer ← optional zstd frame.
enum Sink {
    Plain(BufWriter<CountingWriter>),
    Zstd(zstd::stream::write::Encoder<'static, BufWriter<CountingWriter>>),
}

impl Sink {
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Plain(w) => w.write_all(buf),
            Self::Zstd(w) => w.write_all(buf),
        }
    }

    /// Flush everything (ending the zstd frame) and report the bytes that reached the file.
    fn finish(self) -> std::io::Result<u64> {
        let mut buffered = match self {
            Self::Plain(w) => w,
            Self::Zstd(w) => w.finish()?,
        };
        buffered.flush()?;
        let counting = buffered
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(counting.count)
    }
}

#[derive(Default, Clone, Copy)]
struct DumpCounts {
    nodes: u64,
    decision: u64,
    chance: u64,
    terminal: u64,
}

struct DumpCtx<'a> {
    spec: &'a DumpSpec,
    cancel: &'a AtomicBool,
    progress: &'a mut dyn FnMut(u64, u64),
    sink: &'a mut Sink,
    /// Pre-compression bytes written — the `maxBytes` meter, checked before buffers flush so
    /// the abort cannot hide behind buffering.
    written: u64,
    counts: DumpCounts,
    truncated: bool,
}

impl DumpCtx<'_> {
    fn write_line<T: Serialize>(&mut self, line: &T) -> Result<(), AnalyzeError> {
        let mut text = serde_json::to_string(line).map_err(|e| AnalyzeError::Io(e.to_string()))?;
        text.push('\n');
        self.sink
            .write_all(text.as_bytes())
            .map_err(|e| AnalyzeError::Io(e.to_string()))?;
        self.written += text.len() as u64;
        if let Some(limit) = self.spec.max_bytes {
            if self.written > limit {
                return Err(AnalyzeError::TooBig {
                    bytes: self.written,
                    limit,
                });
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HeaderLine {
    t: &'static str,
    format_version: u32,
    generator: &'static str,
    engine_rev: &'static str,
    source_job_id: JobId,
    board: Vec<String>,
    pot: i32,
    effective_stack: i32,
    history: Vec<usize>,
    max_board_cards: u8,
    include: DumpInclude,
    oop_hands: Vec<String>,
    ip_hands: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeLine<'a> {
    t: &'static str,
    history: &'a [usize],
    player: usize,
    actions: Vec<String>,
    is_locked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    strategy: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ev_detail: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ev: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    equity: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<Vec<f32>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChanceLine<'a> {
    t: &'static str,
    history: &'a [usize],
    street: &'static str,
    cards: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummaryLine {
    t: &'static str,
    complete: bool,
    nodes: u64,
    decision_nodes: u64,
    chance_nodes: u64,
    terminal_nodes: u64,
    truncated: bool,
    bytes_written: u64,
}

/// The recursive traversal. Tree depth is bounded by the action tree (well under 50), so
/// recursion is fine; siblings after the first are restored with `apply_history`, exactly like
/// the engine's own `visit_recursive`.
fn dump_node(game: &mut PostFlopGame, ctx: &mut DumpCtx<'_>) -> Result<(), AnalyzeError> {
    if ctx.cancel.load(Ordering::Relaxed) {
        return Err(AnalyzeError::Cancelled);
    }

    ctx.counts.nodes += 1;
    if game.is_terminal_node() {
        // Not written: any listed action without a child line is terminal (or pruned).
        ctx.counts.terminal += 1;
        (ctx.progress)(ctx.counts.nodes, ctx.written);
        return Ok(());
    }

    let history = game.history().to_vec();

    if game.is_chance_node() {
        ctx.counts.chance += 1;
        let street = if game.current_board().len() == 3 {
            "turn"
        } else {
            "river"
        };
        let mask = game.possible_cards();
        let dealable: Vec<u8> = (0..52u8).filter(|&c| mask & (1u64 << c) != 0).collect();
        let line = ChanceLine {
            t: "chance",
            history: &history,
            street,
            cards: dealable.iter().map(|&c| render_card(c)).collect(),
        };
        ctx.write_line(&line)?;
        (ctx.progress)(ctx.counts.nodes, ctx.written);

        // Street bound and storage boundary both refuse descent; the chance line stays, and
        // the summary says the file is truncated.
        if game.current_board().len() >= usize::from(ctx.spec.max_board_cards) || !may_descend(game)
        {
            ctx.truncated = true;
            return Ok(());
        }

        let children = chance_children(game);

        for (index, &card) in children.iter().enumerate() {
            if index > 0 {
                game.apply_history(&history);
            }
            game.play(usize::from(card));
            dump_node(game, ctx)?;
        }
        return Ok(());
    }

    // A decision node.
    ctx.counts.decision += 1;
    let player = game.current_player();
    let actions: Vec<String> = game
        .available_actions()
        .iter()
        .map(crate::convert::action_to_string)
        .collect();
    let num_actions = actions.len();
    let num_hands = game.private_cards(player).len().max(1);
    let include = ctx.spec.include;

    let chunk =
        |flat: Vec<f32>| -> Vec<Vec<f32>> { flat.chunks(num_hands).map(<[f32]>::to_vec).collect() };
    // Cached only when a reader below needs it — for a strategy-only dump it is wasted work,
    // and O(#OOP × #IP) per node on a bunching source.
    if include.ev || include.equity || include.weights {
        game.cache_normalized_weights();
    }
    let line = NodeLine {
        t: "node",
        history: &history,
        player,
        actions,
        is_locked: game.current_locking_strategy().is_some(),
        strategy: include.strategy.then(|| chunk(game.strategy())),
        ev_detail: include
            .ev
            .then(|| chunk(game.expected_values_detail(player))),
        ev: include.ev.then(|| game.expected_values(player)),
        equity: include.equity.then(|| game.equity(player)),
        weights: include
            .weights
            .then(|| game.normalized_weights(player).to_vec()),
    };
    ctx.write_line(&line)?;
    (ctx.progress)(ctx.counts.nodes, ctx.written);

    for index in 0..num_actions {
        if index > 0 {
            game.apply_history(&history);
        }
        game.play(index);
        dump_node(game, ctx)?;
    }
    Ok(())
}

/// The chance node's children to descend, ascending card id.
///
/// The isomorphism dedupe, public-API only: the storage representatives are the
/// `Action::Chance` children. If every one of them is itself dealable, storage and actual
/// coordinates agree here, and playing exactly those cards reproduces the engine's
/// representative-only traversal. If any is not dealable, a suit swap is active above us —
/// fall back to every dealable card: correct, merely undeduplicated. Shared by the dump and
/// the simplification; for the latter the dedupe also keeps the locks array free of
/// last-write-wins duplicates, since isomorphic branches share one storage node.
fn chance_children(game: &PostFlopGame) -> Vec<u8> {
    let mask = game.possible_cards();
    let representatives: Vec<u8> = game
        .available_actions()
        .iter()
        .filter_map(|a| match a {
            postflop_solver::Action::Chance(c) => Some(*c),
            _ => None,
        })
        .collect();
    let all_dealable = representatives.iter().all(|&c| mask & (1u64 << c) != 0);
    let mut children = if all_dealable {
        representatives
    } else {
        (0..52u8).filter(|&c| mask & (1u64 << c) != 0).collect()
    };
    children.sort_unstable();
    children
}

// ---------------------------------------------------------------------------------------------
// The simplification.
// ---------------------------------------------------------------------------------------------

/// Walk the subtree under the spec's base and round the requested player's strategy onto the
/// `1/grid` grid, emitting one `Lock` per locked decision node. `progress` receives the
/// running node count.
///
/// The traversal is [`run_dump`]'s, minus the writer: `apply_history` sibling restore,
/// `chance_children` isomorphism dedupe, the street bound and `may_descend` storage guard
/// both setting `truncated`. A base that is a terminal or chance node, or a region where only
/// the opponent acts, is legal and yields an empty locks array — a well-formed question with
/// an empty answer.
///
/// # Errors
///
/// [`AnalyzeError`] — a base history that does not describe a node, or cancellation.
pub fn run_simplify(
    game: &mut PostFlopGame,
    spec: &SimplifySpec,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<SimplifyResult, AnalyzeError> {
    engine::walk(game, &spec.history)?;

    let mut ctx = SimplifyCtx {
        spec,
        cancel,
        progress,
        locks: Vec::new(),
        nodes_visited: 0,
        nodes_locked: 0,
        hands_locked: 0,
        hands_purified: 0,
        hands_free: 0,
        max_deviation: 0.0,
        l1_weighted: 0.0,
        weight_sum: 0.0,
        truncated: false,
    };
    simplify_node(game, &mut ctx)?;

    // 0-guarded like every other range average here: a region with no locked hands has no
    // meaningful mean.
    let mean_deviation = if ctx.weight_sum > 0.0 {
        (ctx.l1_weighted / ctx.weight_sum) as f32
    } else {
        0.0
    };

    Ok(SimplifyResult {
        format_version: FORMAT_VERSION,
        source_job_id: spec.job_id,
        player: spec.player,
        history: spec.history.clone(),
        grid: spec.grid,
        purify_threshold: spec.purify_threshold,
        max_board_cards: spec.max_board_cards,
        locks: ctx.locks,
        nodes_visited: ctx.nodes_visited,
        nodes_locked: ctx.nodes_locked,
        hands_locked: ctx.hands_locked,
        hands_purified: ctx.hands_purified,
        hands_free: ctx.hands_free,
        max_deviation: ctx.max_deviation,
        mean_deviation,
        truncated: ctx.truncated,
    })
}

struct SimplifyCtx<'a> {
    spec: &'a SimplifySpec,
    cancel: &'a AtomicBool,
    progress: &'a mut dyn FnMut(u64),
    locks: Vec<crate::spot::Lock>,
    nodes_visited: u64,
    nodes_locked: u64,
    hands_locked: u64,
    hands_purified: u64,
    hands_free: u64,
    max_deviation: f32,
    /// `Σ w·L1` over locked hands, and the matching `Σ w` — the reach-weighted mean's parts.
    l1_weighted: f64,
    weight_sum: f64,
    truncated: bool,
}

/// The recursive traversal, [`dump_node`]'s twin (see its comment on recursion depth and
/// sibling restore).
fn simplify_node(game: &mut PostFlopGame, ctx: &mut SimplifyCtx<'_>) -> Result<(), AnalyzeError> {
    if ctx.cancel.load(Ordering::Relaxed) {
        return Err(AnalyzeError::Cancelled);
    }

    ctx.nodes_visited += 1;
    (ctx.progress)(ctx.nodes_visited);
    if game.is_terminal_node() {
        return Ok(());
    }

    let history = game.history().to_vec();

    if game.is_chance_node() {
        // Street bound and storage boundary both refuse descent, and the result says so.
        if game.current_board().len() >= usize::from(ctx.spec.max_board_cards) || !may_descend(game)
        {
            ctx.truncated = true;
            return Ok(());
        }
        for (index, &card) in chance_children(game).iter().enumerate() {
            if index > 0 {
                game.apply_history(&history);
            }
            game.play(usize::from(card));
            simplify_node(game, ctx)?;
        }
        return Ok(());
    }

    let player = game.current_player();
    if player == ctx.spec.player {
        simplify_decision_node(game, ctx, &history);
    }

    // Descend through both players' nodes: the requested player's next decision may sit below
    // an opponent action.
    let num_actions = game.available_actions().len();
    for index in 0..num_actions {
        if index > 0 {
            game.apply_history(&history);
        }
        game.play(index);
        simplify_node(game, ctx)?;
    }
    Ok(())
}

/// Round one decision node of the requested player, pushing a `Lock` unless every hand came
/// out free.
fn simplify_decision_node(game: &mut PostFlopGame, ctx: &mut SimplifyCtx<'_>, history: &[usize]) {
    let player = game.current_player();
    // The zero-weight skip needs per-node normalized weights — the O(#OOP × #IP) cost on a
    // bunching source that makes simplify a job rather than a synchronous command.
    game.cache_normalized_weights();
    let weights = game.normalized_weights(player).to_vec();
    let num_hands = weights.len();
    let num_actions = game.available_actions().len();
    let flat = game.strategy();

    let mut rows = vec![vec![0.0f32; num_hands]; num_actions];
    let mut any_locked = false;
    for h in 0..num_hands {
        if weights[h] == 0.0 {
            // All-zero column: the engine's "free hand" encoding.
            ctx.hands_free += 1;
            continue;
        }
        let column: Vec<f32> = (0..num_actions).map(|a| flat[a * num_hands + h]).collect();
        let (rounded, purified) = round_column(&column, ctx.spec.grid, ctx.spec.purify_threshold);
        if purified {
            ctx.hands_purified += 1;
        }
        ctx.hands_locked += 1;
        any_locked = true;

        let mut l1 = 0.0f64;
        for a in 0..num_actions {
            let deviation = (rounded[a] - column[a]).abs();
            ctx.max_deviation = ctx.max_deviation.max(deviation);
            l1 += f64::from(deviation);
            rows[a][h] = rounded[a];
        }
        ctx.l1_weighted += f64::from(weights[h]) * l1;
        ctx.weight_sum += f64::from(weights[h]);
    }

    if any_locked {
        ctx.nodes_locked += 1;
        let hands = ctx.spec.include_hands.then(|| {
            game.private_cards(player)
                .iter()
                .map(|&h| crate::convert::hole_to_string(h).unwrap_or_else(|_| "??".to_owned()))
                .collect()
        });
        ctx.locks.push(crate::spot::Lock {
            history: history.to_vec(),
            strategy: rows,
            hands,
        });
    }
}

/// Round one hand's action distribution onto the `1/grid` grid. Returns the rounded column and
/// whether purification fired.
///
/// Purify runs **before** grid rounding — the threshold judges the original probabilities, not
/// numbers the rounding already moved — with ties going to the lowest action index (the
/// engine's best-response convention). The grid step is largest-remainder renormalization in
/// integer units: `u[a] = round(p[a]·grid)`, then `Σu` is repaired to exactly `grid` by
/// incrementing the largest positive remainders (deficit) or decrementing positive entries
/// with the smallest remainders (surplus), ties to the lowest index for determinism. The
/// result always has a positive entry (`Σu = grid ≥ 1`), so a locked column can never
/// degenerate into the free-hand encoding, and the ratios are exactly on-grid — the engine's
/// per-hand normalization divides by ≈1 and is a no-op in ratio terms.
fn round_column(p: &[f32], grid: u32, purify_threshold: Option<f32>) -> (Vec<f32>, bool) {
    let mut best = 0;
    for (a, &v) in p.iter().enumerate() {
        // Strict comparison: ties keep the lowest index.
        if v > p[best] {
            best = a;
        }
    }
    if let Some(threshold) = purify_threshold {
        if p[best] >= threshold {
            let mut column = vec![0.0; p.len()];
            column[best] = 1.0;
            return (column, true);
        }
    }

    let grid_f = f64::from(grid);
    let exact: Vec<f64> = p.iter().map(|&v| f64::from(v) * grid_f).collect();
    let mut units: Vec<i64> = exact.iter().map(|&e| e.round() as i64).collect();
    let mut sum: i64 = units.iter().sum();
    let target = i64::from(grid);
    while sum < target {
        let mut pick = 0;
        let mut best_remainder = f64::NEG_INFINITY;
        for (a, &e) in exact.iter().enumerate() {
            let remainder = e - units[a] as f64;
            if remainder > best_remainder {
                best_remainder = remainder;
                pick = a;
            }
        }
        units[pick] += 1;
        sum += 1;
    }
    while sum > target {
        let mut pick = None;
        let mut best_remainder = f64::INFINITY;
        for (a, &e) in exact.iter().enumerate() {
            if units[a] == 0 {
                continue;
            }
            let remainder = e - units[a] as f64;
            if remainder < best_remainder {
                best_remainder = remainder;
                pick = Some(a);
            }
        }
        let pick = pick.expect("a positive sum implies a positive entry");
        units[pick] -= 1;
        sum -= 1;
    }

    (
        units.iter().map(|&u| (u as f64 / grid_f) as f32).collect(),
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_lands_on_the_grid_and_repairs_the_sum() {
        // Plain rounding that already sums right.
        let (rounded, purified) = round_column(&[0.6, 0.4], 2, None);
        assert_eq!(rounded, vec![0.5, 0.5]);
        assert!(!purified);

        // Quarters.
        let (rounded, _) = round_column(&[0.7, 0.2, 0.1], 4, None);
        assert_eq!(rounded, vec![0.75, 0.25, 0.0]);

        // Deficit repair: five near-uniform actions at grid 2 — every unit rounds to zero and
        // the two missing units go to the largest remainders, ties broken by lowest index.
        let (rounded, _) = round_column(&[0.2, 0.2, 0.2, 0.2, 0.2], 2, None);
        assert_eq!(rounded, vec![0.5, 0.5, 0.0, 0.0, 0.0]);

        // Surplus repair: both 0.5s round up at grid 1 and the lowest index is decremented.
        let (rounded, _) = round_column(&[0.5, 0.5], 1, None);
        assert_eq!(rounded, vec![0.0, 1.0]);

        // Determinism: the same column rounds the same way every time.
        for _ in 0..3 {
            let (again, _) = round_column(&[0.2, 0.2, 0.2, 0.2, 0.2], 2, None);
            assert_eq!(again, vec![0.5, 0.5, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn grid_one_is_full_purification() {
        let (rounded, purified) = round_column(&[0.6, 0.3, 0.1], 1, None);
        assert_eq!(rounded, vec![1.0, 0.0, 0.0]);
        assert!(!purified, "the grid did it, not the threshold");
    }

    #[test]
    fn purify_judges_the_original_probabilities_before_rounding() {
        // 0.85 meets the 0.8 threshold: pure, and flagged as purified.
        let (rounded, purified) = round_column(&[0.85, 0.15], 2, Some(0.8));
        assert_eq!(rounded, vec![1.0, 0.0]);
        assert!(purified);

        // 0.6 does not meet 0.9 — but at grid 2 it rounds to a half-half mix, which a
        // threshold applied *after* rounding would have judged differently.
        let (rounded, purified) = round_column(&[0.6, 0.4], 2, Some(0.9));
        assert_eq!(rounded, vec![0.5, 0.5]);
        assert!(!purified);

        // Tie on the max: the lowest action index wins, the engine's own convention.
        let (rounded, purified) = round_column(&[0.5, 0.5], 2, Some(0.5));
        assert_eq!(rounded, vec![1.0, 0.0]);
        assert!(purified);
    }

    #[test]
    fn every_rounded_column_sums_to_one_on_the_grid() {
        // Awkward inputs: float noise, a dominant tail entry, a grid that is not a power of
        // two. The invariant is Σ = 1 (within float addition) and every entry a multiple of
        // 1/grid in integer units.
        for (p, grid) in [
            (vec![0.333_333_34f32, 0.333_333_34, 0.333_333_34], 3u32),
            (vec![0.05, 0.05, 0.9], 2),
            (vec![0.998, 0.001, 0.001], 64),
            (vec![0.25; 4], 2),
        ] {
            let (rounded, _) = round_column(&p, grid, None);
            let sum: f64 = rounded.iter().map(|&v| f64::from(v)).sum();
            assert!((sum - 1.0).abs() < 1e-6, "{p:?} at grid {grid}: sum {sum}");
            assert!(
                rounded.iter().any(|&v| v > 0.0),
                "{p:?} at grid {grid}: a locked column must pin something"
            );
            for &v in &rounded {
                let units = f64::from(v) * f64::from(grid);
                assert!(
                    (units - units.round()).abs() < 1e-6,
                    "{p:?} at grid {grid}: {v} is off-grid"
                );
            }
        }
    }
}
