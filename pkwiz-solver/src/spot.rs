//! The **input shape**: a spot to solve, expressed the way a hand history already describes one.
//!
//! The request type is built around what a parsed hand already gives you — a board, a pot, an
//! effective stack — because "solve this spot from a real hand" is the thing hosts actually want.
//! Everything else has a defensible default, so the minimum viable request is five fields:
//!
//! ```json
//! {"oop": "66+,A8s+", "ip": "QQ-22,AQs-A2s", "board": "Td9d6hQc", "pot": 200,
//!  "effectiveStack": 900}
//! ```
//!
//! # Units are the caller's
//!
//! `pot` and `effectiveStack` are plain integers and the engine only ever compares them to each
//! other, so a caller working in cents (as `pkwiz-model` does) passes cents, and a caller thinking
//! in big blinds passes big blinds. The one rule is that they agree with each other and with the
//! bet sizes, which are all pot-relative by default.
//!
//! # Ranges never arrive as `postflop_solver::Range`
//!
//! A [`RangeSpec`] is either our notation (parsed by `pkwiz-eval`, which is `rs-poker`'s grammar
//! plus our weights) or an explicit [`pkwiz_range::Range`]. The engine's own `FromStr` for `Range`
//! is deliberately never called: keeping that type inside [`crate::convert`] is what stops the
//! engine's licence travelling with its API, and routing every range through one parser means a
//! host and this solver cannot disagree about what `JTs-67s` means.

use pkwiz_range::Card;
use postflop_solver::{
    Action, ActionTree, BetSizeOptions, BoardState, DonkSizeOptions, TreeConfig,
};
use serde::{Deserialize, Serialize};

/// A range, either as notation or already expanded.
///
/// Untagged, so the wire form is a bare string (`"QQ+,AKs"`) or the JSON array a
/// [`pkwiz_range::Range`] serialises to — a range editor round-trips its own state without a
/// lossy trip back through notation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RangeSpec {
    /// `"66+,A8s+,A5s-A4s"`, with optional `:weight` suffixes.
    Notation(String),
    /// An explicit weighted combination list.
    Explicit(pkwiz_range::Range),
}

impl RangeSpec {
    /// Resolve to our own range type. Notation is parsed here; explicit ranges pass through.
    pub fn resolve(&self) -> Result<pkwiz_range::Range, SpotError> {
        match self {
            Self::Notation(s) => pkwiz_range::Range::parse(s).map_err(|e| SpotError::Range {
                notation: s.clone(),
                reason: e.to_string(),
            }),
            Self::Explicit(r) => Ok(r.clone()),
        }
    }
}

/// The board, as text (`"Td9d6h"`, `"Td 9d 6h Qc"`) or as a list of cards (`["Td","9d","6h"]`).
///
/// Three, four or five cards; the count *is* the street, which is why a hand's board can be
/// forwarded verbatim without the caller also having to say which street it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BoardSpec {
    Text(String),
    Cards(Vec<String>),
}

impl BoardSpec {
    /// Parse into our own cards.
    ///
    /// Cards are parsed by [`pkwiz_range::Card`], not by the engine's `card_from_str`, for the
    /// same reason ranges are: one parser, one set of quirks. The two encodings happen to be
    /// identical (`rank * 4 + suit`, clubs first) which is what makes [`crate::convert`] a cast
    /// rather than a table.
    pub fn cards(&self) -> Result<Vec<Card>, SpotError> {
        let raw: Vec<String> = match self {
            Self::Cards(v) => v.clone(),
            Self::Text(s) => {
                let compact: Vec<char> = s
                    .chars()
                    .filter(|c| !c.is_whitespace() && *c != ',')
                    .collect();
                if !compact.len().is_multiple_of(2) {
                    return Err(SpotError::Board {
                        board: s.clone(),
                        reason: "an odd number of characters cannot be a list of cards".to_owned(),
                    });
                }
                compact.chunks(2).map(|c| c.iter().collect()).collect()
            }
        };

        let text = || match self {
            Self::Text(s) => s.clone(),
            Self::Cards(v) => v.join(""),
        };

        let cards = raw
            .iter()
            .map(|c| Card::parse(c))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| SpotError::Board {
                board: text(),
                reason: e.to_string(),
            })?;

        if !(3..=5).contains(&cards.len()) {
            return Err(SpotError::Board {
                board: text(),
                reason: format!(
                    "a postflop board is three, four or five cards; got {}",
                    cards.len()
                ),
            });
        }

        let mut seen = 0u64;
        for c in &cards {
            let bit = 1u64 << c.index();
            if seen & bit != 0 {
                return Err(SpotError::Board {
                    board: text(),
                    reason: format!("{c} appears twice"),
                });
            }
            seen |= bit;
        }

        Ok(cards)
    }
}

/// Bet and raise sizes for one street, in the engine's notation.
///
/// `"60%"` pot-relative, `"2.5x"` relative to the previous bet, `"e"` geometric, `"a"` all-in,
/// `"100c"` a constant amount — comma-separated for several options. The grammar belongs to
/// `postflop-solver`; validating it is one `TryFrom` away and its error message is passed through
/// verbatim, so a typo in the UI reads as a typo and not as "invalid spot".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreetSizing {
    /// First-in bet sizes.
    pub bet: String,
    /// Raise sizes.
    pub raise: String,
}

impl Default for StreetSizing {
    /// Half pot, 2.5× raises — the usual small-tree default, and cheap enough to solve
    /// interactively.
    fn default() -> Self {
        Self {
            bet: "50%".to_owned(),
            raise: "2.5x".to_owned(),
        }
    }
}

impl StreetSizing {
    /// A street with no betting at all: the only line is check-through.
    #[must_use]
    pub fn none() -> Self {
        Self {
            bet: String::new(),
            raise: String::new(),
        }
    }

    fn options(&self, street: &'static str) -> Result<BetSizeOptions, SpotError> {
        BetSizeOptions::try_from((self.bet.as_str(), self.raise.as_str())).map_err(|reason| {
            SpotError::Sizing {
                street,
                reason: reason.to_string(),
            }
        })
    }
}

/// The whole bet-sizing configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Sizing {
    pub flop: StreetSizing,
    pub turn: StreetSizing,
    pub river: StreetSizing,
    /// Donk sizes for the turn; `None` means the engine's defaults.
    pub turn_donk: Option<String>,
    pub river_donk: Option<String>,
    /// Add an all-in action when the largest bet is within this multiple of the pot. `0` disables.
    pub add_allin_threshold: f64,
    /// Force all-in when the SPR after a call would be at or below this. `0` disables.
    pub force_allin_threshold: f64,
    /// Merge bet sizes that are within this fraction of each other (PioSOLVER's algorithm).
    pub merging_threshold: f64,
}

impl Default for Sizing {
    fn default() -> Self {
        Self {
            flop: StreetSizing::default(),
            turn: StreetSizing::default(),
            river: StreetSizing::default(),
            turn_donk: None,
            river_donk: None,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        }
    }
}

/// Rake, off by default because most of what gets solved is a tournament or a rake-capped pot
/// where it moves nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Rake {
    /// `0.0`–`1.0`.
    pub rate: f64,
    /// Absolute cap, in the same units as the pot.
    pub cap: f64,
}

/// When to stop iterating.
///
/// Both conditions are always live: whichever is hit first ends the solve, so a caller who asks
/// for an unreachable exploitability still gets an answer, and a caller who asks for a million
/// iterations still stops once the answer is good enough.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Stop {
    /// Hard iteration cap.
    pub max_iterations: u32,
    /// Absolute target, in pot units. Wins over [`Self::target_exploitability_pct`] if set.
    pub target_exploitability: Option<f32>,
    /// Target as a percentage of the starting pot. The industry convention is 0.5%.
    pub target_exploitability_pct: f64,
    /// How often to measure exploitability, in iterations.
    ///
    /// Measuring is a full best-response pass and costs about as much as an iteration, so doing
    /// it every time would halve throughput. Ten matches the engine's own `solve()`.
    pub check_interval: u32,
}

impl Default for Stop {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            target_exploitability: None,
            target_exploitability_pct: 0.5,
            check_interval: 10,
        }
    }
}

impl Stop {
    /// The absolute target this stop condition implies for a given pot.
    #[must_use]
    pub fn target_for(&self, starting_pot: i32) -> f32 {
        self.target_exploitability.unwrap_or_else(|| {
            (f64::from(starting_pot) * self.target_exploitability_pct / 100.0) as f32
        })
    }
}

/// Serde's default for [`Spot::compression_level`] — a function because `#[serde(default)]` on an
/// `Option` means `None`, and `None` here means "write it raw". `pub(crate)` because
/// `DumpSpec::compression_level` shares the same default by path.
pub(crate) fn default_compression_level() -> Option<i32> {
    Spot::default_compression_level()
}

/// Everything needed to prepare bunching-effect data: who folded, and on what flop.
///
/// The result is keyed by exactly these two things, which is why it is a job of its own rather
/// than a field the solve recomputes: preparing takes seconds to minutes and the answer is valid
/// for **every** solve on that flop — turn and river included, because only the first three board
/// cards have to match.
///
/// Each fold range must be suit-symmetric (class notation like `"A9o-A2o,K4s:0.73"` qualifies; a
/// suit-specific combo list like `"AhKd"` does not) — the engine refuses anything else, because
/// its tables are built on suit isomorphism. At most four fold players.
///
/// Solves that use the prepared data are markedly slower: terminal evaluation goes from
/// O(#OOP + #IP) to O(#OOP × #IP) private hands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BunchingSpec {
    /// One range per folded player, 1–4 of them, each suit-symmetric.
    pub fold_ranges: Vec<RangeSpec>,
    /// Exactly three cards. Solves referencing this preparation must share them (sorted).
    pub flop: BoardSpec,
    /// Caps the preparation's **peak** memory, temporary tables included; the four-fold-player
    /// case peaks at ~3.7 GB. Absent means [`crate::DEFAULT_MEMORY_LIMIT`].
    #[serde(default)]
    pub max_memory_bytes: Option<u64>,
    /// Where to write the prepared data when it finishes. Absent means keep it in memory only.
    #[serde(default)]
    pub save_path: Option<String>,
    /// zstd level for the file, as [`Spot::compression_level`].
    #[serde(default = "default_compression_level")]
    pub compression_level: Option<i32>,
    /// Free-text note stored alongside the saved data — which seats folded, typically.
    #[serde(default)]
    pub memo: Option<String>,
}

impl BunchingSpec {
    /// Parse the flop and the ranges, and let the engine judge the rest.
    ///
    /// The engine is the sole authority on suit-symmetry, emptiness and the fold-player cap —
    /// its checks are private, so they are not duplicated here; its message comes back verbatim
    /// inside [`SpotError::Bunching`]. The returned data is *unprocessed*: cheap to build, cheap
    /// to throw away, which is why both the submit path and the worker can afford to call this.
    ///
    /// # Errors
    ///
    /// If the flop is not exactly three cards, a range does not parse, or the engine refuses the
    /// configuration.
    pub fn validate(&self) -> Result<postflop_solver::BunchingData, SpotError> {
        let cards = self.flop.cards()?;
        if cards.len() != 3 {
            return Err(SpotError::Board {
                board: match &self.flop {
                    BoardSpec::Text(s) => s.clone(),
                    BoardSpec::Cards(v) => v.join(""),
                },
                reason: format!(
                    "a bunching flop is exactly three cards; got {}",
                    cards.len()
                ),
            });
        }

        let mut ranges = Vec::with_capacity(self.fold_ranges.len());
        for spec in &self.fold_ranges {
            let ours = spec.resolve()?;
            ranges.push(
                crate::convert::to_engine_range(&ours)
                    .map_err(|reason| SpotError::Bunching { reason })?,
            );
        }

        let flop = [
            crate::convert::to_engine_card(cards[0]),
            crate::convert::to_engine_card(cards[1]),
            crate::convert::to_engine_card(cards[2]),
        ];
        postflop_solver::BunchingData::new(&ranges, flop)
            .map_err(|reason| SpotError::Bunching { reason })
    }
}

/// How a solve names the bunching data it wants applied.
///
/// Untagged: the two wire shapes are told apart by their field name — `{"jobId":7}` names a
/// `prepareBunching`/`openBunching` job in this session, `{"path":"/x.bunching"}` a file loaded
/// when the solve starts. A typo'd field name matches neither variant and surfaces as serde's
/// generic untagged error inside `bad_request` — the same trade [`RangeSpec`] and [`BoardSpec`]
/// already accept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BunchingRef {
    /// A preparation job in this session.
    #[serde(rename_all = "camelCase")]
    Job { job_id: crate::jobs::JobId },
    /// A `.bunching` file on disk.
    File { path: String },
}

/// Everything needed to solve one spot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Spot {
    /// Out-of-position player's range (player 0).
    pub oop: RangeSpec,
    /// In-position player's range (player 1).
    pub ip: RangeSpec,
    pub board: BoardSpec,
    /// Pot at the start of the solved street.
    pub pot: i32,
    /// Effective stack behind, i.e. `min(stack_oop, stack_ip)`.
    pub effective_stack: i32,
    #[serde(default)]
    pub sizing: Sizing,
    #[serde(default)]
    pub rake: Rake,
    #[serde(default)]
    pub stop: Stop,
    /// Store node values as 16-bit integers rather than 32-bit floats: roughly half the memory
    /// for a small loss of precision. Worth it on a deep flop tree, pointless on a river.
    #[serde(default)]
    pub compress: bool,
    /// Refuse to allocate more than this. Absent means [`crate::DEFAULT_MEMORY_LIMIT`].
    ///
    /// This is the difference between "that tree is too big, here is how big" and the sidecar
    /// being killed by the OOM reaper mid-solve.
    #[serde(default)]
    pub max_memory_bytes: Option<u64>,
    /// Where to write the solved tree when it finishes. Absent means keep it in memory only.
    #[serde(default)]
    pub save_path: Option<String>,
    /// zstd level for any file written for this job. `null` writes the tree raw.
    ///
    /// One knob rather than one per `save` call, so every file a job writes is the same shape as
    /// the one its own `savePath` produced. Defaults to
    /// [`crate::engine::DEFAULT_COMPRESSION_LEVEL`].
    #[serde(default = "default_compression_level")]
    pub compression_level: Option<i32>,
    /// Free-text note stored alongside a saved solution — the hand id it came from, typically.
    #[serde(default)]
    pub memo: Option<String>,
    /// Extra action lines grafted onto the tree the sizing produced, before `removedLines`.
    ///
    /// Each line is a comma-separated path of actions in `NodeView.actions`' exact rendering
    /// (`"Bet(300)"`, `"Bet(300), Raise(900)"`); amounts are the street-cumulative integers
    /// NodeView shows. Chance actions are omitted — the tree treats all turn/river cards as
    /// one node. The subtree under an added action gets the spot's normal sizing, so its
    /// default lines can themselves be pruned by `removedLines` (adds apply first).
    #[serde(default)]
    pub added_lines: Vec<String>,
    /// Action lines pruned from the tree, applied after `addedLines`. Same grammar; the
    /// amounts must be the exact integers the engine derived (the numbers NodeView renders).
    #[serde(default)]
    pub removed_lines: Vec<String>,
    /// Strategies to pin before the CFR loop.
    ///
    /// Applied to the freshly built tree, in order, after memory allocation and before
    /// iteration 0; the solver then optimizes everything else *against* the pinned play.
    /// Empty means none. See [`Lock`] for the layout contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locks: Vec<Lock>,
    /// Bunching-effect data to apply before solving. Absent means none, which is every solve
    /// that existed before this field did.
    ///
    /// Only the first three board cards, sorted, must match the preparation's flop — the
    /// engine's own rule, which is what lets one preparation serve the flop, turn and river of
    /// the same hand. Expect the solve to be several times slower: terminal evaluation becomes
    /// O(#OOP × #IP). A `savePath` file written by a bunching solve does **not** carry the
    /// effect (the engine's format has no field for it), so keep the preparation job or its
    /// file around for reopening.
    #[serde(default)]
    pub bunching: Option<BunchingRef>,
}

/// Pin (part of) the acting player's strategy at one node before the solve.
///
/// The engine's semantics, which the wire inherits: a hand whose column is all zeros is left
/// **free** for the solver; a hand with any positive entry is **locked**, and the engine
/// normalizes its column to sum 1 (so the values need not sum to 1 themselves). Locking a node
/// below an isomorphism-eliminated chance card works — the engine converts the strategy into
/// storage coordinates itself — and locking the same storage node twice is last-write-wins.
///
/// A history that does not reach a decision node, a strategy of the wrong dimensions, or a
/// `hands` guard that does not match fails the **job** (`phase: "failed"`, the reason in
/// `error`): those need the built tree to check. Everything checkable from the request alone
/// fails the `solve` command synchronously.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lock {
    /// Action indices from the root — the `node` command's convention exactly: at a chance
    /// node the index is the dealt card's id (0–51). Empty means the root.
    #[serde(default)]
    pub history: Vec<usize>,
    /// Action-major, `NodeView.strategy`'s shape at that node: `strategy[a][h]`. Rows are the
    /// node's actions in `NodeView.actions` order; columns are the acting player's hands in
    /// `NodeView.hands` order (positive-weight combos, `(low, high)` pairs, lexicographic).
    /// Values must be finite and non-negative.
    pub strategy: Vec<Vec<f32>>,
    /// Optional guard against hand-order mistakes: if present, must equal the acting player's
    /// hand list at that node (`"QsQh"` style, highest card first), checked entry-by-entry
    /// before the lock is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hands: Option<Vec<String>>,
}

impl Spot {
    /// What [`Spot::compression_level`] is when the caller says nothing.
    #[must_use]
    pub const fn default_compression_level() -> Option<i32> {
        Some(crate::engine::DEFAULT_COMPRESSION_LEVEL)
    }

    /// The constructor "solve this spot" actually uses: the three things a hand already knows,
    /// plus the two ranges.
    ///
    /// Everything else takes its default, which is a solvable tree rather than a placeholder.
    #[must_use]
    pub fn from_hand(board: &[Card], pot: i32, effective_stack: i32, oop: &str, ip: &str) -> Self {
        Self {
            oop: RangeSpec::Notation(oop.to_owned()),
            ip: RangeSpec::Notation(ip.to_owned()),
            board: BoardSpec::Cards(board.iter().map(ToString::to_string).collect()),
            pot,
            effective_stack,
            sizing: Sizing::default(),
            rake: Rake::default(),
            stop: Stop::default(),
            compress: false,
            max_memory_bytes: None,
            save_path: None,
            compression_level: default_compression_level(),
            memo: None,
            added_lines: Vec::new(),
            removed_lines: Vec::new(),
            locks: Vec::new(),
            bunching: None,
        }
    }

    /// Validate everything that can be validated without building a tree.
    ///
    /// Called on the request thread so a typo comes back as a synchronous error on the `solve`
    /// command rather than as a job that fails a second later — the difference between a form
    /// that can show a field error and one that cannot.
    pub fn validate(&self) -> Result<Validated, SpotError> {
        if self.pot <= 0 {
            return Err(SpotError::Amount {
                field: "pot",
                reason: "must be greater than zero".to_owned(),
            });
        }
        if self.effective_stack <= 0 {
            return Err(SpotError::Amount {
                field: "effectiveStack",
                reason: "must be greater than zero".to_owned(),
            });
        }
        if !(0.0..=1.0).contains(&self.rake.rate) {
            return Err(SpotError::Amount {
                field: "rake.rate",
                reason: "must be between 0 and 1".to_owned(),
            });
        }
        if self.rake.cap < 0.0 {
            return Err(SpotError::Amount {
                field: "rake.cap",
                reason: "must not be negative".to_owned(),
            });
        }
        if self.stop.max_iterations == 0 {
            return Err(SpotError::Amount {
                field: "stop.maxIterations",
                reason: "must be at least one".to_owned(),
            });
        }
        if self.stop.check_interval == 0 {
            return Err(SpotError::Amount {
                field: "stop.checkInterval",
                reason: "must be at least one".to_owned(),
            });
        }

        let board = self.board.cards()?;
        let oop = self.oop.resolve()?;
        let ip = self.ip.resolve()?;
        for (label, range) in [("oop", &oop), ("ip", &ip)] {
            if range.without(&board).is_empty() {
                return Err(SpotError::EmptyRange { player: label });
            }
        }

        let initial_state = match board.len() {
            3 => BoardState::Flop,
            4 => BoardState::Turn,
            _ => BoardState::River,
        };

        let tree_config = TreeConfig {
            initial_state,
            starting_pot: self.pot,
            effective_stack: self.effective_stack,
            rake_rate: self.rake.rate,
            rake_cap: self.rake.cap,
            flop_bet_sizes: pair(self.sizing.flop.options("flop")?),
            turn_bet_sizes: pair(self.sizing.turn.options("turn")?),
            river_bet_sizes: pair(self.sizing.river.options("river")?),
            turn_donk_sizes: donk(self.sizing.turn_donk.as_deref(), "turnDonk")?,
            river_donk_sizes: donk(self.sizing.river_donk.as_deref(), "riverDonk")?,
            add_allin_threshold: self.sizing.add_allin_threshold,
            force_allin_threshold: self.sizing.force_allin_threshold,
            merging_threshold: self.sizing.merging_threshold,
        };

        // The syntactic half of lock validation — everything checkable without a tree fails
        // the command here; the structural half (does the history reach a decision node with
        // these dimensions?) runs in the worker and fails the job.
        for (index, lock) in self.locks.iter().enumerate() {
            let err = |reason: &str| SpotError::Lock {
                index,
                reason: reason.to_owned(),
            };
            let Some(first_row) = lock.strategy.first() else {
                return Err(err("`strategy` needs at least one action row"));
            };
            if first_row.is_empty() {
                return Err(err("`strategy` rows need at least one hand column"));
            }
            if lock.strategy.iter().any(|row| row.len() != first_row.len()) {
                return Err(err(
                    "every `strategy` row must have the same number of hands",
                ));
            }
            if lock
                .strategy
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err(err("`strategy` values must be finite and non-negative"));
            }
            if let Some(hands) = &lock.hands {
                if hands.len() != first_row.len() {
                    return Err(err(
                        "`hands` must have exactly one entry per `strategy` column",
                    ));
                }
            }
            if lock.history.contains(&usize::MAX) {
                return Err(err("`history` entries must be explicit indices"));
            }
        }

        // Tree edits: parse every line, then dry-run them against a throwaway tree so a bad
        // edit fails this command rather than a job a moment later. The dry-run only happens
        // when edits are present, and an `ActionTree::new` failure is left for `construct` to
        // report exactly as today.
        if self.added_lines.len() + self.removed_lines.len() > 512 {
            return Err(SpotError::Line {
                line: "…".to_owned(),
                reason: "too many lines (the cap is 512)".to_owned(),
            });
        }
        let added_lines = parse_lines(&self.added_lines)?;
        let removed_lines = parse_lines(&self.removed_lines)?;
        if !(added_lines.is_empty() && removed_lines.is_empty()) {
            if let Ok(mut tree) = ActionTree::new(tree_config.clone()) {
                apply_edits(&mut tree, &added_lines, &removed_lines)?;
            }
        }

        Ok(Validated {
            board,
            oop,
            ip,
            tree_config,
            added_lines,
            removed_lines,
        })
    }
}

/// Parse each edit line into engine actions, rejecting anything the grammar does not name.
fn parse_lines(lines: &[String]) -> Result<Vec<Vec<Action>>, SpotError> {
    lines
        .iter()
        .map(|line| {
            let err = |reason: String| SpotError::Line {
                line: line.clone(),
                reason,
            };
            let segments: Vec<&str> = line.split(',').map(str::trim).collect();
            if segments.iter().any(|s| s.is_empty()) {
                return Err(err(
                    "a line needs at least one action, and no empty ones".to_owned()
                ));
            }
            if segments.len() > 64 {
                return Err(err(
                    "too many actions in one line (the cap is 64)".to_owned()
                ));
            }
            segments
                .into_iter()
                .map(|s| crate::convert::action_from_str(s).map_err(&err))
                .collect()
        })
        .collect()
}

/// Render a parsed line back into the canonical grammar, for error messages.
fn render_line(line: &[Action]) -> String {
    if line.is_empty() {
        return "(root)".to_owned();
    }
    line.iter()
        .map(crate::convert::action_to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Apply parsed edits to a built tree: all adds (in order), then all removes (in order), then
/// refuse any edit set that left a node with no actions — `ActionTree::remove_line` has no
/// last-action guard, and the game constructor's later refusal names no line.
pub(crate) fn apply_edits(
    tree: &mut ActionTree,
    added: &[Vec<Action>],
    removed: &[Vec<Action>],
) -> Result<(), SpotError> {
    for line in added {
        tree.add_line(line).map_err(|reason| SpotError::Edit {
            op: "add",
            line: render_line(line),
            reason,
        })?;
    }
    for line in removed {
        tree.remove_line(line).map_err(|reason| SpotError::Edit {
            op: "remove",
            line: render_line(line),
            reason,
        })?;
    }
    if let Some(path) = tree.invalid_terminals().first() {
        return Err(SpotError::EmptyNode {
            line: render_line(path),
        });
    }
    Ok(())
}

fn pair(options: BetSizeOptions) -> [BetSizeOptions; 2] {
    [options.clone(), options]
}

fn donk(spec: Option<&str>, street: &'static str) -> Result<Option<DonkSizeOptions>, SpotError> {
    spec.map(|s| {
        DonkSizeOptions::try_from(s).map_err(|reason| SpotError::Sizing {
            street,
            reason: reason.to_string(),
        })
    })
    .transpose()
}

/// A [`Spot`] whose strings have all been parsed. Only the tree build can fail from here.
#[derive(Debug, Clone)]
pub struct Validated {
    pub board: Vec<Card>,
    pub oop: pkwiz_range::Range,
    pub ip: pkwiz_range::Range,
    pub tree_config: TreeConfig,
    pub added_lines: Vec<Vec<Action>>,
    pub removed_lines: Vec<Vec<Action>>,
}

/// A spot the caller described badly.
///
/// Every variant names the field it is about, because these are form errors and a message that
/// does not say which box is wrong is barely better than no message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpotError {
    #[error("`{notation}` is not a range: {reason}")]
    Range { notation: String, reason: String },
    #[error("`{board}` is not a board: {reason}")]
    Board { board: String, reason: String },
    #[error("{street} bet sizing is invalid: {reason}")]
    Sizing {
        street: &'static str,
        reason: String,
    },
    #[error("`{field}` {reason}")]
    Amount { field: &'static str, reason: String },
    #[error("the {player} range has no combinations left once the board is removed")]
    EmptyRange { player: &'static str },
    #[error("`locks[{index}]` is invalid: {reason}")]
    Lock { index: usize, reason: String },
    #[error("`{line}` is not an action line: {reason}")]
    Line { line: String, reason: String },
    #[error("cannot {op} the line `{line}`: {reason}")]
    Edit {
        op: &'static str,
        line: String,
        reason: String,
    },
    #[error("these edits leave the node after `{line}` with no actions")]
    EmptyNode { line: String },
    #[error("the bunching configuration is invalid: {reason}")]
    Bunching { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_board_is_the_street() {
        let flop = BoardSpec::Text("Td9d6h".to_owned()).cards().unwrap();
        assert_eq!(flop.len(), 3);
        assert_eq!(
            BoardSpec::Text("Td 9d 6h Qc".to_owned())
                .cards()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(
            BoardSpec::Cards(vec![
                "Td".into(),
                "9d".into(),
                "6h".into(),
                "Qc".into(),
                "2s".into()
            ])
            .cards()
            .unwrap()
            .len(),
            5
        );
        // And the parsed cards really are those cards, in order.
        assert_eq!(
            flop.iter().map(ToString::to_string).collect::<String>(),
            "Td9d6h"
        );
    }

    #[test]
    fn a_bad_board_says_what_is_wrong_with_it() {
        for (board, needle) in [
            ("Td9d", "three, four or five"),
            ("Td9d6hQc2s3c", "three, four or five"),
            ("Td9d6", "odd number"),
            ("TdTd6h", "appears twice"),
            ("TdXX6h", "X"),
        ] {
            let err = BoardSpec::Text(board.to_owned()).cards().unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "{board}: {err} lacks {needle}"
            );
        }
    }

    #[test]
    fn notation_goes_through_our_parser_not_the_engines() {
        // `rs-poker` drops the top endpoint of a low-first dashed range and
        // `pkwiz-eval` normalises it. The spot layer inherits that fix for free, which is the
        // whole reason the engine's own `Range: FromStr` is never called.
        let a = RangeSpec::Notation("JTs-67s".to_owned()).resolve().unwrap();
        let b = RangeSpec::Notation("JTs-76s".to_owned()).resolve().unwrap();
        assert_eq!(a.len(), 20);
        assert_eq!(a, b);
    }

    #[test]
    fn an_explicit_range_survives_the_wire_as_itself() {
        let spot: Spot = serde_json::from_str(
            r#"{"oop":[{"combo":"AsKd","weight":0.25}],"ip":"QQ+","board":"Td9d6h",
                "pot":100,"effectiveStack":500}"#,
        )
        .unwrap();
        let oop = spot.oop.resolve().unwrap();
        assert_eq!(oop.len(), 1);
        assert!((oop.total_weight() - 0.25).abs() < 1e-9);
        assert_eq!(spot.ip.resolve().unwrap().len(), 18);
        // Defaults filled in, not demanded.
        assert_eq!(spot.stop.max_iterations, 1000);
        assert!((spot.stop.target_for(100) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_what_the_engine_would_only_discover_later() {
        let base = || Spot::from_hand(&[], 100, 500, "QQ+", "QQ+");
        let with = |f: &dyn Fn(&mut Spot)| {
            let mut s = base();
            s.board = BoardSpec::Text("Td9d6h".to_owned());
            f(&mut s);
            s.validate().unwrap_err()
        };

        assert!(matches!(
            with(&|s| s.pot = 0),
            SpotError::Amount { field: "pot", .. }
        ));
        assert!(matches!(
            with(&|s| s.effective_stack = -1),
            SpotError::Amount {
                field: "effectiveStack",
                ..
            }
        ));
        assert!(matches!(
            with(&|s| s.rake.rate = 1.5),
            SpotError::Amount {
                field: "rake.rate",
                ..
            }
        ));
        assert!(matches!(
            with(&|s| s.stop.max_iterations = 0),
            SpotError::Amount { .. }
        ));
        assert!(matches!(
            with(&|s| s.sizing.flop.bet = "banana".to_owned()),
            SpotError::Sizing { street: "flop", .. }
        ));
        assert!(matches!(
            with(&|s| s.oop = RangeSpec::Notation("XX".to_owned())),
            SpotError::Range { .. }
        ));
        // A range wholly blocked by the board is empty, and the engine's message for that is
        // considerably less clear than ours.
        assert!(matches!(
            with(&|s| {
                s.board = BoardSpec::Text("TsTc6h".to_owned());
                s.oop = RangeSpec::Notation("TsTc".to_owned());
            }),
            SpotError::EmptyRange { player: "oop" }
        ));
    }

    #[test]
    fn from_hand_is_the_whole_request() {
        let board: Vec<Card> = ["Td", "9d", "6h", "Qc"]
            .iter()
            .map(|c| Card::parse(c).unwrap())
            .collect();
        let spot = Spot::from_hand(&board, 200, 900, "66+,A8s+", "QQ-22,AQs-A2s");
        let validated = spot.validate().unwrap();
        assert_eq!(validated.tree_config.initial_state, BoardState::Turn);
        assert_eq!(validated.tree_config.starting_pot, 200);
        assert_eq!(validated.tree_config.effective_stack, 900);
        assert_eq!(validated.board.len(), 4);
    }

    #[test]
    fn empty_sizing_is_a_check_through_tree_not_an_error() {
        let options = StreetSizing::none().options("river").unwrap();
        assert!(options.bet.is_empty() && options.raise.is_empty());
    }

    #[test]
    fn a_bunching_flop_is_exactly_three_cards() {
        let spec: BunchingSpec =
            serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"Td9d6hQc"}"#).unwrap();
        // `unwrap_err` needs `Debug` on the success type, which the engine's data lacks.
        let Err(err) = spec.validate() else {
            panic!("a four-card flop validated");
        };
        assert!(err.to_string().contains("exactly three cards"), "{err}");
    }

    #[test]
    fn a_bunching_spec_fills_its_defaults_like_a_spot_does() {
        let spec: BunchingSpec =
            serde_json::from_str(r#"{"foldRanges":["AA"],"flop":"Td9d6h"}"#).unwrap();
        assert_eq!(spec.compression_level, Some(3));
        assert_eq!(spec.save_path, None);
        assert_eq!(spec.max_memory_bytes, None);
        assert_eq!(spec.memo, None);
        // And the minimal spec validates: the engine accepts one suit-symmetric fold range.
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn a_bunching_ref_is_told_apart_by_its_field_name() {
        let job: BunchingRef = serde_json::from_str(r#"{"jobId":7}"#).unwrap();
        assert_eq!(job, BunchingRef::Job { job_id: 7 });
        let file: BunchingRef = serde_json::from_str(r#"{"path":"/x.bunching"}"#).unwrap();
        assert_eq!(
            file,
            BunchingRef::File {
                path: "/x.bunching".to_owned()
            }
        );
        // Round-trip: what a host reads back is what it sent.
        for r in [job, file] {
            let json = serde_json::to_string(&r).unwrap();
            assert_eq!(serde_json::from_str::<BunchingRef>(&json).unwrap(), r);
        }
    }

    #[test]
    fn a_spot_without_bunching_still_parses_old_wire_json_verbatim() {
        // The PROTOCOL_VERSION 1 guard: every solve request that worked before this field
        // existed must keep working, and must mean "no bunching".
        let spot: Spot = serde_json::from_str(
            r#"{"oop":"QQ+","ip":"QQ+","board":"Td9d6h","pot":100,"effectiveStack":500}"#,
        )
        .unwrap();
        assert_eq!(spot.bunching, None);
        assert!(spot.validate().is_ok());

        let with_ref: Spot = serde_json::from_str(
            r#"{"oop":"QQ+","ip":"QQ+","board":"Td9d6h","pot":100,"effectiveStack":500,
                "bunching":{"jobId":3}}"#,
        )
        .unwrap();
        assert_eq!(with_ref.bunching, Some(BunchingRef::Job { job_id: 3 }));
    }
}
