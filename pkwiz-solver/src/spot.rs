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
use postflop_solver::{BetSizeOptions, BoardState, DonkSizeOptions, TreeConfig};
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
/// `Option` means `None`, and `None` here means "write it raw".
fn default_compression_level() -> Option<i32> {
    Spot::default_compression_level()
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

        Ok(Validated {
            board,
            oop,
            ip,
            tree_config,
        })
    }
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
}
