//! Hand-category classification for aggregation reports.
//!
//! The engine evaluates hands with a private `Hand` type and a private `hand_strength` table
//! (`src/hand.rs`, `src/card.rs` — both `pub(crate)`), so the sidecar cannot link them without
//! moving `ENGINE_REV` for nothing. This module re-derives the *category* — not the strength —
//! from rank/suit bitsets, exactly the way the engine's `Hand::evaluate_internal` classifies a
//! seven-card hand, and then subtypes the classes a report reader actually thinks in: a set is
//! not a trips, an overpair is not "one pair".
//!
//! A bug here mislabels a category; it can never change a strategy or an EV — those all come
//! from the engine. The taxonomy (where "second pair" ends and "weak pair" begins) is a product
//! decision; `formatVersion` on every report is the escape hatch if it ever has to move.
//!
//! Cards are engine ids (`rank * 4 + suit`, deuce 0 through ace 12, clubs 0 through spades 3 —
//! the encoding [`crate::convert`] verifies against both sides).

use serde::{Deserialize, Serialize};

/// The made-hand category of a hole pair on a board, strongest first.
///
/// Classes `Straight` and above are taken from the best five-card hand regardless of whether the
/// hole cards participate. Three-of-a-kind and the pair classes subtype by participation:
/// board-only trips, board-only two pair and board-only pairs all fall through to
/// [`Self::HighCard`], because a report reader asking "what does this range *hold*" does not
/// mean the board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MadeHand {
    StraightFlush,
    Quads,
    FullHouse,
    Flush,
    Straight,
    /// Three of a kind holding a pocket pair of the trip rank.
    Set,
    /// Three of a kind from a board pair plus one hole card.
    Trips,
    /// Two pair with at least one hole card in one of the two pairs.
    TwoPair,
    /// A pocket pair above the highest board rank.
    Overpair,
    /// A hole card pairing the highest board rank.
    TopPair,
    /// A hole card pairing the second-highest board rank.
    SecondPair,
    /// A hole card pairing any lower board rank.
    WeakPair,
    /// A pocket pair at or below the highest board rank (and not a set).
    Underpair,
    HighCard,
}

impl MadeHand {
    /// Every category, strongest first — the stable row order reports emit.
    pub const ALL: [Self; 14] = [
        Self::StraightFlush,
        Self::Quads,
        Self::FullHouse,
        Self::Flush,
        Self::Straight,
        Self::Set,
        Self::Trips,
        Self::TwoPair,
        Self::Overpair,
        Self::TopPair,
        Self::SecondPair,
        Self::WeakPair,
        Self::Underpair,
        Self::HighCard,
    ];
}

/// Draws a hand holds alongside its made category.
///
/// All `false` on a five-card board, and for made hands of [`MadeHand::Flush`] or better —
/// drawing to what you already beat is not a draw anyone charts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Draws {
    /// Exactly four cards of one suit among hole + board, at least one of them a hole card.
    pub flush_draw: bool,
    /// Two or more distinct ranks complete a straight using at least one hole card.
    pub open_ended: bool,
    /// Exactly one such rank.
    pub gutshot: bool,
}

/// The 13-bit rank masks of every straight, the wheel included.
const STRAIGHT_WINDOWS: [i32; 10] = [
    0b1_0000_0000_1111, // A-2-3-4-5
    0b0_0000_0001_1111,
    0b0_0000_0011_1110,
    0b0_0000_0111_1100,
    0b0_0000_1111_1000,
    0b0_0001_1111_0000,
    0b0_0011_1110_0000,
    0b0_0111_1100_0000,
    0b0_1111_1000_0000,
    0b1_1111_0000_0000, // T-J-Q-K-A
];

#[inline]
fn has_straight(rankset: i32) -> bool {
    // Every window has five bits, so a full intersection is a made straight.
    STRAIGHT_WINDOWS
        .iter()
        .any(|&w| (rankset & w).count_ones() == 5)
}

/// Classify one hole pair against a board of three, four or five cards.
///
/// `hole` and `board` are engine card ids (0–51); the board must not overlap the hole cards —
/// the caller gets both from the same [`postflop_solver::PostFlopGame`], which guarantees it.
#[must_use]
pub fn classify(hole: (u8, u8), board: &[u8]) -> (MadeHand, Draws) {
    debug_assert!((3..=5).contains(&board.len()), "a postflop board");

    let hole_ranks = [i32::from(hole.0) / 4, i32::from(hole.1) / 4];
    let pocket_pair = hole_ranks[0] == hole_ranks[1];

    let mut rankset = 0i32;
    let mut rankset_suit = [0i32; 4];
    let mut rank_count = [0u8; 13];
    let mut board_rankset = 0i32;
    for &card in board {
        let (rank, suit) = (usize::from(card) / 4, usize::from(card) % 4);
        rankset |= 1 << rank;
        rankset_suit[suit] |= 1 << rank;
        rank_count[rank] += 1;
        board_rankset |= 1 << rank;
    }
    for &card in &[hole.0, hole.1] {
        let (rank, suit) = (usize::from(card) / 4, usize::from(card) % 4);
        rankset |= 1 << rank;
        rankset_suit[suit] |= 1 << rank;
        rank_count[rank] += 1;
    }
    let hole_rankset = (1 << hole_ranks[0]) | (1 << hole_ranks[1]);
    // The highest board rank, and the second-highest *distinct* board rank.
    let top_board = 31 - board_rankset.leading_zeros() as i32;
    let second_board = {
        let rest = board_rankset & !(1 << top_board);
        if rest == 0 {
            -1
        } else {
            31 - rest.leading_zeros() as i32
        }
    };

    let flush_suit = (0..4).find(|&s| rankset_suit[s].count_ones() >= 5);
    let trip_rank = (0..13).rev().find(|&r| rank_count[r] == 3);
    let pair_ranks: Vec<usize> = (0..13).rev().filter(|&r| rank_count[r] == 2).collect();

    let made = if let Some(suit) = flush_suit {
        if has_straight(rankset_suit[suit]) {
            MadeHand::StraightFlush
        } else {
            MadeHand::Flush
        }
    } else if rank_count.contains(&4) {
        MadeHand::Quads
    } else if trip_rank.is_some() && (rank_count.iter().filter(|&&c| c >= 2).count() >= 2) {
        // Two trips, or trips plus a pair: either way a full house.
        MadeHand::FullHouse
    } else if has_straight(rankset) {
        MadeHand::Straight
    } else if let Some(t) = trip_rank {
        // Three of a kind: subtype by how the hole participates.
        let t = t as i32;
        if pocket_pair && hole_ranks[0] == t {
            MadeHand::Set
        } else if board_rankset & (1 << t) != 0
            && rank_count_on(board, t) == 2
            && (hole_ranks[0] == t || hole_ranks[1] == t)
        {
            MadeHand::Trips
        } else {
            // Trips entirely on the board: the hole holds kickers, nothing more.
            MadeHand::HighCard
        }
    } else if pair_ranks.len() >= 2 {
        // Two pair, judged on the two pairs the best five-card hand uses (the top two).
        let top_two = [pair_ranks[0] as i32, pair_ranks[1] as i32];
        if top_two.contains(&hole_ranks[0]) || top_two.contains(&hole_ranks[1]) {
            MadeHand::TwoPair
        } else {
            classify_pairless(
                pocket_pair,
                hole_ranks,
                board_rankset,
                top_board,
                second_board,
            )
        }
    } else if pair_ranks.len() == 1 {
        let pair = pair_ranks[0] as i32;
        if pocket_pair && hole_ranks[0] == pair {
            if pair > top_board {
                MadeHand::Overpair
            } else {
                MadeHand::Underpair
            }
        } else if hole_ranks.contains(&pair) {
            // One hole card pairing a board rank; where on the board says which pair it is.
            if pair == top_board {
                MadeHand::TopPair
            } else if pair == second_board {
                MadeHand::SecondPair
            } else {
                MadeHand::WeakPair
            }
        } else {
            // The pair is entirely on the board.
            MadeHand::HighCard
        }
    } else {
        MadeHand::HighCard
    };

    let draws = if board.len() < 5 && made > MadeHand::Flush {
        compute_draws(rankset, rankset_suit, hole, hole_rankset)
    } else {
        Draws::default()
    };

    (made, draws)
}

/// The pair classing for a hand whose best-five two pair is entirely on the board: the hole may
/// still hold a (lower) pocket pair or pair a board rank the best hand does not use — but a
/// pair the best hand does not use is not a made pair, so everything here is board-relative.
fn classify_pairless(
    pocket_pair: bool,
    hole_ranks: [i32; 2],
    board_rankset: i32,
    top_board: i32,
    second_board: i32,
) -> MadeHand {
    if pocket_pair && board_rankset & (1 << hole_ranks[0]) == 0 {
        // A pocket pair below both board pairs is still a pair in the reader's sense.
        if hole_ranks[0] > top_board {
            MadeHand::Overpair
        } else {
            MadeHand::Underpair
        }
    } else if board_rankset & ((1 << hole_ranks[0]) | (1 << hole_ranks[1])) != 0 && !pocket_pair {
        let paired = if board_rankset & (1 << hole_ranks[0]) != 0 {
            hole_ranks[0]
        } else {
            hole_ranks[1]
        };
        if paired == top_board {
            MadeHand::TopPair
        } else if paired == second_board {
            MadeHand::SecondPair
        } else {
            MadeHand::WeakPair
        }
    } else {
        MadeHand::HighCard
    }
}

/// How many board cards hold rank `r`.
fn rank_count_on(board: &[u8], r: i32) -> u8 {
    board.iter().filter(|&&c| i32::from(c) / 4 == r).count() as u8
}

fn compute_draws(rankset: i32, rankset_suit: [i32; 4], hole: (u8, u8), hole_rankset: i32) -> Draws {
    // Flush draw: exactly four of one suit among hole + board, with a hole card of that suit.
    let hole_suits = [usize::from(hole.0) % 4, usize::from(hole.1) % 4];
    let flush_draw = (0..4).any(|s| {
        // Suited hole cards share a rankset bit only if they share a rank, which they cannot;
        // count_ones over the per-suit rankset is the card count.
        rankset_suit[s].count_ones() == 4 && hole_suits.contains(&s)
    });

    // Straight draws: distinct absent ranks whose addition completes a straight that uses at
    // least one hole card. Two or more is open-ended (double-gutshots included, deliberately),
    // exactly one is a gutshot.
    let mut outs = 0u32;
    for r in 0..13 {
        if rankset & (1 << r) != 0 {
            continue;
        }
        let with = rankset | (1 << r);
        let completes = STRAIGHT_WINDOWS
            .iter()
            .any(|&w| with & w == w && w & (1 << r) != 0 && w & hole_rankset != 0);
        if completes {
            outs += 1;
        }
    }

    Draws {
        flush_draw,
        open_ended: outs >= 2,
        gutshot: outs == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `"AsKd"` → engine ids, so the cases below read like hands.
    fn cards(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|c| {
                let rank = b"23456789TJQKA"
                    .iter()
                    .position(|&r| r == c[0])
                    .expect("a rank") as u8;
                let suit = b"cdhs".iter().position(|&x| x == c[1]).expect("a suit") as u8;
                rank * 4 + suit
            })
            .collect()
    }

    fn run(hole: &str, board: &str) -> (MadeHand, Draws) {
        let h = cards(hole);
        classify((h[0], h[1]), &cards(board))
    }

    #[test]
    fn made_hand_categories_follow_the_table() {
        use MadeHand::*;
        let no = Draws::default();
        for (hole, board, made) in [
            // The pair family on an unpaired flop.
            ("AsAh", "Td9d6h", Overpair),
            ("JdJh", "Td9d6h", Overpair),  // just above top card
            ("8c8h", "Td9d6h", Underpair), // between board ranks
            ("5c5h", "Td9d6h", Underpair),
            ("9c9h", "Td9d6h", Set),
            ("AcTc", "Td9d6h", TopPair),
            ("Ac9c", "Td9d6h", SecondPair),
            ("Ac6c", "Td9d6h", WeakPair),
            ("AcKd", "Td9d6h", HighCard),
            ("Th9h", "Td9d6h", TwoPair),
            // Paired boards: set vs trips vs board-only fall-through.
            ("Ac8d", "TdTh6h2c", HighCard), // board pair, no participation
            ("AcTc", "TdTh6h2c", Trips),    // board pair + hole card
            ("6c6s", "TdTh2h3c", TwoPair),  // pocket pair below the board pair still plays
            ("6c6s", "TdTh6h3c", FullHouse),
            ("Ac2d", "TdTh2h3c", TwoPair), // hole card in the second pair
            ("AcKd", "TdThTs2c", HighCard), // board trips, kickers only
            ("2c2d", "TdThTs4c", FullHouse),
            // The strong finals, hole participation irrelevant by rule.
            ("2c3c", "AsKsQsJsTs", StraightFlush),
            ("AcAd", "AsAhKs2d3c", Quads),
            ("AcKc", "AsAhKs2d", FullHouse),
            ("Ad2d", "KdQd7d6c", Flush),
            ("8d7c", "Td9h6h2c", Straight),
            ("AcKd", "QhJsTh9c2d", Straight), // board does most of the work
        ] {
            let (m, _) = run(hole, board);
            assert_eq!(m, made, "{hole} on {board}");
            let _ = no;
        }
    }

    #[test]
    fn draws_are_flush_and_straight_shaped() {
        // Open-ended plus flush draw: four diamonds, and both 6 and J complete a straight
        // through the hole cards.
        let (m, d) = run("8d7d", "Td9d2h");
        assert_eq!(m, MadeHand::HighCard);
        assert!(d.flush_draw && d.open_ended && !d.gutshot, "{d:?}");

        // Open-ended without the flush draw.
        let (m, d) = run("QcJc", "Td9d6h");
        assert_eq!(m, MadeHand::HighCard);
        assert!(!d.flush_draw && d.open_ended && !d.gutshot, "{d:?}");

        // Gutshot plus flush draw: only the jack fills K-Q-T-9.
        let (m, d) = run("KdQd", "Td9d6h");
        assert_eq!(m, MadeHand::HighCard);
        assert!(d.flush_draw && !d.open_ended && d.gutshot, "{d:?}");

        // The wheel gutshot: only a 5 completes A-2-3-4.
        let (_, d) = run("Ac2c", "3d4h9s");
        assert!(!d.open_ended && d.gutshot, "{d:?}");

        // Two hole diamonds are only a backdoor draw: three of a suit is not a flush draw.
        let (_, d) = run("Ad2d", "Kd7c8s");
        assert!(!d.flush_draw, "{d:?}");

        // Four to a flush on the board with no hole card of the suit is the board's draw.
        let (_, d) = run("AcKc", "2d5d9dTd");
        assert!(!d.flush_draw, "{d:?}");

        // A board-only draw: 5 or T completes a straight using no hole card, so it is the
        // board's draw, not this hand's.
        let (_, d) = run("Ac2c", "9d8h7s6s");
        assert!(!d.open_ended && !d.gutshot, "{d:?}");
    }

    #[test]
    fn river_boards_and_strong_hands_report_no_draws() {
        // Five-card board: the hand is what it is.
        let (_, d) = run("8d7d", "Td9d2h3c4s");
        assert_eq!(d, Draws::default());

        // A made flush does not also hold its own draw.
        let (m, d) = run("Ad2d", "KdQd7d6c");
        assert_eq!(m, MadeHand::Flush);
        assert_eq!(d, Draws::default());

        // A made straight below a flush still reports draws to better hands.
        let (m, d) = run("8h7h", "Td9h6h2c");
        assert_eq!(m, MadeHand::Straight);
        assert!(d.flush_draw, "{d:?}");
    }

    #[test]
    fn two_pair_on_the_board_falls_through_to_the_hole() {
        use MadeHand::*;
        for (hole, board, made) in [
            ("AcKd", "TdTh6h6c", HighCard), // no participation at all
            // AA makes aces-up: the hole pair joins the best five, so it is two pair, not an
            // overpair — the rule is participation in the best hand's pairs.
            ("AcAd", "TdTh6h6c", TwoPair),
            ("2c2d", "TdTh6h6c", Underpair), // below both board pairs: not in the best five
            ("Ac6d", "TdTh6h6c", FullHouse), // pairing the lower board pair fills up
            ("Ac6d", "TdTh6h2c", TwoPair),   // hole card in the lower pair of an unfilled board
        ] {
            let (m, _) = run(hole, board);
            assert_eq!(m, made, "{hole} on {board}");
        }
    }
}
