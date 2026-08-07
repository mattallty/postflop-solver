use super::*;

use crate::interface::*;
use crate::utility::*;
use std::cell::Cell;
use std::ptr;

use bincode::{
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
};

impl PostFlopGame {
    /// Returns the storage mode of this instance.
    ///
    /// The storage mode represents the deepest accessible node in the game tree.
    /// For example, if the storage mode is `BoardState::Turn`, then the game tree
    /// contains no information after the river deal.
    #[inline]
    pub fn storage_mode(&self) -> BoardState {
        self.storage_mode
    }

    /// Returns the target storage mode, which is used for serialization.
    #[inline]
    pub fn target_storage_mode(&self) -> BoardState {
        self.target_storage_mode
    }

    /// Sets the target storage mode.
    #[inline]
    pub fn set_target_storage_mode(&mut self, mode: BoardState) -> Result<(), String> {
        if mode > self.storage_mode {
            return Err("Cannot set target to a higher value than the current storage".to_string());
        }

        if mode < self.tree_config.initial_state {
            return Err("Cannot set target to a lower value than the initial state".to_string());
        }

        self.target_storage_mode = mode;
        Ok(())
    }

    /// Returns the memory usage when the target storage mode is used for serialization.
    #[inline]
    pub fn target_memory_usage(&self) -> u64 {
        match self.target_storage_mode {
            BoardState::River => match self.is_compression_enabled {
                false => self.memory_usage().0,
                true => self.memory_usage().1,
            },
            _ => {
                let num_target_storage = self.num_target_storage();
                num_target_storage.iter().map(|&x| x as u64).sum::<u64>() + self.misc_memory_usage
            }
        }
    }

    /// Structural validation of a freshly decoded game, run before anything walks the tree.
    ///
    /// The per-node storage offsets are already bounds-checked during decoding (see
    /// `checked_offset`); what remains is everything whose consistency spans nodes. A file is a
    /// trust boundary: every check here corresponds to a way a corrupt or crafted file could
    /// otherwise reach an out-of-bounds read or a panic through safe methods.
    ///
    /// Must run after `check_card_config` and `init_card_fields` — the node counts are
    /// recomputed from the decoded action tree and card config, which is what makes the
    /// decoded `num_nodes` a claim that can be checked rather than trusted.
    fn validate_decoded(&mut self) -> Result<(), String> {
        // `num_nodes` drives `allocate_memory` and the street indexing; recompute it from the
        // tree the file actually contains.
        let counted = self.count_num_nodes();
        if counted != self.num_nodes {
            return Err(format!(
                "corrupt file: node counts {:?} do not match the action tree, which yields {:?}",
                self.num_nodes, counted,
            ));
        }

        // The arena must hold exactly the prefix its storage mode claims.
        let expected_len = match self.storage_mode {
            BoardState::Flop => counted[0],
            BoardState::Turn => counted[0] + counted[1],
            BoardState::River => counted[0] + counted[1] + counted[2],
        };
        if self.node_arena.len() as u64 != expected_len {
            return Err(format!(
                "corrupt file: node arena holds {} nodes where its storage mode requires {}",
                self.node_arena.len(),
                expected_len,
            ));
        }

        let arena_len = self.node_arena.len();
        for index in 0..arena_len {
            let node = self.node_arena[index].lock();

            // Terminal evaluation and the isomorphism tables index by these cards, and the
            // per-board tables (`hand_strength`, `valid_indices`) are populated only for
            // boards the card config can deal. A card pair that is well-formed but impossible
            // — a turn equal to a flop card, a river the config fixed differently — lands on
            // an empty table entry, whose sentinel arithmetic underflows in debug and reads
            // out of bounds through `get_unchecked` in release. So a node's cards must be
            // exactly what the config allows dealt there.
            let config = &self.card_config;
            let flop = config.flop;
            let turn_ok = if config.turn != NOT_DEALT {
                node.turn == config.turn
            } else {
                node.turn == NOT_DEALT || (node.turn < 52 && !flop.contains(&node.turn))
            };
            let river_ok = if config.river != NOT_DEALT {
                node.river == config.river
            } else {
                node.river == NOT_DEALT
                    || (node.river < 52
                        && !flop.contains(&node.river)
                        && node.turn != NOT_DEALT
                        && node.river != node.turn)
            };
            if !turn_ok || !river_ok {
                return Err(format!(
                    "corrupt file: node {index} carries board cards {}/{}, which this \
                     configuration cannot deal",
                    node.turn, node.river,
                ));
            }

            if node.is_terminal() {
                if node.num_children != 0 || node.num_elements != 0 || node.num_elements_ip != 0 {
                    return Err(format!(
                        "corrupt file: terminal node {index} claims children or storage",
                    ));
                }
                continue;
            }

            // The element counts drive every storage slice the accessors and the finalizer
            // build, and each has exactly one value the tree structure allows. (The decode-time
            // bounds checks used the counts from the file; pinning them here makes those
            // checks checks of the true sizes.)
            let (expected_elements, expected_elements_ip) = if node.is_chance() {
                let elements = node
                    .cfvalue_storage_player()
                    .map_or(0, |player| self.private_cards[player].len());
                (elements, 0)
            } else {
                if node.player > 1 {
                    return Err(format!(
                        "corrupt file: node {index} names player {}, which is not a player",
                        node.player,
                    ));
                }
                let elements =
                    node.num_children as usize * self.private_cards[node.player as usize].len();
                let elements_ip = match node.prev_action {
                    Action::None | Action::Chance(_) => self.private_cards[1].len(),
                    _ => 0,
                };
                (elements, elements_ip)
            };
            if node.num_elements as usize != expected_elements
                || node.num_elements_ip as usize != expected_elements_ip
            {
                return Err(format!(
                    "corrupt file: node {index} claims {}/{} storage elements where its \
                     structure requires {expected_elements}/{expected_elements_ip}",
                    node.num_elements, node.num_elements_ip,
                ));
            }

            // A chance node at the boundary of a reduced storage mode legitimately points past
            // the arena; the access guards refuse those. Everything else must stay inside it.
            let is_boundary = node.is_chance()
                && match node.turn {
                    NOT_DEALT => self.storage_mode == BoardState::Flop,
                    _ => self.storage_mode <= BoardState::Turn,
                };
            if !is_boundary {
                let end = (node.children_offset as usize)
                    .checked_add(node.num_children as usize)
                    .and_then(|span| span.checked_add(index));
                // `children_offset == 0` would make a node its own child: every recursive
                // traversal (the solver, the finalizer run at the end of this very decode)
                // would then recurse without end. The builder always lays children strictly
                // after their parent, so offsets of at least 1 also keep the tree forward-only
                // and every traversal finite.
                if node.num_children == 0
                    || node.children_offset == 0
                    || end.is_none_or(|end| end > arena_len)
                {
                    return Err(format!(
                        "corrupt file: node {index} points at children {}..+{}, which is not a \
                         forward range inside the arena of {arena_len}",
                        node.children_offset, node.num_children,
                    ));
                }
            }

            // A locked node's strategy is read with `.unwrap()` and trusted for its length.
            if node.is_locked {
                if node.is_chance() {
                    return Err(format!("corrupt file: chance node {index} claims a lock"));
                }
                let expected =
                    node.num_children as usize * self.private_cards[node.player as usize].len();
                match self.locking_strategy.get(&index) {
                    Some(strategy) if strategy.len() == expected => {}
                    Some(strategy) => {
                        return Err(format!(
                            "corrupt file: node {index} lock has {} weights where its \
                             strategy needs {expected}",
                            strategy.len(),
                        ));
                    }
                    None => {
                        return Err(format!(
                            "corrupt file: node {index} is locked but carries no strategy"
                        ));
                    }
                }
            }
        }

        // The reverse direction: every stored lock must belong to a locked node in the arena.
        for &index in self.locking_strategy.keys() {
            if index >= arena_len || !self.node_arena[index].lock().is_locked {
                return Err(format!(
                    "corrupt file: a lock is stored for node {index}, which is not a locked \
                     node of the arena"
                ));
            }
        }

        Ok(())
    }

    /// Returns the number of storage elements required for the target storage mode.
    fn num_target_storage(&self) -> [usize; 4] {
        if self.state <= State::TreeBuilt {
            return [0; 4];
        }

        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        if self.target_storage_mode == BoardState::River {
            // omit storing the counterfactual values
            return [num_bytes * self.num_storage as usize, 0, 0, 0];
        }

        let mut node_index = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0],
            _ => self.num_nodes[0] + self.num_nodes[1],
        } as usize;

        let mut num_storage = [0; 4];

        while num_storage.contains(&0) {
            node_index -= 1;
            let node = self.node_arena[node_index].lock();
            if num_storage[0] == 0 && !node.is_terminal() && !node.is_chance() {
                let offset = unsafe { node.storage1.offset_from(self.storage1.as_ptr()) };
                let offset_ip = unsafe { node.storage3.offset_from(self.storage_ip.as_ptr()) };
                let len = num_bytes * node.num_elements as usize;
                let len_ip = num_bytes * node.num_elements_ip as usize;
                num_storage[0] = offset as usize + len;
                num_storage[1] = offset as usize + len;
                num_storage[2] = offset_ip as usize + len_ip;
            }
            if num_storage[3] == 0 && node.is_chance() {
                let offset = unsafe { node.storage1.offset_from(self.storage_chance.as_ptr()) };
                let len = num_bytes * node.num_elements as usize;
                num_storage[3] = offset as usize + len;
            }
        }

        num_storage
    }
}

static VERSION_STR: &str = "2023-03-19";

thread_local! {
    static PTR_BASE: Cell<[*const u8; 2]> = const { Cell::new([ptr::null(); 2]) };
    static CHANCE_BASE: Cell<*const u8> = const { Cell::new(ptr::null()) };
    static PTR_BASE_MUT: Cell<[*mut u8; 3]> = const { Cell::new([ptr::null_mut(); 3]) };
    static CHANCE_BASE_MUT: Cell<*mut u8> = const { Cell::new(ptr::null_mut()) };
    /// Lengths of the buffers the mutable bases point into — `[storage1, storage2, storage_ip,
    /// storage_chance]` — plus the element width, so `PostFlopNode::decode` can refuse an
    /// offset that would place a node's storage outside its buffer instead of building a
    /// pointer from whatever the file says.
    static STORAGE_LENS: Cell<[usize; 4]> = const { Cell::new([0; 4]) };
    static STORAGE_NUM_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Validates one decoded storage offset: in bounds, aligned to the element width, with room for
/// `num_elements` elements before the end of a buffer of `buffer_len` bytes.
fn checked_offset(
    offset: isize,
    num_elements: usize,
    num_bytes: usize,
    buffer_len: usize,
    what: &str,
) -> Result<usize, DecodeError> {
    let err = || {
        DecodeError::OtherString(format!(
            "corrupt file: {what} offset {offset} with {num_elements} elements does not fit \
             its buffer of {buffer_len} bytes"
        ))
    };
    let offset = usize::try_from(offset).map_err(|_| err())?;
    if offset % num_bytes.max(1) != 0 {
        return Err(err());
    }
    let len = num_elements.checked_mul(num_bytes).ok_or_else(err)?;
    let end = offset.checked_add(len).ok_or_else(err)?;
    if end > buffer_len {
        return Err(err());
    }
    Ok(offset)
}

impl Encode for PostFlopGame {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        if self.state < State::MemoryAllocated {
            return Err(EncodeError::Other("Game's memory is not allocated"));
        }

        let num_storage = self.num_target_storage();

        // version
        VERSION_STR.to_string().encode(encoder)?;

        // contents
        self.state.encode(encoder)?;
        self.card_config.encode(encoder)?;
        self.tree_config.encode(encoder)?;
        self.added_lines.encode(encoder)?;
        self.removed_lines.encode(encoder)?;
        self.action_root.encode(encoder)?;
        self.target_storage_mode.encode(encoder)?;
        self.num_nodes.encode(encoder)?;
        self.is_compression_enabled.encode(encoder)?;
        self.num_storage.encode(encoder)?;
        self.num_storage_ip.encode(encoder)?;
        self.num_storage_chance.encode(encoder)?;
        self.misc_memory_usage.encode(encoder)?;
        self.storage1[0..num_storage[0]].encode(encoder)?;
        self.storage2[0..num_storage[1]].encode(encoder)?;
        self.storage_ip[0..num_storage[2]].encode(encoder)?;
        self.storage_chance[0..num_storage[3]].encode(encoder)?;

        let num_nodes = match self.target_storage_mode {
            BoardState::Flop => self.num_nodes[0] as usize,
            BoardState::Turn => (self.num_nodes[0] + self.num_nodes[1]) as usize,
            BoardState::River => self.node_arena.len(),
        };

        // locking strategy (need to filter)
        let mut locking_strategy = self.locking_strategy.clone();
        locking_strategy.retain(|&i, _| i < num_nodes);
        locking_strategy.encode(encoder)?;

        // store base pointers
        PTR_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set([self.storage1.as_ptr(), self.storage_ip.as_ptr()]);
            } else {
                c.set([ptr::null(); 2]);
            }
        });

        CHANCE_BASE.with(|c| {
            if self.state >= State::MemoryAllocated {
                c.set(self.storage_chance.as_ptr());
            } else {
                c.set(ptr::null());
            }
        });

        // game tree
        self.node_arena[0..num_nodes].encode(encoder)?;

        Ok(())
    }
}

impl<Context> Decode<Context> for PostFlopGame {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        // version check
        let version = String::decode(decoder)?;
        if version != VERSION_STR {
            return Err(DecodeError::OtherString(format!(
                "Version mismatch: expected '{VERSION_STR}', but got '{version}'"
            )));
        }

        // game instance
        let mut game = Self {
            state: Decode::decode(decoder)?,
            card_config: Decode::decode(decoder)?,
            tree_config: Decode::decode(decoder)?,
            added_lines: Decode::decode(decoder)?,
            removed_lines: Decode::decode(decoder)?,
            action_root: Decode::decode(decoder)?,
            storage_mode: Decode::decode(decoder)?,
            num_nodes: Decode::decode(decoder)?,
            is_compression_enabled: Decode::decode(decoder)?,
            num_storage: Decode::decode(decoder)?,
            num_storage_ip: Decode::decode(decoder)?,
            num_storage_chance: Decode::decode(decoder)?,
            misc_memory_usage: Decode::decode(decoder)?,
            storage1: Decode::decode(decoder)?,
            storage2: Decode::decode(decoder)?,
            storage_ip: Decode::decode(decoder)?,
            storage_chance: Decode::decode(decoder)?,
            locking_strategy: Decode::decode(decoder)?,
            ..Default::default()
        };

        game.target_storage_mode = game.storage_mode;
        if game.storage_mode == BoardState::River && game.state >= State::MemoryAllocated {
            let num_bytes = if game.is_compression_enabled { 2 } else { 4 };
            // The encoder stores exactly `num_bytes * num_storage` bytes of `storage1` for a
            // river-mode save, so a mismatch means the counters and the content disagree —
            // refuse before sizing fresh allocations from the forged counter.
            if game.storage1.len() as u64 != num_bytes * game.num_storage {
                return Err(DecodeError::OtherString(format!(
                    "corrupt file: storage length {} does not match its counter {}",
                    game.storage1.len(),
                    num_bytes * game.num_storage,
                )));
            }
            game.storage2 = vec![0; (num_bytes * game.num_storage) as usize];
            game.storage_ip = vec![0; (num_bytes * game.num_storage_ip) as usize];
            game.storage_chance = vec![0; (num_bytes * game.num_storage_chance) as usize];
        }

        // store base pointers
        PTR_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set([
                    game.storage1.as_mut_ptr(),
                    game.storage2.as_mut_ptr(),
                    game.storage_ip.as_mut_ptr(),
                ]);
            } else {
                c.set([ptr::null_mut(); 3]);
            }
        });

        CHANCE_BASE_MUT.with(|c| {
            if game.state >= State::MemoryAllocated {
                c.set(game.storage_chance.as_mut_ptr());
            } else {
                c.set(ptr::null_mut());
            }
        });

        STORAGE_LENS.with(|c| {
            c.set([
                game.storage1.len(),
                game.storage2.len(),
                game.storage_ip.len(),
                game.storage_chance.len(),
            ]);
        });
        STORAGE_NUM_BYTES.with(|c| c.set(if game.is_compression_enabled { 2 } else { 4 }));

        // game tree
        game.node_arena = Decode::decode(decoder)?;

        // initialization — the structural validation runs before anything walks the tree
        // (`back_to_root` already reads the arena)
        game.check_card_config().map_err(DecodeError::OtherString)?;
        game.init_card_fields();
        game.validate_decoded().map_err(DecodeError::OtherString)?;
        game.init_interpreter();
        game.back_to_root();

        // restore the counterfactual values
        if game.storage_mode == BoardState::River && game.state == State::Solved {
            game.state = State::MemoryAllocated;
            finalize(&mut game);
        }

        Ok(game)
    }
}

impl Encode for PostFlopNode {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // contents
        self.prev_action.encode(encoder)?;
        self.player.encode(encoder)?;
        self.turn.encode(encoder)?;
        self.river.encode(encoder)?;
        self.is_locked.encode(encoder)?;
        self.amount.encode(encoder)?;
        self.children_offset.encode(encoder)?;
        self.num_children.encode(encoder)?;
        self.num_elements_ip.encode(encoder)?;
        self.num_elements.encode(encoder)?;
        self.scale1.encode(encoder)?;
        self.scale2.encode(encoder)?;
        self.scale3.encode(encoder)?;

        // pointer offset
        if !self.storage1.is_null() {
            if self.is_terminal() {
                // do nothing
            } else if self.is_chance() {
                let base = CHANCE_BASE.with(|c| c.get());
                unsafe { self.storage1.offset_from(base).encode(encoder)? };
            } else {
                let bases = PTR_BASE.with(|c| c.get());
                unsafe {
                    self.storage1.offset_from(bases[0]).encode(encoder)?;
                    self.storage3.offset_from(bases[1]).encode(encoder)?;
                }
            }
        }

        Ok(())
    }
}

impl<Context> Decode<Context> for PostFlopNode {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        // node instance
        let mut node = Self {
            prev_action: Decode::decode(decoder)?,
            player: Decode::decode(decoder)?,
            turn: Decode::decode(decoder)?,
            river: Decode::decode(decoder)?,
            is_locked: Decode::decode(decoder)?,
            amount: Decode::decode(decoder)?,
            children_offset: Decode::decode(decoder)?,
            num_children: Decode::decode(decoder)?,
            num_elements_ip: Decode::decode(decoder)?,
            num_elements: Decode::decode(decoder)?,
            scale1: Decode::decode(decoder)?,
            scale2: Decode::decode(decoder)?,
            scale3: Decode::decode(decoder)?,
            ..Default::default()
        };

        // Pointers. The offsets come from the file, so each is bounds-checked against the
        // buffer it indexes before any pointer is formed: an unchecked `base.offset(...)` here
        // was an arbitrary-read/write primitive for anyone who could hand the process a file.
        if node.is_terminal() {
            // do nothing
        } else if node.is_chance() {
            let base = CHANCE_BASE_MUT.with(|c| c.get());
            if !base.is_null() {
                let lens = STORAGE_LENS.with(|c| c.get());
                let num_bytes = STORAGE_NUM_BYTES.with(|c| c.get());
                let offset = checked_offset(
                    isize::decode(decoder)?,
                    node.num_elements as usize,
                    num_bytes,
                    lens[3],
                    "chance storage",
                )?;
                node.storage1 = unsafe { base.add(offset) };
            }
        } else {
            let bases = PTR_BASE_MUT.with(|c| c.get());
            if !bases[0].is_null() {
                let lens = STORAGE_LENS.with(|c| c.get());
                let num_bytes = STORAGE_NUM_BYTES.with(|c| c.get());
                let offset = checked_offset(
                    isize::decode(decoder)?,
                    node.num_elements as usize,
                    num_bytes,
                    lens[0].min(lens[1]),
                    "node storage",
                )?;
                let offset_ip = checked_offset(
                    isize::decode(decoder)?,
                    node.num_elements_ip as usize,
                    num_bytes,
                    lens[2],
                    "IP storage",
                )?;
                node.storage1 = unsafe { bases[0].add(offset) };
                node.storage2 = unsafe { bases[1].add(offset) };
                node.storage3 = unsafe { bases[2].add(offset_ip) };
            }
        }

        Ok(node)
    }
}
