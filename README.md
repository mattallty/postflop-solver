# postflop-solver

An open-source postflop solver library written in Rust.

> [!NOTE]
> This is a **fork** of [b-inary/postflop-solver], whose author suspended development in
> October 2023 ([upstream issue #46]) to work on a commercial solver.
> This fork keeps the engine building and running on modern Rust toolchains and adds a few
> library and example features. See [Fork changes](#fork-changes) below.

[b-inary/postflop-solver]: https://github.com/b-inary/postflop-solver
[upstream issue #46]: https://github.com/b-inary/postflop-solver/issues/46

Upstream documentation (does not cover the additions made in this fork):
https://b-inary.github.io/postflop_solver/postflop_solver/

**Related repositories** (upstream, unmaintained)
- Web app (WASM Postflop): https://github.com/b-inary/wasm-postflop
- Desktop app (Desktop Postflop): https://github.com/b-inary/desktop-postflop

**Note:**
The original purpose of this library was to serve as a backend engine for the GUI applications
([WASM Postflop] and [Desktop Postflop]), so direct use by users/developers was not a design goal
and breaking changes were often made without version changes.
See [CHANGES.md](CHANGES.md) for the upstream list of breaking changes.

[WASM Postflop]: https://github.com/b-inary/wasm-postflop
[Desktop Postflop]: https://github.com/b-inary/desktop-postflop

## Usage

- `Cargo.toml`

```toml
[dependencies]
postflop-solver = { git = "https://github.com/mattallty/postflop-solver" }
```

The crate builds on current stable Rust (tested with 1.90.0); the optional `custom-alloc`
feature still requires nightly.

- Examples

You can find examples in the [examples](examples) directory:

| Example | Description |
| --- | --- |
| [`basic`](examples/basic.rs) | Build, solve, and query a game configured in Rust code. |
| [`from_config`](examples/from_config.rs) | Same, but driven by an external JSON file. |
| [`file_io`](examples/file_io.rs) | Save and load a solved game tree. |
| [`node_locking`](examples/node_locking.rs) | Lock strategies at specific nodes. |

If you have cloned this repository, you can run an example with the following command:

```sh
$ cargo run --release --example basic
```

You can also drive the solver from an external JSON configuration file instead of
hard-coding it, using the `from_config` example:

```sh
$ cargo run --release --example from_config -- examples/config.json
```

See [examples/config.json](examples/config.json) for the expected schema (ranges, board,
pot/stack, bet sizes, and solver stopping criteria).

## Fork changes

Relative to the last upstream commit:

- **Builds on modern Rust.** `bincode` now resolves to the released 2.0, whose `Decode` /
  `BorrowDecode` traits gained a `Context` generic parameter; the four hand-written bincode
  impls were migrated. A handful of raw-pointer field accesses were made explicit for
  rustc's `dangerous_implicit_autorefs` lint, and the remaining stable/nightly clippy and
  rustdoc warnings were fixed so CI passes with `--deny warnings`.
- **`PostFlopGame::visit`** — traverse the whole subtree below the current node, invoking a
  closure at each node with the game positioned there (so `strategy`, `expected_values`,
  `available_actions`, etc. are all available). The current node is restored afterwards.
  Useful for statistics, custom exports, and strategy pruning. (upstream issue #15)
- **`PostFlopGame::free_memory`** — release a solved game's storage buffers while keeping the
  tree structure and configuration, so multiple game instances can be kept around without
  holding all of their memory. `allocate_memory` can be called again without rebuilding.
  (upstream issue #29)
- **`from_config` example** — solve a game described by a JSON file instead of Rust code.
  (upstream issue #53)

The solver algorithm and its numerical behaviour are unchanged; the PioSOLVER-reference
accuracy tests and the bunching tests still pass.

## Implementation details

- **Algorithm**: The solver uses the state-of-the-art [Discounted CFR] algorithm.
  Currently, the value of γ is set to 3.0 instead of the 2.0 recommended in the original paper.
  Also, the solver resets the cumulative strategy when the number of iterations is a power of 4.
- **Performance**: The solver engine is highly optimized for performance with maintainable code.
  The engine supports multithreading by default, and it takes full advantage of unsafe Rust in hot spots.
  The original author reviewed the assembly output from the compiler and ensured that SIMD instructions are used as much as possible.
  Combined with the algorithm described above, the performance surpasses paid solvers such as PioSOLVER and GTO+.
- **Isomorphism**: The solver does not perform any abstraction.
  However, isomorphic chances (turn and river deals) are combined into one.
  For example, if the flop is monotone, the three non-dealt suits are isomorphic, allowing us to skip the calculation for two of the three suits.
- **Precision**: 32-bit floating-point numbers are used in most places.
  When calculating summations, temporary values use 64-bit floating-point numbers.
  There is also a compression option where each game node stores the values by 16-bit integers with a single 32-bit floating-point scaling factor.
- **Bunching effect**: At the time of writing, this is the only implementation that can handle the bunching effect.
  It supports up to four folded players (6-max game).
  The implementation correctly counts the number of card combinations and does not rely on heuristics such as manipulating the probability distribution of the deck.
  Note, however, that enabling the bunching effect increases the time complexity of the evaluation at the terminal nodes and slows down the computation significantly.

[Discounted CFR]: https://arxiv.org/abs/1809.04040

## Crate features

- `bincode`: Uses [bincode] crate (2.0) to serialize and deserialize the `PostFlopGame` struct.
  This feature is required to save and load the game tree.
  Enabled by default.
- `custom-alloc`: Uses custom memory allocator in solving process (only available in nightly Rust).
  It significantly reduces the number of calls of the default allocator, so it is recommended to use this feature when the default allocator is not so efficient.
  Note that this feature assumes that, at most, only one instance of `PostFlopGame` is available when solving in a program.
  Disabled by default.
- `rayon`: Uses [rayon] crate for parallelization.
  Enabled by default.
- `zstd`: Uses [zstd] crate to compress and decompress the game tree.
  This feature is required to save and load the game tree with compression.
  Disabled by default.

[bincode]: https://github.com/bincode-org/bincode
[rayon]: https://github.com/rayon-rs/rayon
[zstd]: https://github.com/gyscos/zstd-rs

## License

Copyright (C) 2022 Wataru Inariba

Modifications in this fork are Copyright (C) 2026 the fork contributors, released under the
same license.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
