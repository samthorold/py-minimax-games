# Choosing a fast core-engine language for the Star Realms bot

> Research notes for #127, kept as written (2026-10-21). Section 4 drops an earlier wrong constraint, that training had to run on the M1; its ranking supersedes the earlier one. The review of #127 chose Rust instead (docs/star-realms.md, Compiled engine). Labels: [V] verified, [A] abstract/README only, [U] unverified.


Research date: 2026-10-21. Context: the pure-Python Star Realms engine (frozen dataclasses, about 1,050 lines of rules, about 45k search nodes/s on CPython, PyPy 1.5-1.7x faster), developed on an M1 with 8 GB and no CUDA. The target is self-play learning (a value/policy net in a Python ML framework) plus search (ISMCTS, PIMC or whole-turn enumeration) that calls the net.

Evidence labels:
- **[V]**: verified from the primary source (docs, repo files, release page, PyPI) during this session.
- **[A]**: abstract or README-level only. The claim is the project's own and was not independently checked.
- **[U]**: unverified. From memory or a secondary source, or no source was found.

---

## 0. The most relevant prior art: a Star Realms engine in Numba (TimStewartJ/space-deck-builder)

This is the strongest single piece of evidence in this report. The project is the same game, made the same decision, and measured it.

- The repo is a Python + uv project that simulates and trains AI for a deckbuilder whose card data comes from the Star Realms Core Set spreadsheet. **[V]** (README; GitHub API: last push 2026-09-20, 1 star, so a one-person project). <https://github.com/TimStewartJ/space-deck-builder>
- PyTorch is the ML layer (CPU/CUDA/ROCm extras in `pyproject.toml`), and Numba is a dev-group dependency pinned `numba>=0.67,<0.68`. **[V]** (pyproject.toml)
- **Architecture** (ADR 0003, `docs/decisions/0003-compiler-backed-numba-vector-engine.md`) **[V]**:
  - Python `Card`/`Effect` trees are compiled into fixed-width opcode tables. A scalar Python interpreter and a Numba SPMD engine both consume those tables, and the Python object engine is kept as an oracle.
  - State is struct-of-arrays card counts, `zones[games, 2, 5, 49]` plus `effect_ready[games, 2, 49, max_bundles]`.
  - Differential tests compare every legal mask, the 505-float observation and every canonical state field after every decision.
- **Measured** (`results/vector_engine_gate.md`, i7-12700K) **[V]**:
  - Primitive engine workload: Numba SPMD 66.5M decisions/s on 20 threads (96M on 16 threads, 13.1M on 1 thread). NumPy masked SIMD managed 1.71M/s, eager torch CPU 1.15M/s and torch ROCm 0.91M/s. All backends produced the same BLAKE2b state digest.
  - Real Gate-2 rules subset: 7.31M decisions/s for engine + mask + observation, and 542k/s with the GPU policy in the loop. The README later reports 5.88M/s and 587k/s on the full rules.
  - Conclusion: rollout becomes **policy-bound**, not engine-bound.
- **How search calls the net** (ADR 0011) **[V]**:
  - The net stays in PyTorch, called from Python. Search is batched across many decisions, candidates and sampled worlds: "every rollout step makes one actor-only evaluation for all active rows".
  - Before batching, about 55 of the 80 ms per searched decision went to small net calls (fixed launch overhead of about 2.5 ms each).
  - Worlds are dealt with a counter-based SplitMix64 hash for determinism.
- **Hard parts they hit** (ADRs 0003 and 0005) **[V]**:
  - Choice/target/pending effects needed "continuation/pending-state IR". Gate 2 supported only 26 of 49 cards at first, and the rest followed later.
  - Stealth Needle copies, Blob Carrier top-decking and Mech World wildcard allies each needed extra state-of-arrays fields.
  - They wrote: "the highest-risk failure mode in this project: silently training a strong policy for subtly wrong rules". This is why they kept a single rule IR and differential tests.

**Takeaways for us:**
- (a) Numba is proven sufficient for exactly this game, at about 100x our current speed per core.
- (b) Requirement 4 ("net inside compiled search") was not needed. Batched calls back into Python/PyTorch were the design.
- (c) The real cost was rules fidelity under a struct-of-arrays rewrite, mitigated by an oracle + differential testing.

---

## 1. Checking the claims made from memory

### a. RL frameworks consume environments through Python APIs, with fast envs in C/C++/Rust/JAX/Numba behind them

**Broadly confirmed.**
- Gymnasium's vector API returns NumPy arrays of shape `(num_envs,)` for rewards, terminations and truncations. It ships `SyncVectorEnv`/`AsyncVectorEnv` and mentions "several custom vector environment implementations". **[V]** <https://gymnasium.farama.org/api/vector/>
- PettingZoo's AEC API is the turn-based multi-agent API. Its classic envs put action masks in the observation dict. No vectorised API is mentioned. **[V]** <https://pettingzoo.farama.org/api/aec/>
- Examples of fast envs behind those APIs:
  - EnvPool (C++, "Easy C++ developer API to add new envs") **[V]**
  - PufferLib (C/CUDA) **[V]**
  - Pgx (JAX) **[V]**
  - OpenSpiel (C++ with Python bindings) **[V]**
  - space-deck-builder (Numba) **[V]**
- Not re-fetched: SB3/CleanRL consuming Gymnasium envs **[U]**, well known but not fetched this session.
- Nuance: PufferLib 5.0 has moved away from "Python API over an env". Its trainer is native CUDA C (below).

### b. Rust + PyO3

- **rust-numpy is mature, with zero-copy both ways.** **[V]**
  - The README pins `numpy = "0.30"` with `pyo3 = "0.30"` and requires Python >= 3.9 and MSRV 1.85.
  - `PyReadonlyArray::as_array()` gives an ndarray view of NumPy memory, with runtime borrow checking (`BorrowError`).
  - `PyArray::from_owned_array` "uses the internal `Vec` of the `ndarray::Array` as the base object of the NumPy array", so it transfers ownership with no copy. `from_array` copies.
  - Sources: <https://github.com/PyO3/rust-numpy>, <https://docs.rs/numpy/latest/numpy/array/struct.PyArray.html>
- **maturin works with uv. Confirmed.** **[V]**
  - `uv init --build-backend maturin` creates a project with `Cargo.toml` and `lib.rs`.
  - uv sets `tool.uv.cache-keys` to cover common source types. `--reinstall` forces a rebuild otherwise.
  - maturin's docs recommend `uv publish`.
  - Sources: <https://docs.astral.sh/uv/concepts/projects/init/>, <https://www.maturin.rs/>
- **"ML inference in Rust is less mature": partly true, and less relevant than assumed.**
  - **ort (ONNX Runtime) is `2.0.0-rc.14`.** It is still a release candidate, but the docs say "production-ready and we recommend new & existing projects use it". A CoreML execution provider exists (`coreml` feature). **[V]** <https://ort.pyke.io/>, <https://ort.pyke.io/perf/execution-providers>. Whether macOS arm64 prebuilt binaries include CoreML is not confirmed **[U]**.
  - **candle** (Hugging Face) has an "optimized CPU backend with optional MKL support for x86 and Accelerate for macs" and loads safetensors/npz/PyTorch files. **[A]** <https://github.com/huggingface/candle>. Its Metal support is implied by a `candle-metal-kernels` crate but not described on the page **[U]**.
  - **burn**: LibTorch "deprecated in 0.22.0 and has been removed from main". It has CubeCL GPU backends including Metal, a pure-Rust CPU backend called "Flex", and ONNX import via `burn-onnx` ("limited set of operators"). **[A]** <https://github.com/tracel-ai/burn>
  - **tch-rs** requires libtorch v2.13.0. It can link to the active Python's torch with `LIBTORCH_USE_PYTORCH`, and M1 setup is pointed at an issue (#488) rather than documented. **[V]** <https://github.com/LaurentMazare/tch-rs>
  - **Practical point (analysis, not sourced):** a Star Realms value/policy net is likely a small MLP; the space-deck-builder policy is 0.28M parameters **[V]**. A hand-written forward pass (a few GEMVs over weights exported to `.npy`/safetensors) is roughly 50-100 lines in any compiled language, deterministic, and dependency-free. On Apple Silicon, CPU BLAS via Accelerate is the realistic path for small batches. That makes requirement 4 achievable in Rust, C++, Numba or Cython without any ML framework.

### c. Numba

- **space-deck-builder uses Numba with PyTorch: confirmed, and for this very game.** See section 0. **[V]**
- **Python 3.13/3.14 support: confirmed.** **[V]**
  - 3.14 support arrived in 0.63.0 (8 Dec 2025), with early free-threading support.
  - 0.65.0 (31 Mar 2026) covers 3.14t. It notes that Intel SVML is unsupported, which costs performance where SVML was used before (irrelevant on M1).
  - 0.67.0 (11 Aug 2026) is current, with NumPy 2.5 support.
  - Sources: <https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html>, <https://numba.readthedocs.io/en/latest/release/0.65.0-notes.html>
- **Language constraints** <https://numba.readthedocs.io/en/stable/reference/pysupported.html>:
  - `enum.Enum` and `IntEnum` subclasses are supported **[V]**. Whether superenums work is not checked and is unlikely **[U]**.
  - User-defined classes are unsupported "except for @jitclass" **[V]**.
  - Dataclasses are not mentioned, so effectively unsupported **[V for absence]**.
  - `typed.List` and `typed.Dict` are "experimental" **[V]**.
  - Tuples and namedtuples are supported, but a namedtuple class can't be created inside jit code **[V]**.
  - `try/except` is partial **[V]**.
  - Recursion is mostly supported **[V]**.
  - Generators are supported, without `send`, `throw` or `close` **[V]**.
- **jitclass:**
  - It lives in `numba.experimental` and the docs say "This is a early version of jitclass support"; it is CPU-only **[V]** <https://numba.readthedocs.io/en/stable/user/jitclass.html>.
  - jitclass is not cacheable, and functions taking jitclass arguments recompile every run. StructRef is the usual workaround but is poorly documented. **[A]**: forum posts only, possibly dated. <https://numba.discourse.group/t/cache-jitted-function-with-jitclass-argument/780>, <https://numba.discourse.group/t/how-to-build-complex-data-structures-with-working-jit-cache/2824>
- **Caching (`cache=True`)** **[V]** <https://numba.readthedocs.io/en/stable/developer/caching.html>:
  - The cache is invalidated when the source file changes, but "Cache invalidation fails to recognize changes in symbols defined in a different file".
  - Globals are frozen as constants.
  - This means stale-cache bugs are possible in a multi-module engine. A pre-commit/CI step should clear the cache or run tests with `NUMBA_DISABLE_JIT=1` as well as without.
- **Debugging** **[U]**, not fetched: `NUMBA_DISABLE_JIT=1` runs the same code as plain Python, which is good for debuggers and coverage. Compiled-mode errors are typing errors that are often hard to read.
- **Type checkers:** no source found on whether Numba ships `py.typed`. Third-party stubs (e.g. one in a pandas fork) treat `jit` as identity, with a catch-all `Any`. **[U]**. Expect `@njit` functions to be typed for ty only through their Python annotations, and nothing checks that Numba's inferred types agree with those annotations.
- **Struct-of-arrays style is forced.** Frozen dataclass states would be rewritten as integer arrays. That is a large rewrite of 1,050 lines of rules, but it also produces the observation tensors directly (requirement 2).

### d. PufferLib

- "Ocean environments are written in C. Mostly very simple C." You copy the `minimal` template and PufferLib looks for `ocean/your_env/your_env.h`. `--debug` enables the address sanitizer. Observation and action metadata go in `binding.c`. **[V]** <https://puffer.ai/docs.html>
- **Throughput:** "Up to 60,000,000 step/second training in only ~10k lines of CUDA C" (PuffeRL 5.0). **[A]**: the vendor's claim, conditions unstated.
- **Important for us:**
  - "As of 5.0, we have a solid --cpu eval mode but **no CPU training option**."
  - "The PufferLib 5.0 native backend is ~10,000 lines of CUDA C."
  - The `minimal` template folder now contains `minimal.cu`, `minimal.h` and `minimal_net.h`.
  - macOS isn't mentioned.
  - All **[V]**. Training on an M1 with no CUDA is therefore **not supported** by current PufferLib. Older 2.x/3.x releases had CPU paths and Python bindings **[U]**.
- **Multi-agent:** "60+ C and CUDA environments … massively multiagent sims" **[V]**. The Ocean folder includes connect4, chess, go, checkers, backgammon, hex and tripletriad (a card game) **[V for folder names; their two-player/self-play handling was not inspected]**. The docs don't describe self-play or opponent policies **[V for absence]**.
- **Verdict:** a useful design reference (C structs + flat buffers), but not a fit for an M1-only workflow.

### e. OpenSpiel

- **Algorithms** (<https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md>) **[V]**:
  - IS-MCTS ("lightly tested")
  - MCTS (thoroughly tested)
  - AlphaZero C++/LibTorch (lightly tested) and Python/TF (thoroughly tested)
  - CFR family, external/outcome-sampling MCCFR, Deep CFR and NFSP (thoroughly tested)
  - PIMC (lightly tested)
  - R-NaD is not in that table. A changelog lists Regularized Nash Dynamics as added **[A]** <https://data.safetycli.com/changelogs/open-spiel>.
  - The algorithms doc gives language per entry only for AlphaZero. In general, "Algorithms and tools are written both in C++ and Python" **[V]**.
- **Adding games** **[V]** <https://github.com/google-deepmind/open_spiel/blob/master/docs/developer_guide.md>:
  - The developer guide expects in-tree games: register in `open_spiel/games/CMakeLists.txt`.
  - Python games are also supported, but must be registered in `open_spiel/python/CMakeLists.txt` and `python/games/__init__.py`.
  - Out-of-tree games are not mentioned **[V for absence]**.
  - Python games can be registered at runtime with `pyspiel.register_game` **[U]**, from memory and not verified. They are slow because every algorithm call crosses into Python **[U]**.
- **Maintenance and packaging** **[V]** <https://pypi.org/project/open-spiel/>:
  - Very active: 2.0 on 14 Jul 2026, 2.0.2 on 12 Aug 2026.
  - Wheels for cp311-cp314 include **macOS 11.0+ arm64**. This supersedes the install docs, which still say "only supported on x86_64".
- **Verdict:**
  - Worth it as a reference and baseline harness: IS-MCTS, PIMC, CFR and exploitability tools.
  - Writing the Star Realms engine as an OpenSpiel C++ game means a fork or vendored build, C++ for the rules, and its observation/legal-actions API.
  - Batched numpy stepping isn't OpenSpiel's model. It steps state objects one at a time **[U]**, consistent with the API but not checked.

### f. JAX / Pgx

- Pgx: "10-100x faster than existing implementations available in Python" on a DGX-A100. Its games are backgammon, chess, shogi, Go and miniatures, plus Kuhn poker in the repo. **[A]** <https://arxiv.org/abs/2303.17503>
- **Branchy rules under vmap: confirmed costly.** The JAX docs say that under `vmap` with a batched predicate "cond is converted to select()", so both branches execute for every element. **[V]** <https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html>
  - Star Realms has dozens of card effects, pending-choice modes and copy effects. A `lax.switch` over about 50 card opcodes would run every branch on every step. Fixed shapes also force padded hands, decks and trade rows.
  - space-deck-builder's measurement, eager torch at 1.1M/s vs Numba at 66M/s for the same SoA workload, points the same way for array-framework engines **[V]**. That is not a JAX-jit measurement.
- **Apple Silicon** **[A/U]**: `jax-metal` 0.1.1 is the last release and is labelled experimental on PyPI. A forum user reports that JAX maintainers closed Metal issues in Dec 2025, saying there was no active development. Third-party `jax-mps` (MLX-based) and `metaljax` (beta) exist. On M1 you should assume **CPU-only JAX**, which loses Pgx's main advantage.
- **Verdict:** a poor fit.

---

## 2. The wider set of options, assessed against the requirements

Key: R1 batched numpy API, R2 in-engine features, R3 zero-copy, R4 net inside search, R5 determinism, R6 rules correctness/safety, R7 uv/pre-commit/CI fit, R8 iteration speed and debuggability.

| Option | Assessment |
|---|---|
| **Numba** (SoA kernels in Python syntax) | R1-R3 are native: numpy in and out, zero copy. R4: hand-written MLP in `@njit` (np.dot works in nopython mode **[U]**), or batch out to PyTorch as space-deck-builder does **[V]**. R5: seed explicitly (the counter-based hash in ADR 0011 **[V]**); parallel `prange` needs per-game RNG streams. R6 is the weak spot: no dataclasses or rich types, everything becomes int arrays and opcodes, and ty can't see Numba's typing. Mitigate with a reference engine + differential tests. R7 is excellent: a pure PyPI dependency, no toolchain, 3.14 wheels **[V]**. R8: `NUMBA_DISABLE_JIT=1` for debugging **[U]**, but compile warm-up and stale caches across files **[V]**. **Proven on this game [V].** |
| **Rust + PyO3 + maturin** | R1-R3: strong (rust-numpy zero copy **[V]**; GIL release possible **[U]**). R4: hand-rolled MLP, or ort/candle/burn **[A]**, or batch out to Python. R5: strong (explicit RNG crates, no hidden globals). **R6 is the best of all options**: enums with exhaustive `match`, ownership and immutability, and you can keep a typed, readable state model rather than SoA. R7: maturin + uv is supported **[V]**. Pre-commit/CI needs `cargo fmt`, `clippy` and `cargo test` hooks plus a Rust toolchain in CI, and uv rebuild-on-change needs `cache-keys` **[V]**. R8: compile times are slower than Numba's, but tests are good. Edits now span two languages. |
| **C++ + pybind11 / nanobind** | R1-R3 good. nanobind claims about 4x faster compiles and 5x smaller binaries than pybind11, and can target the Stable ABI from 3.12 **[A]** <https://github.com/wjakob/nanobind>. R4: LibTorch, ONNX Runtime C++ or MLX C++ **[V: MLX has a C++ API]**. R6: weaker than Rust (UB, no exhaustive matching by default; sanitizers help). R7: scikit-build-core + CMake works with uv **[U for specifics]**. Best if you adopt OpenSpiel. Prior art: sts_lightspeed (Slay the Spire, C++17, "1M random playouts in 5s with 16 threads", with a pybind11 submodule) **[A]** <https://github.com/gamerpuppy/sts_lightspeed>. |
| **Cython** | Cython 3.3.0 (22 Aug 2026) has cp314 and macOS arm64 wheels **[V]** <https://pypi.org/project/Cython/>. Typed memoryviews give zero-copy numpy **[U]**. A C-level speedup is possible but needs disciplined `cdef` typing everywhere. Rules safety is about C's level, and tooling for type checking and debugging is weaker. A middle ground with no advantage over Numba or Rust here. |
| **mypyc** | It would compile the existing typed Python with dataclasses mostly unchanged. Typical speedups are a few times, not 100x **[U]**. Cheap to try as a baseline, but it won't deliver batched SoA throughput. |
| **PyPy** | Already measured at 1.5-1.7x. It's a dead end for the ML stack (no PyTorch on PyPy **[U]**). |
| **JAX / Pgx-style** | See 1f. Branch divergence → select **[V]**, fixed shapes, no Metal on M1 **[A]**. Poor fit. |
| **PyTorch-native engine** | Eager torch reached about 1.1M/s on CPU vs 66M/s for Numba on the same SoA workload **[V, space-deck-builder]**. Poor fit. |
| **EnvPool C++ framework** | macOS wheels for 3.12-3.14 and multi-player support **[A]**. Its API targets Gym-style envs with a thread pool, not search or cloning **[U]**. Not a fit for search. |
| **PufferLib** | See 1d. No CPU training in 5.0 **[V]**. Not a fit on M1. |
| **OpenSpiel** | See 1e. Good algorithms, macOS arm64 wheels **[V]**. Hosting our game means C++ in-tree or a slow Python game. Good as a harness for comparing bots, poor as an engine home. |
| **Julia (juliacall/PythonCall; AlphaZero.jl, ReinforcementLearning.jl)** | PythonCall advertises fast non-copying array conversion **[A]**. Startup and first-call compile latency are real **[A, forum]**. AlphaZero.jl claims "one to two orders of magnitude faster than … pure Python" **[A]**. ReinforcementLearning.jl's README has "[TODO:]" sections and an unclear status **[V]**. Adds a second runtime and package manager outside uv. Poor fit for R7. |
| **Mojo** | Mojo 1.0 (11 Aug 2026), open-sourced 18 Aug 2026 **[A, secondary]**. Calling Mojo from Python is "in early development", described as beta, with limitations (no kwargs, no keyword-only args, few types convertible from Python), and NumPy isn't mentioned **[V]** <https://mojolang.org/docs/manual/python/mojo-from-python/>. Too immature for this project. |
| **Codon** | Has `@codon.jit` with zero-copy numpy pointer passing, Apache-2.0 **[A]**. No 2026 status found. A separate compiler outside uv. Risky. |
| **Taichi** | Status unknown: PyPI and release data could not be retrieved **[U]**. A GPU/physics-oriented DSL. Not a fit. |
| **Zig / Go (cgo)** | No numpy binding ecosystem comparable to PyO3, and cgo call overhead **[U]**. Not recommended. |
| **RLCard / LightZero** | RLCard is pure Python, with poker/UNO/Dou Dizhu **[V]**. LightZero (v0.2.0, Mar 2026) has C++/Cython MCTS (`ctree`) but only perfect-information board games among its examples **[A]**. Both are references, not engine hosts. |
| **Ludii** | A general game system **[V]**. Java and RL API not confirmed **[U]**. Not a fit. |

### Published performance comparisons

- No benchmark compares Rust, Numba and C++ on game simulators **[V for absence in the search]**.
- Indirect evidence:
  - Rust is roughly on par with C++ on general routines **[A]** <https://arxiv.org/pdf/2209.09127>.
  - A CERN report found C++ between 0.9x and 1.6x the speed of Numba on simulation kernels **[A]**.
  - The best direct evidence is space-deck-builder's own Numba vs NumPy vs torch gate on Star Realms-shaped work **[V]**.
- What other card-game AI projects chose:
  - Slay the Spire: C++ (sts_lightspeed) **[A]**.
  - Hearthstone: SabberStone in C#, a one-person project with an ageing README **[V]**.
  - Star Realms: Numba (space-deck-builder) **[V]**.
  - RLCard: Python **[V]**.

---

## 3. Ranked shortlist

### 1. Numba struct-of-arrays engine, keeping the current Python engine as the oracle
- **Pros:**
  - Proven on this exact game at about 7M decisions/s with masks and observations **[V]**.
  - Stays in one language and fits uv/ty/pytest/pre-commit with no new toolchain.
  - Numpy-native for R1-R3. The net stays in PyTorch, called in batches (the ADR 0011 pattern).
- **Cons:**
  - Rules become opcode/array code with weak typing, and ty can't check jit semantics.
  - jitclass is experimental and uncacheable; cache invalidation misses changes in other files.
  - Compile warm-up slows the test loop. Superenum `match` + `assert_never` exhaustiveness is lost inside kernels.
  - Correctness rests entirely on differential tests against the Python engine.

### 2. Rust core with PyO3/maturin (typed state model, batch API, features emitted into numpy buffers)
- **Pros:**
  - Best correctness story for intricate rules (exhaustive enums, immutability by default), mirroring the current dataclasses + superenums + `assert_never` discipline.
  - Zero-copy numpy **[V]** and first-class uv support **[V]**.
  - Search can live entirely in Rust, with a hand-written MLP, or ort/candle when needed.
- **Cons:**
  - A second language and toolchain in CI and pre-commit, slower edit/compile cycles, and rebuild-on-edit friction under uv.
  - Rust ML inference crates are in flux (ort is an RC, burn dropped LibTorch) **[V/A]**.
  - Nobody has published this stack for a deckbuilder **[V for absence]**.

### 3. C++ with nanobind, optionally as an OpenSpiel game
- **Pros:**
  - Access to OpenSpiel's C++ IS-MCTS, PIMC and CFR and LibTorch AlphaZero.
  - MLX's C++ API for Apple-native inference.
  - Precedent in sts_lightspeed.
- **Cons:**
  - Weakest safety for intricate rules.
  - OpenSpiel games live in-tree, under CMake.
  - CMake/scikit-build in a uv repo.

### 4. mypyc (baseline only)
- Near-zero rewrite cost. Worth an afternoon to learn how much "compile what we have" buys, but it is unlikely to reach the throughput that self-play needs.

**Ruled out for this setup:** JAX/Pgx (branch divergence, no Metal), PufferLib 5 (no CPU training), Mojo (beta interop), Julia (second runtime), Codon/Taichi (uncertain status).

### What the spike should measure

Port the same slice to Numba and to Rust: the core turn loop, plus one pending-choice card (e.g. a scrap-from-hand/discard choice), one ally card and Stealth Needle's copy effect.

1. **Throughput** on the M1, single core and all cores: random-playout decisions/s for engine + legal mask + observation, at batch sizes of 1 (search) and 1k-16k (self-play).
2. **Search path:** ISMCTS/PIMC nodes/s with leaf values from a 0.3M-parameter MLP in three ways: (a) a hand-written forward pass in-engine, (b) batched calls back to PyTorch CPU/MPS, (c) ort or candle (Rust only). Measure the latency of a batch of about 1-256 rows, since ADR 0011 found per-call overhead dominated.
3. **Correctness effort:** time to implement the copy card and the pending choice, and the number of bugs the differential test against the existing Python engine caught.
4. **Developer loop:** cold and warm `pytest` time (Numba compile, `cargo build` via `uv sync`), quality of the debugging story, and the ty/clippy coverage of the rules code.
5. **Determinism:** bit-identical replay of seeded batched games across thread counts.
6. **Tooling cost:** lines of pre-commit/CI configuration added, and CI time on GitHub macOS and Linux runners.

**Decision rule (suggested):** if Numba reaches more than 2M decisions/s/core with masks and observations, and the copy/choice cards stay readable under differential tests, choose Numba. Choose Rust if Numba's rules code becomes opaque or the warm-up and caching slow the test loop noticeably, as long as Rust's throughput is similar or better.

### Where the evidence is thin
- No head-to-head Rust vs Numba measurement exists for card-game engines.
- The space-deck-builder numbers come from an x86 16-core desktop, not an M1.
- The jitclass caching limitation comes from forum posts.
- The behaviour of OpenSpiel Python games and their out-of-tree registration is from memory.
- Whether the Ocean board games in PufferLib support self-play was not inspected.

---

## 4. Without the M1 training constraint

Revised constraint: training may run anywhere, including rented CUDA GPUs and many-core cloud CPUs. **Inference and play must run on the M1 (8 GB)**, ideally with search, and development and tests should be workable there.

### 4.1 PufferLib 5.0 with a CUDA cloud trainer

I read the source of the `5.0` branch through the GitHub API, so most of this subsection is **[V]**. Files: `src/pufferenv.h`, `src/pufferl.cu`, `src/puffercpu.c`, `build.sh`, `ocean/connect4/connect4.h`, `ocean/chess/chess.h`, `ocean/tripletriad/tripletriad.h`. <https://github.com/PufferAI/PufferLib/tree/5.0>

- **Env API.** A C env implements `puf_init`, `puf_reset`, `puf_step`, `puf_render`, `puf_close` and `puf_log`. Each agent has buffers for observations, actions, rewards and terminals, plus `unsigned char* action_mask` and an `int policy` slot. **[V]** (`src/pufferenv.h`)
- **Action masks.** The trainer allocates an `(total_agents, mask_size)` mask "always" and applies it to the logits when sampling. **[V]** (`pufferl.cu`, around lines 450 and 600-672)
- **Self-play.**
  - `policies[0]` is the policy being trained, and `policies[i>0]` are "frozen opponents (selfplay/match)", which may use a different architecture. **[V]**
  - A `Selfplay` struct keeps a pool of historical checkpoints: `SELFPLAY_MAX_HIST 8`, `SELFPLAY_MAX_LADDER 16`. **[V]**
  - The chess env has `CHESS_MODE_SELFPLAY` and multi-bank historical opponents. **[V]**
  - **Constraint:** "GPU env backend does not support selfplay or multi-policy (match)". So a self-play env must be a **CPU C env**, stepped on the host, with the CUDA trainer doing the learning. **[V]** (`pufferl.cu` around line 1847)
- **Turn-based two-player games** are written as one learning agent per seat, with the opponent's move made inside `puf_step`.
  - connect4 and tripletriad use a scripted bot opponent inside the step (`compute_env_move` negamax; `get_bot_card_selection`). **[V]**
  - Chess interleaves a learner move and an opponent ply, and supports self-play. **[V]**
  - There is no deckbuilder or imperfect-information card game other than tripletriad.
- **Throughput.** "Up to 60M steps/s" **[A]**. That figure is for simple GPU-native envs. Self-play needs the CPU backend, so a Star Realms env would be limited by the host CPU's C stepping, plus host-to-device copies (`cudaMemcpyAsync` of the mask and observations, **[V]**).
- **Running on the M1.**
  - `build.sh` has a non-Linux branch that uses `raylib-5.5_macos`, Homebrew `libomp` and a `box2d-macos-arm64` binary. **[V]** So standalone C envs and `--cpu` eval builds are intended to compile on Apple Silicon. I did not build one **[U]**.
  - The address sanitizer is **disabled on macOS** (`SANITIZE_FLAGS=()`), so `--debug` sanitizer runs need Linux. **[V]**
  - **Policy export:** `src/puffercpu.c` is a pure-C CPU forward pass. It reads "flat fp32 tensors" from a `*_weights.bin` file and implements linear, ReLU, conv2d, embedding, MinGRU and a "Linear encoder → N x MinGRU → Linear decoder (fused value)" net. **[V]** So a trained policy runs on the M1 in C without any ML framework, which also suits search written in C.
- **Costs and risks.**
  - The rules must be written in C, the least safe option for intricate rules. The sanitizer is unavailable on the Mac.
  - The framework moves fast: 5.0 is ~10k lines of CUDA C with no CPU training **[V]**.
  - The trainer is PPO-centric (PuffeRL). Search-based training such as expert iteration or AlphaZero targets is not provided **[U: I found no search component in `src/`]**.
  - A deckbuilder's action space (play card i, buy row slot j, scrap choices, pending modes) must be flattened into a fixed discrete space with masks. That is the same work any option needs.

### 4.2 JAX on GPU for rule-heavy games

- **Evidence that complex rules can be written in JAX:**
  - **Craftax** (Crafter + NetHack-like mechanics): "up to 250x faster than the Python-native original", and 1B PPO steps "in under an hour using only a single GPU". **[A]** <https://arxiv.org/abs/2402.16801>. Its README warns that compilation takes about 30 s to render the first frame and about 20 s more for the first action. **[V]** <https://github.com/MichaelTMatthews/Craftax>
  - **Mahjax** (Riichi Mahjong, a rule-heavy imperfect-information tile game): 2M steps/s (1M with red fives) on **eight A100s**. **[A]** <https://arxiv.org/abs/2605.20577>
  - **Pgx** has chess, shogi, Go, backgammon, Kuhn and Leduc poker, bridge bidding and Sparrow Mahjong. **[V]** <https://github.com/sotetsuk/pgx>
    - Its paper gives per-game speed only in figures. In the text, the slowest env reaches "about 10^5 samples/second" on one A100 at batch 1024. **[V]** <https://arxiv.org/html/2303.17503>
    - It notes that the API is not well suited to games with chance players "such as poker". **[V]**
    - The repo notes that "Go and chess do not perform well on TPUs". **[V]**
  - **JaxMARL** includes Hanabi, a cooperative partially observable card game. **[V]** <https://github.com/FLAIROx/JaxMARL>. Its speed figures are not on the README **[V for absence]**.
  - **Generals.io** superhuman agent: JAX-native simulator ">40 million frames per second on one GPU", and "seven days on four GPUs" to superhuman. **[A]** <https://arxiv.org/abs/2606.23348>
  - **Card or deckbuilding games:** I found no JAX deckbuilder or trading card game. The nearest is a JAX gin-rummy engine on Hugging Face, of unknown quality **[A]**.
- **Handling divergence and variable-length piles.**
  - Under `vmap`, `cond` becomes `select` and both branches run **[V]** (JAX docs, section 1f). The cost is the sum of all branches executed for every game.
  - For Star Realms this means every card-effect opcode and pending mode on every step, with decks, hands and discard piles as fixed-size count vectors (49 card types × zones, as space-deck-builder already does).
  - Count-vector state removes most variable-length problems. The ordered top-deck segment, copy effects and pending choices need padded fixed-size buffers.
  - No source quantifies select overhead for a card game **[V for absence]**. Pgx's slowest env at about 10^5 samples/s per A100 versus space-deck-builder's 5.9-7.3M decisions/s on a 16-thread desktop CPU (Numba) is a loose signal. These are different games and the comparison is not controlled.
- **Search in JAX.** `mctx` is a JAX-native batched MCTS (AlphaZero, MuZero, Gumbel MuZero) that is fully jit-compilable, with dynamics supplied as a `recurrent_fn`. **[V]** <https://github.com/google-deepmind/mctx>
  - Pgx's AlphaZero example uses mctx and Gumbel AlphaZero on 9x9 Go **[V]**.
  - mctx searches a single known state, so ISMCTS or PIMC over hidden cards would need sampled worlds as extra batch rows **[U, analysis]**.
- **Play on the M1.**
  - jaxlib 0.12.0 (19 Oct 2026) ships macOS arm64 wheels for cp312-cp315 **[V]** <https://pypi.org/project/jaxlib/>. The same JAX engine and net run on the M1 CPU.
  - Metal acceleration is not reliably available **[A]**, see section 1f.
  - The only M1 JAX CPU benchmark found is from 2021: "for any MatMul-heavy work, Jax is likely going to be slower than NumPy" with Accelerate **[A, dated]** <https://github.com/tlkh/m1-cpu-benchmarks>.
  - For play at batch 1-256 inside a search, the vmapped engine runs every branch for every row and per-call dispatch overhead applies **[U]**. Both hurt single-game search latency.
  - `jax.export` serialises StableHLO. It can target other platforms (`platforms=`), but running the artifact needs a JAX runtime within a compatibility window ("up to 6 months newer"). **[V]** <https://docs.jax.dev/en/latest/export/export.html>. This export path is for deployment, not a framework-free one.
- **Verdict.** Viable in principle. Craftax, Mahjax and Generals.io show rule-rich games train fast on GPUs. But:
  - (a) no deckbuilder exists to copy;
  - (b) branch divergence is structurally worst for card-effect dispatch;
  - (c) M1 play is CPU JAX;
  - (d) development means jit compile times of tens of seconds **[V, Craftax]**, and debugging inside `jit` is hard **[U]**.
  - It suits PPO-style training at huge batch. Its fit for single-game search on the M1 is weaker.

### 4.3 Numba, Rust and C++ with cloud training

- **The architecture all three share:** the engine runs on many CPU cores, and a GPU learner and inference server batch the net calls. This is exactly what space-deck-builder does (Numba engine on CPU, PyTorch on CUDA/ROCm, batched calls) **[V]**.
  - It is the most proven shape for this game.
  - Engine speed only needs to exceed the policy's throughput: theirs was about 13x over **[V]**.
- **Cloud CPUs.** Modal charges $0.0000131 per physical core-second, about $0.047 per core-hour, with one physical core = 2 vCPU **[V]** <https://modal.com/pricing>. A 32-core engine fleet costs about $1.50/hour. GPU pods come with a provider-defined vCPU count **[U]**.
- **Numba.** Unchanged strengths. It now also benefits from cloud many-core boxes (the `prange` SPMD scaling in section 0) **[V]**. M1 play stays in the same Python process.
- **Rust.** It can also own the search: a batched ISMCTS in Rust that hands leaf batches to Python. On the M1 the net runs in-process in Rust via a hand-written MLP, ort (CoreML or CPU EP) or candle (Accelerate) **[A/V]**.
- **C++.** Same shape. It can reuse OpenSpiel's C++ search, or PufferLib's `puffercpu.c` approach for C inference **[V]**.

**Inference options on the M1 (8 GB) for a small policy/value net:**

| Option | Status / evidence | Notes |
|---|---|---|
| Hand-written forward pass (NumPy + Accelerate, or in-engine) | PufferLib does this in C for deployment **[V]**. NumPy matmul on M1 reaches about 1.9 TFLOPS via Accelerate **[A, dated]** | Deterministic, no dependencies, lowest latency for batch 1-256. Recommended default. |
| PyTorch CPU / MPS | MPS per-call overhead not measured **[U]**. space-deck-builder found a fixed ~2.5 ms launch/sync per GPU call on ROCm, which dominated small batches **[V]** | Expect CPU to beat MPS for small batches, by analogy **[U]**. |
| ONNX Runtime (Python or `ort` in Rust) with CoreML EP | Official macOS wheels include the CoreML EP. Compilation can take "even minutes" without a cache, and dynamic shapes may hurt performance **[V]** <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html> | Portable export from any trainer. The CPU EP is probably best for tiny MLPs **[U]**. |
| MLX (Python or C++) | Has a C++ API **[V]** | Apple-native, but needs a weight-conversion step **[U]**. |
| JAX CPU | arm64 wheels available **[V]** | Only worthwhile if the engine is also JAX. |

### 4.4 Whole-stack frameworks

- **OpenSpiel.**
  - Its C++ AlphaZero (LibTorch) "uses threads, a shared cache, supports batched inference" and "can train significantly faster" than the Python TF1 version. Both are "re-implementations for illustration". **[V]** <https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md>
  - The algorithms table lists the C++ AlphaZero as lightly tested **[V]**.
  - AlphaZero assumes perfect information. Applying it to Star Realms means determinization, which is the PIMC weakness **[U, analysis]**.
  - R-NaD is designed for imperfect-information games, but its OpenSpiel status is only **[A]**.
  - Plugging in our game means writing it as an in-tree C++ OpenSpiel game **[V]**.
- **LightZero.** A custom env is a DI-engine `BaseEnv` whose `reset`/`step` return `{'observation', 'action_mask', 'to_play'}`, with `self_play_mode` and `play_with_bot_mode` for board games **[V]** <https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/envs/customize_envs.md>.
  - The env is Python. A Numba or Rust engine could sit behind it.
  - Its search assumes MuZero/AlphaZero-style perfect-information trees **[A]**.
- **Sample Factory.** "Single- & multi-agent training, self-play, supports training multiple policies at once" and "custom environments are first-class citizens" **[A]** <https://www.samplefactory.dev/>. The latest dates on the page are from 2023, so maintenance looks slow **[V]**. PPO only, with no search.
- **TorchRL 0.13.** Has action-masking transforms and multi-agent TensorDict support. MCTS and self-play are not mentioned on the README **[A]** <https://github.com/pytorch/rl>.
- **EnvPool.** C++ env framework with multi-player support **[A]**. Throughput-oriented Gym stepping, with no search or cloning **[U]**.
- **PufferLib.** See 4.1. It has the strongest built-in self-play and masks of the PPO frameworks, but its envs are in C.
- **Verdict.** No framework provides imperfect-information search plus self-play for a custom deckbuilder out of the box.
  - The best-supported pattern remains a custom engine + PyTorch PPO/expert-iteration with batched inference. This is space-deck-builder's path: PPO, then turn-bounded search, then expert iteration **[V]**.
  - OpenSpiel is still useful as a bot-evaluation harness.

### 4.5 Rough cloud cost

GPU prices (on-demand, checked today):
- **RunPod** **[V]** <https://www.runpod.io/pricing>:
  - RTX 4090: $0.34/h community, $0.89/h secure
  - RTX 5090: $0.69 / $1.19
  - L4: $0.44 / $0.59
  - A100 80GB SXM: $1.39 / $1.79
  - H100 SXM: $2.69 / $3.99
- **Lambda** **[V]** <https://lambda.ai/pricing>: A10 $1.29, A100 $1.99-2.79, GH200 $2.29, H100 $3.29-4.29, B200 $6.69-6.99 per GPU-hour.
- **Modal** **[V]** (per-second billing, derived hourly): T4 about $0.59, L4 about $0.80, A100-80GB about $2.50, H100 about $3.95. CPU about $0.047 per core-hour. $30/month of free credit on the Starter plan.
- **Colab and AWS** not checked **[U]**.

Estimates:
- **A space-deck-builder-scale PPO run** (about 800k games/seed, about 1 h on one GPU, per the brief). Their policy is tiny (0.28M parameters) and their run was CPU-engine-bound plus PPO optimisation **[V]**.
  - One consumer GPU (4090/5090) with 16-32 vCPUs is enough. Roughly **$0.35-$1.50 per seed-hour** on RunPod, or about $1-3 on Modal including CPU cores.
  - Ten seeds and a few ablations come to **tens of dollars**. This is an estimate from the listed prices, not a measured run.
- **AlphaZero/expert-iteration with search.** Cost scales with simulations per move × games.
  - Reference points:
    - Generals.io: 7 days × 4 GPUs, about 670 GPU-hours **[A]**, which is about $230 on RunPod 4090 community or about $1.8k on H100 SXM community.
    - AlphaZero.jl Connect Four: about 1 hour per iteration on one RTX 2070 **[A]**, so tens of GPU-hours per run.
    - space-deck-builder's single round of expert iteration gained +19 Elo **[V]**.
  - A plausible budget for Star Realms is **tens to a few hundred GPU-hours plus a many-core CPU fleet**, about **$20-$500 per serious run** at community-cloud prices **[U, extrapolation]**.
  - Search-heavy self-play is usually limited by CPU and inference latency, not by the GPU **[U]**. Price the CPU cores: 64 cores × 100 h is about $300 on Modal.

### 4.6 Revised ranked shortlist (training anywhere, play on the M1)

1. **Numba engine (CPU, many cores) + PyTorch learner on a rented GPU; on the M1, same engine + NumPy/Accelerate or PyTorch-CPU forward pass inside batched search.** Unchanged at #1.
   - It is the only stack demonstrated end to end on Star Realms, including cloud-style CPU-engine/GPU-learner training, search and expert iteration **[V]**.
   - One language, and identical code for training and M1 play.
   - Weaknesses as in section 3: untyped kernels, and correctness rests on differential tests.
2. **Rust engine + search (PyO3/maturin) + PyTorch learner on a rented GPU; on the M1, Rust search with an in-process hand-written MLP or ort.** Unchanged at #2, slightly strengthened.
   - Removing the M1-training constraint doesn't hurt it.
   - Owning search natively in Rust gives the best single-game play latency on the M1 with no Python in the loop.
   - It also gives the best rules safety.
3. **PufferLib 5.0 C env (CPU backend) + CUDA PuffeRL with built-in self-play and masks; on the M1, the C env + `puffercpu.c` forward pass.** New entry, displacing generic C++.
   - Strongest turnkey PPO + self-play + C-inference export story **[V]**.
   - But: rules in C, no sanitizer on macOS, no built-in search-based training, PPO-centric, a fast-moving codebase, and self-play limited to CPU-stepped envs.
4. **JAX engine (Pgx-style) + mctx on rented GPUs; on the M1, CPU JAX.** Now a contender rather than ruled out, but last.
   - Rule-rich JAX games train fast (Craftax, Mahjax, Generals.io) **[A]**.
   - But: no deckbuilder precedent, select-over-all-branches cost for card-effect dispatch **[V]**, tens-of-seconds jit compiles **[V]**, weak single-game search latency on M1 CPU **[U]**, and the most rule-rewrite risk.

Still out: OpenSpiel as the engine host (in-tree C++, AlphaZero assumes perfect information), plus Mojo, Julia, Codon and Taichi.

**Spike additions for the new constraint:**
- (a) On a rented 4090 or 5090 with 16-32 vCPUs: Numba vs Rust engine decisions/s at batch 16k with PPO in the loop. Is it policy-bound?
- (b) On the M1: single-game ISMCTS/PIMC decisions/s and move latency with each inference option (hand-written NumPy MLP, PyTorch CPU, PyTorch MPS, ORT CPU, ORT CoreML, and Rust in-process) at batch sizes 1-256.
- (c) If PufferLib is considered: port the Gate-2 subset (play, buy, attack, end turn, draw) to a CPU C env and check that `--cpu` eval builds on macOS.
- (d) If JAX is considered: implement the same subset and measure steps/s per A100 or 4090 against Numba-per-core-dollar, plus jit compile time.
