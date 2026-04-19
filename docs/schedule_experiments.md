# Halide schedule experiments — history & decision log

Target SoC: **Samsung Exynos 2600 / SM8850** (arm64-v8a, Cortex-X5 big
cluster × 4). All measurements on-device; `CLOCK_MONOTONIC_RAW`; trimmed
mean with top/bottom 5% dropped; cv ≤ 0.20 stability gate; risk-adjusted
score `trimmed_mean × (1 + 2·cv)`.

This file exists so **future engineers don't repeat these experiments**.
Read it before re-opening the schedule search. The short version:

> We swept 4 hand-tuned presets and Halide 21's Mullapudi2016
> autoscheduler across 52 (op, src, dst, rot) cells at prod resolutions.
> Preset **v1 — narrow-parallel (split y by 16, parallel, vectorize 16)**
> won or tied the majority of cells. Mullapudi2016 was decisively worse
> (1 win / 17 ties / 34 losses, median −19.7 %). The codebase is now
> hard-coded to v1 and all sweep scaffolding was removed.

---

## Scope of what got tried

### Hand-tuned preset ladder

Four schedules applied uniformly across the 3-channel interleaved
outputs (`nv21_resize_rgb_bt709_*` and `rotate_fixed`):

| Preset | y split | parallel dim | vector width | Notes |
|---|---|---|---|---|
| v0 | 32  | outer y | 16 (u8) / 8 (area)  | baseline default |
| **v1** | **16** | **outer y** | **16 (u8) / 8 (area)**  | **narrow-parallel — winner** |
| v2 | 64  | outer y | 16 (u8) / 8 (area)  | wide-parallel |
| v3 | 32  | outer y | 32 (u8) / 16 (area) | wider vector |

Rotate used the same ladder with `vectorize(x, 16)` throughout; v3 went
`vec 32, unsplit`. The `area` interp uses a smaller vector lane (8) because
its reduction prevents wider lanes from being efficient.

### Mullapudi2016 autoscheduler

Halide 21's heuristic, platform-agnostic autoscheduler. Loaded via
`-p libautoschedule_mullapudi2016.so`, invoked with the generator param
`autoscheduler.name=Mullapudi2016`. Generators gained `set_estimates()`
at the prod-median shape (src 2296×3056, dst 640×640 for resize;
src 3000×4000 3-ch interleaved for rotate) plus a `using_autoscheduler()`
early-return so the manual schedule wouldn't collide. Autoscheduler
params: `parallelism=4` (Cortex-X5 width), `last_level_cache_size=8388608`
(8 MiB SM8850 L3 estimate).

### Schedulers we deliberately did NOT try

- **Adams2019 / Anderson2021**: neural cost models trained on x86.
  Their cost model has no concept of NEON dotprod, arm64 LSE, or the
  SM8850 memory subsystem. If the heuristic Mullapudi already loses by
  20 %, an x86-trained ML scheduler is not going to close that gap —
  it has no signal to learn from on this target. Worth revisiting only
  if someone ports a training run to arm64 hardware.
- **Li2018**: GPU-only. We're running on the CPU cluster.

---

## Results

### Per-preset win count (hand-tuned only)

Risk-adjusted winner per cell, counted across 52 cells.

| Op / interp | cells | v0 wins | **v1 wins** | v2 wins | v3 wins |
|---|--:|--:|--:|--:|--:|
| `nv21_resize_rgb/nearest` | 15 | 2 | **10** | 1 | 2 |
| `nv21_resize_rgb/linear`  | 15 | 0 | **10** | 3 | 2 |
| `nv21_resize_rgb/area`    | 13 | 1 | **12** | 0 | 0 |
| `rotate` (3-ch, 90/180/270) | 9 | 2 | **5** | 2 | 0 |
| **TOTAL** | **52** | **5** | **37** | **6** | **4** |

v1 is the majority winner across every category. Median speed-up vs the
v0 baseline is **+18 %**, peak **+62.9 %** on rotate 90° @ 3000×4000.

### v1 vs v0 (picked operations)

| Config | v0 tmean | v1 tmean | Δ |
|---|--:|--:|--:|
| `nv21_resize_rgb/linear 2296×3056 → 640×640` | 724 µs | 585 µs | **−19 %** |
| `nv21_resize_rgb/area 2296×3056 → 640×640`   | 1.52 ms | 1.44 ms | **−5 %** |
| `rotate 90° @ 3000×4000 3-ch` | 16.1 ms | 10.3 ms | **−36 %** |

### Mullapudi2016 vs best-per-cell hand-tuned

Positive Δ% = autoscheduler faster; win = ≥10 % faster, loss = ≥10 % slower.

| Op / interp | cells | median Δ% | mean Δ% | wins | ties | losses |
|---|--:|--:|--:|--:|--:|--:|
| `nv21_resize_rgb/nearest` | 15 | −12.0 | −23.3 | 0 | 6 | 9 |
| `nv21_resize_rgb/linear`  | 15 | −38.4 | −30.7 | 0 | 5 | 10 |
| `nv21_resize_rgb/area`    | 13 | −11.7 | −20.1 | 1 | 5 | 7 |
| `rotate`                  |  9 | −49.0 | −58.7 | 0 | 1 | 8 |
| **TOTAL**                 | **52** | **−19.7** | **−30.7** | **1** | **17** | **34** |

Worst-case autoscheduler losses:

- `rotate 90° @ 3000×4000` — auto 21.5 ms vs hand 10.3 ms — **2.09×**
- `rotate 90° @ 2296×3056` — auto 9.5 ms vs hand 4.8 ms — **1.98×**
- `nv21_resize_rgb/linear 3000×4000 → 384×384` — auto 772 µs vs hand 424 µs — **1.82×**

---

## Why Mullapudi2016 loses here

Two structural reasons visible in the worst-case cells:

1. **Rotate is memory-copy, not compute.** `rotate_fixed` is pure index
   remapping — one uint8 load, one uint8 store, per output pixel.
   Mullapudi's cost model treats it as compute-heavy and tiles
   aggressively; the extra tile-loop bookkeeping dominates the ~1–2 GB/s
   memory copy the op actually is. Hand-tuned v1 just streams memory
   row-banded: `split y 16 + parallel + vectorize 16`. No tiling, no
   producer fusion, just bandwidth.
2. **The fused NV21 pipeline has multiple small producer stages**
   (`y_resized`, the u/v reduction). Mullapudi tries to tile-and-fuse
   them in ways that evict each other from L1. Hand schedules do
   `compute_at(output, yi)` once per output row band, matching the SoC's
   64 KB L1 footprint. The autoscheduler has no signal for this cache
   budget — its `last_level_cache_size` param only models L3.

---

## Decision

**Production AOT uses v1 unconditionally.** Generators hard-code the
schedule; there is no `schedule_preset` GeneratorParam and no `--variant`
bench flag. The `_sched*` / `_auto_mull` archives and their dispatchers
were removed (commit following these experiments).

If we retarget to a new SoC with meaningfully different L1/L3 / core
topology (e.g., a tablet with a different big-cluster width, or a newer
SoC generation), **re-run this sweep before deciding again**. The result
above is SM8850-specific; it is not a general statement about
autoscheduling.

### When to revisit

Re-open the schedule search if any of the following becomes true:

- Target SoC changes (different big-cluster count / L1 / L3 / memory BW).
- Halide bumps to a new major version with a redesigned autoscheduler
  (Halide 22+ if/when it lands).
- An x86→arm64 port of the Adams2019 / Anderson2021 cost model training
  pipeline ships (at which point the ML schedulers become worth
  retrying — their cost model would finally have arm64 signal).
- A new op is added whose access pattern is fundamentally different from
  the current set (e.g., heavy reductions, stencils > 5×5, FP math).

### What to keep, what to throw away on re-run

Keep:

- The 52-cell prod matrix (`bench_results_prod.csv` dimensions).
- The cv ≤ 0.20 stability gate + 45 s cooldown between rows.
- The risk-adjusted score `trimmed_mean × (1 + 2·cv)` with k = 2.0.

Throw away:

- Any assumption that v1 is still the winner. Re-measure.
- The Mullapudi2016-only scope — if the target changed enough to
  warrant re-running, also try whatever autoscheduler ships with the
  newer Halide.

---

## How to reproduce (if you must)

The sweep drivers (`run_schedule_sweep.sh`, `run_autoscheduler_sweep.sh`)
and the preset/autoscheduler AOT variants (`*_sched{0..3}.a`,
`*_auto_mull.a`) were **deleted** — they are visible in git history at
the commit that immediately precedes the v1 lock-in. To reproduce:

1. `git checkout` the commit before the cleanup.
2. `bash halide/build_generators.sh` — rebuilds the sweep variants.
3. `bash app/src/bench_exe/run_bench.sh --build-only`
4. `bash app/src/bench_exe/run_schedule_sweep.sh | tee bench_schedule_sweep.csv`
5. `bash app/src/bench_exe/run_autoscheduler_sweep.sh | tee bench_autoscheduler.csv`
6. Post-process with the Python snippet that used to live in
   `app/src/bench_exe/README.md` (also in git history).

Do not restore any of that scaffolding on `main` — it's been
deliberately removed and this document is why.
