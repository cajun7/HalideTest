# bench — on-device Halide-vs-OpenCV benchmark

Standalone arm64 executable pushed to the phone via adb. Prints **one CSV row
per run** to stdout. No JNI, no gradle; build with ndk-build and go.

```
BINARY:       /data/local/tmp/bench        (pushed by run_bench.sh)
SOURCE:       app/src/bench_exe/bench_main.cpp
CLOCK:        CLOCK_MONOTONIC_RAW
```

The Halide AOT pipelines linked in here use the single locked-in schedule —
the empirical sweep that chose it is documented in
[docs/schedule_experiments.md](../../../docs/schedule_experiments.md). Don't
re-run that sweep without reading the doc first.

---

## Quick start

```bash
# Build + push + smoke test in one shot
bash app/src/bench_exe/run_bench.sh --smoke

# Run the production-resolution sweep (~90 min, writes CSV to stdout)
bash app/src/bench_exe/run_bench.sh --prod | tee bench_results_prod.csv
```

`--build-only` just builds the executable without pushing.

---

## CLI flags

| Flag | Values | Applies to | Notes |
|---|---|---|---|
| `--backend=` | `halide`, `opencv_neon` | all ops | |
| `--op=` | `rotate`, `rotate_1c`, `nv21_to_rgb`, `nv21_resize_rgb` | — | required |
| `--interp=` | `nearest`, `linear`, `area` | `nv21_resize_rgb` | |
| `--resolution=` | `WxH` | all | source; default 1920×1080 |
| `--dst=` | `WxH` | `nv21_resize_rgb` | destination; default = src/2 |
| `--rot=` | `90`, `180`, `270`, `0` | `rotate`, `rotate_1c` | `0` = arbitrary (uses `--angle`) |
| `--angle=` | degrees | `rotate` arbitrary | |
| `--iters=` | N | all | measured iterations |
| `--warmup=` | M | all | untimed warmup passes |
| `--stress=` | 0..8 | all | # DRAM-thrash background threads |
| `--seed=` | int | all | PRNG seed for random input |
| `--input-file=` | path | all | raw bytes instead of random |
| `--csv-header` | flag | — | prints header line first |

---

## Example invocations

```bash
# Halide fused NV21 -> bilinear 640x640 RGB, portrait source
adb shell /data/local/tmp/bench --csv-header \
    --backend=halide --op=nv21_resize_rgb --interp=linear \
    --resolution=2296x3056 --dst=640x640 --iters=300 --warmup=30

# OpenCV reference, same config
adb shell /data/local/tmp/bench \
    --backend=opencv_neon --op=nv21_resize_rgb --interp=linear \
    --resolution=2296x3056 --dst=640x640 --iters=300 --warmup=30

# Rotate 3-ch
adb shell /data/local/tmp/bench \
    --backend=halide --op=rotate --rot=90 \
    --resolution=3000x4000 --iters=300 --warmup=30

# Rotate 1-ch mask
adb shell /data/local/tmp/bench \
    --backend=halide --op=rotate_1c --rot=180 \
    --resolution=1080x1440 --iters=200

# NV21 -> RGB full-frame (no resize), under DRAM contention
adb shell /data/local/tmp/bench \
    --backend=halide --op=nv21_to_rgb \
    --resolution=3840x2160 --stress=4 --iters=200
```

---

## CSV output

```
backend,op,interp,src_w,src_h,dst_w,dst_h,rot,stress,iters,
min_us,mean_us,trimmed_mean_us,stddev_us,cv,p50_us,p95_us,p99_us,max_us
```

| Column | Meaning |
|---|---|
| `min_us` | Fastest single iteration — a lower bound on what the op can do. |
| `mean_us` | Plain arithmetic mean. |
| `trimmed_mean_us` | Mean after dropping the top 5% and bottom 5% of samples. More robust against thermal/scheduler outliers than `mean_us`, without throwing away the tail structure the way `p50` does. |
| `stddev_us` | Standard deviation. |
| `cv` | `stddev / mean`. Unitless — comparable across resolutions. |
| `p50_us` | Median per-iteration cost. |
| `p95_us` | Worst 5% of frames — **the oncall number** for sustained load. |
| `p99_us` | Frame-drop risk signal. |
| `max_us` | Worst single iteration (often a scheduler artifact — don't over-index). |

### How to read it

For a real-time camera pipeline, `trimmed_mean_us` + low `cv` is what matters.
A schedule with median 2.0 ms ± 0.1 ms beats one with median 1.8 ms ± 0.6 ms
every time — the 0.6 ms tail translates directly to dropped frames. `p50` on
its own hides bimodal distributions caused by thermal state transitions, so
always cross-check against `p95`/`p99`.

Expected Halide-vs-OpenCV speedup range from previous landscape-resolution
measurements: **2×–3.5× typical**, with `area` being the only op where hand-
tuned NEON still wins (Halide's reduction scheduling there is the bottleneck).

---

## Files in this directory

- `bench_main.cpp` — executable entry point + per-op dispatch.
- `bench_clock.h` — `CLOCK_MONOTONIC_RAW` timing + `Stats` (trimmed mean, cv, percentiles).
- `bench_stress.{h,cpp}` — background DRAM-thrashing stressor threads.
- `jni/Android.mk`, `jni/Application.mk` — ndk-build config.
- `run_bench.sh` — build + push + default/prod sweep driver.
