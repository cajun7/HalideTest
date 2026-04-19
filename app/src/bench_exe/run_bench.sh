#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — build, push, and sweep the benchmark matrix on-device.
#
# Usage:
#   bash app/src/bench_exe/run_bench.sh                      # full matrix -> CSV on stdout
#   bash app/src/bench_exe/run_bench.sh --build-only         # just build (local)
#   bash app/src/bench_exe/run_bench.sh --smoke              # one quick row
#
# Output format: CSV. First line is the header. One row per (backend, op,
# interp, resolution, stress) combination.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

NDK_ROOT="${ANDROID_NDK_HOME:-${NDK_ROOT:-${ANDROID_HOME:-/opt/homebrew/share/android-ndk}/ndk}}"
if [ ! -f "${NDK_ROOT}/ndk-build" ]; then
    for c in "${HOME}/Library/Android/sdk/ndk/"*"/ndk-build" \
             "/opt/homebrew/share/android-ndk/ndk-build"; do
        [ -f "$c" ] && { NDK_ROOT="$(dirname "$c")"; break; }
    done
fi
[ -f "${NDK_ROOT}/ndk-build" ] || { echo "ndk-build not found"; exit 1; }

OPENCV_VERSION="${OPENCV_VERSION:-3.4.16}"
MODE="${1:-full}"

echo "=== Building bench executable ===" >&2
"${NDK_ROOT}/ndk-build" \
    NDK_PROJECT_PATH="${SCRIPT_DIR}" \
    APP_BUILD_SCRIPT="${SCRIPT_DIR}/jni/Android.mk" \
    NDK_APPLICATION_MK="${SCRIPT_DIR}/jni/Application.mk" \
    NDK_OUT="${SCRIPT_DIR}/obj" \
    NDK_LIBS_OUT="${SCRIPT_DIR}/libs" \
    OPENCV_VERSION="${OPENCV_VERSION}" \
    -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) >&2

[ "$MODE" = "--build-only" ] && { echo "built: ${SCRIPT_DIR}/libs/arm64-v8a/bench" >&2; exit 0; }

echo "=== Pushing to device ===" >&2
adb push "${SCRIPT_DIR}/libs/arm64-v8a/bench" /data/local/tmp/ >/dev/null
adb shell chmod 755 /data/local/tmp/bench

BENCH() { adb shell /data/local/tmp/bench "$@"; }

# Thermal-aware cooldown: sleep `base` seconds, then optionally poll the
# SoC thermal sysfs until temp falls below 40°C. The sysfs read is
# best-effort — devices that restrict it just fall back to the plain sleep.
cooldown() {
    local base="$1"
    sleep "$base"
    local t
    if t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | tr -d '\r'); then
        [ -z "$t" ] && return 0
        # temp is reported in milli-°C on most SoCs (e.g. 45000 = 45°C).
        while [ "$t" -gt 40000 ] 2>/dev/null; do
            sleep 10
            t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | tr -d '\r')
            [ -z "$t" ] && break
        done
    fi
}

# upscale?: returns 0 (true) if dst-area > src-area
upscale() {
    local s="$1" d="$2"
    local sw="${s%%x*}"; local sh="${s##*x}"
    local dw="${d%%x*}"; local dh="${d##*x}"
    [ $((dw * dh)) -gt $((sw * sh)) ]
}

if [ "$MODE" = "--smoke" ]; then
    BENCH --csv-header \
          --backend=halide --op=nv21_resize_rgb --interp=linear \
          --resolution=1920x1080 --dst=640x480 --iters=50 --stress=0
    exit 0
fi

if [ "$MODE" = "--prod" ]; then
    # -------------------------------------------------------------------------
    # Production-resolution matrix. Portrait 3:4 camera sources and square ML
    # preprocessor destinations. stress=0 only (clean numbers). 300 iters
    # per row for tight confidence intervals; 45s cooldown between rows and
    # 120s between op-groups to keep the SoC in a steady thermal band.
    # -------------------------------------------------------------------------
    PROD_SRCS=(1080x1440 2296x3056 3000x4000)
    PROD_DSTS=(384x384 518x518 640x640 1280x1280 1408x1408)

    # Header once.
    BENCH --csv-header \
          --backend=halide --op=rotate --rot=90 \
          --resolution=1080x1440 --iters=10 --stress=0

    # NV21 -> resize -> BT.709 RGB  (every src × dst × interp, both backends)
    for backend in halide opencv_neon; do
      for interp in nearest linear area; do
        for src in "${PROD_SRCS[@]}"; do
          for dst in "${PROD_DSTS[@]}"; do
            # area is a downscale-only resampling mode; OpenCV falls back to
            # bilinear for upscale, which would be apples-to-oranges.
            if [ "$interp" = "area" ] && upscale "$src" "$dst"; then continue; fi
            BENCH --backend=$backend --op=nv21_resize_rgb --interp=$interp \
                  --resolution=$src --dst=$dst --stress=0 \
                  --iters=300 --warmup=30 || true
            cooldown 45
          done
        done
        cooldown 120   # op-group rest between interp modes
      done
    done

    # Rotate 3-ch (fixed angles)
    for backend in halide opencv_neon; do
      for src in "${PROD_SRCS[@]}"; do
        for rot in 90 180 270; do
          BENCH --backend=$backend --op=rotate --rot=$rot \
                --resolution=$src --stress=0 \
                --iters=300 --warmup=30 || true
          cooldown 45
        done
      done
      cooldown 120
    done
    exit 0
fi

# ---------- Matrix ----------
# Header once.
BENCH --csv-header \
      --backend=halide --op=rotate --rot=90 \
      --resolution=1920x1080 --iters=50 --stress=0

# Rotate 3-ch (fixed angles)
for backend in halide opencv_neon; do
  for rot in 90 180 270; do
    for res in 1280x720 1920x1080 3840x2160; do
      for stress in 0 2 4; do
        BENCH --backend=$backend --op=rotate --rot=$rot \
              --resolution=$res --stress=$stress --iters=200 --warmup=20 || true
      done
    done
  done
done

# Rotate 1-ch (masks)
for backend in halide opencv_neon; do
  for rot in 90 180 270; do
    for res in 640x480 1280x720 1920x1080; do
      for stress in 0 2 4; do
        BENCH --backend=$backend --op=rotate_1c --rot=$rot \
              --resolution=$res --stress=$stress --iters=200 --warmup=20 || true
      done
    done
  done
done

# NV21 -> BT.709 RGB (size unchanged)
for backend in halide opencv_neon; do
  for res in 1280x720 1920x1080 3840x2160; do
    for stress in 0 2 4; do
      BENCH --backend=$backend --op=nv21_to_rgb \
            --resolution=$res --stress=$stress --iters=200 --warmup=20 || true
    done
  done
done

# Fused NV21 resize + BT.709 RGB
for backend in halide opencv_neon; do
  for interp in nearest linear area; do
    # downscale (the common case), same-size, and upscale (not for area)
    for cfg in "1920x1080:640x480" "1920x1080:1280x720" "1280x720:640x480" "640x480:1280x720"; do
      src="${cfg%%:*}"; dst="${cfg##*:}"
      if [ "$interp" = "area" ] && [ "$src" = "640x480" ]; then continue; fi
      for stress in 0 2 4; do
        BENCH --backend=$backend --op=nv21_resize_rgb --interp=$interp \
              --resolution=$src --dst=$dst --stress=$stress --iters=200 --warmup=20 || true
      done
    done
  done
done
