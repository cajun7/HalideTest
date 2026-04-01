#!/bin/bash
# Build GoogleTest native tests and run on device via adb.
#
# Prerequisites:
#   - Android NDK installed (set ANDROID_NDK_HOME or NDK_ROOT)
#   - Device connected via adb
#   - Halide generators already built (run halide/build_generators.sh first)
#   - OpenCV Android SDK extracted (run scripts/setup_opencv.sh first)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

# Find NDK
NDK_ROOT="${ANDROID_NDK_HOME:-${NDK_ROOT:-${ANDROID_HOME:-/opt/homebrew/share/android-ndk}/ndk}}"
if [ ! -f "${NDK_ROOT}/ndk-build" ]; then
    # Try common locations
    for candidate in \
        "${HOME}/Library/Android/sdk/ndk/"*"/ndk-build" \
        "/opt/homebrew/share/android-ndk/ndk-build" \
        "${ANDROID_HOME:-/dev/null}/ndk/"*"/ndk-build"; do
        if [ -f "$candidate" ]; then
            NDK_ROOT="$(dirname "$candidate")"
            break
        fi
    done
fi

if [ ! -f "${NDK_ROOT}/ndk-build" ]; then
    echo "ERROR: Cannot find ndk-build. Set ANDROID_NDK_HOME."
    exit 1
fi

echo "Using NDK: ${NDK_ROOT}"

OPENCV_VERSION="${OPENCV_VERSION:-3.4.16}"
echo "Using OpenCV: ${OPENCV_VERSION}"
echo ""

echo "=== Building test executable ==="
"${NDK_ROOT}/ndk-build" \
    NDK_PROJECT_PATH="${SCRIPT_DIR}" \
    APP_BUILD_SCRIPT="${SCRIPT_DIR}/jni/Android.mk" \
    NDK_APPLICATION_MK="${SCRIPT_DIR}/jni/Application.mk" \
    NDK_OUT="${SCRIPT_DIR}/obj" \
    NDK_LIBS_OUT="${SCRIPT_DIR}/libs" \
    OPENCV_VERSION="${OPENCV_VERSION}" \
    -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo "=== Pushing test binary to device ==="
adb push "${SCRIPT_DIR}/libs/arm64-v8a/halide_tests" /data/local/tmp/
adb shell chmod 755 /data/local/tmp/halide_tests

echo ""
echo "=== Running tests on device ==="
adb shell /data/local/tmp/halide_tests "$@"
