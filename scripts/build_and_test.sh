#!/bin/bash
# Unified build-and-test script for the HalideTest project.
#
# Usage:
#   bash scripts/build_and_test.sh apk              # Build debug APK only
#   bash scripts/build_and_test.sh apk --install     # Build + install APK on device
#   bash scripts/build_and_test.sh run               # Build APK, uninstall old, install, launch
#   bash scripts/build_and_test.sh test              # Build & run all GoogleTests on device
#   bash scripts/build_and_test.sh test --filter='ResizeTarget*'
#                                                     # Run specific test filter
#   bash scripts/build_and_test.sh all               # Build APK + run all tests
#   bash scripts/build_and_test.sh all --install --filter='Flip*'
#                                                     # Build+install APK, run filtered tests
#   bash scripts/build_and_test.sh generators         # Rebuild Halide AOT generators
#
# Environment variables:
#   OPENCV_VERSION   Override OpenCV version (default: 3.4.16)
#   ANDROID_NDK_HOME Path to Android NDK
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
TEST_DIR="${PROJECT_ROOT}/app/src/test_native"

OPENCV_VERSION="${OPENCV_VERSION:-3.4.16}"
APP_PACKAGE="com.example.halidetest"
APP_ACTIVITY="${APP_PACKAGE}/.MainActivity"

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
COMMAND=""
INSTALL_APK=false
GTEST_FILTER=""
GTEST_EXTRA_ARGS=()

usage() {
    cat <<EOF
${BOLD}Usage:${NC}  bash scripts/build_and_test.sh <command> [options]

${BOLD}Commands:${NC}
  apk          Build debug APK
  run          Build APK, uninstall old, install new, and launch the app
  test         Build & run GoogleTest native tests on device
  all          Build APK + run tests
  generators   Rebuild Halide AOT generators

${BOLD}Options:${NC}
  --install              Install APK on connected device after build
  --filter=<pattern>     GoogleTest filter (e.g. --filter='Flip*')
  --opencv=<version>     Override OpenCV version (default: ${OPENCV_VERSION})
  --release              Build release APK instead of debug
  --help                 Show this help

${BOLD}Examples:${NC}
  bash scripts/build_and_test.sh run                 # Full cycle: build, deploy, launch
  bash scripts/build_and_test.sh apk --install
  bash scripts/build_and_test.sh test --filter='FusedBilinear*'
  bash scripts/build_and_test.sh all --install --filter='Resize*'
  OPENCV_VERSION=3.4.16 bash scripts/build_and_test.sh all
EOF
    exit 0
}

BUILD_TYPE="Debug"

if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --install)
            INSTALL_APK=true
            ;;
        --filter=*)
            GTEST_FILTER="${1#--filter=}"
            ;;
        --opencv=*)
            OPENCV_VERSION="${1#--opencv=}"
            ;;
        --release)
            BUILD_TYPE="Release"
            ;;
        --help|-h)
            usage
            ;;
        --gtest_*)
            # Pass through raw gtest args
            GTEST_EXTRA_ARGS+=("$1")
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
    shift
done

# ---------------------------------------------------------------
# Helper: find NDK
# ---------------------------------------------------------------
find_ndk() {
    local ndk="${ANDROID_NDK_HOME:-${NDK_ROOT:-}}"
    if [ -n "${ndk}" ] && [ -f "${ndk}/ndk-build" ]; then
        echo "${ndk}"
        return
    fi
    for candidate in \
        "${HOME}/Android/Sdk/ndk/"*"/ndk-build" \
        "${HOME}/Library/Android/sdk/ndk/"*"/ndk-build" \
        "/opt/homebrew/share/android-ndk/ndk-build" \
        "${ANDROID_HOME:-/dev/null}/ndk/"*"/ndk-build"; do
        if [ -f "$candidate" ] 2>/dev/null; then
            dirname "$candidate"
            return
        fi
    done
    echo ""
}

# ---------------------------------------------------------------
# Build APK
# ---------------------------------------------------------------
build_apk() {
    local gradle_task="assemble${BUILD_TYPE}"
    echo -e "${CYAN}${BOLD}=== Building ${BUILD_TYPE} APK (OpenCV ${OPENCV_VERSION}) ===${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    ./gradlew "${gradle_task}" -PopencvVersion="${OPENCV_VERSION}"

    local apk_dir
    if [ "${BUILD_TYPE}" = "Release" ]; then
        apk_dir="${PROJECT_ROOT}/app/build/outputs/apk/release"
    else
        apk_dir="${PROJECT_ROOT}/app/build/outputs/apk/debug"
    fi

    echo ""
    echo -e "${GREEN}${BOLD}APK built successfully.${NC}"
    ls -lh "${apk_dir}"/*.apk 2>/dev/null || true
    echo ""

    if [ "${INSTALL_APK}" = true ]; then
        echo -e "${CYAN}=== Installing APK on device ===${NC}"
        local apk_file
        apk_file=$(find "${apk_dir}" -name "*.apk" | head -1)
        if [ -z "${apk_file}" ]; then
            echo -e "${RED}ERROR: No APK found in ${apk_dir}${NC}"
            return 1
        fi
        adb install -r "${apk_file}"
        echo -e "${GREEN}APK installed.${NC}"
        echo ""
    fi
}

# ---------------------------------------------------------------
# Deploy APK: uninstall old, install new, launch
# ---------------------------------------------------------------
deploy_and_launch() {
    local apk_dir
    if [ "${BUILD_TYPE}" = "Release" ]; then
        apk_dir="${PROJECT_ROOT}/app/build/outputs/apk/release"
    else
        apk_dir="${PROJECT_ROOT}/app/build/outputs/apk/debug"
    fi

    local apk_file
    apk_file=$(find "${apk_dir}" -name "*.apk" 2>/dev/null | head -1)
    if [ -z "${apk_file}" ]; then
        echo -e "${RED}ERROR: No APK found in ${apk_dir}. Build first.${NC}"
        return 1
    fi

    echo -e "${CYAN}=== Uninstalling old APK ===${NC}"
    adb uninstall "${APP_PACKAGE}" 2>/dev/null || true

    echo -e "${CYAN}=== Installing new APK ===${NC}"
    adb install "${apk_file}"
    echo -e "${GREEN}APK installed.${NC}"
    echo ""

    echo -e "${CYAN}=== Launching app ===${NC}"
    adb shell am start -n "${APP_ACTIVITY}"
    echo -e "${GREEN}${BOLD}App launched.${NC}"
}

# ---------------------------------------------------------------
# Build & run native tests
# ---------------------------------------------------------------
build_and_run_tests() {
    local ndk_root
    ndk_root="$(find_ndk)"
    if [ -z "${ndk_root}" ] || [ ! -f "${ndk_root}/ndk-build" ]; then
        echo -e "${RED}ERROR: Cannot find ndk-build. Set ANDROID_NDK_HOME.${NC}"
        exit 1
    fi

    echo -e "${CYAN}${BOLD}=== Building GoogleTest executable ===${NC}"
    echo -e "  NDK:    ${ndk_root}"
    echo -e "  OpenCV: ${OPENCV_VERSION}"
    echo ""

    "${ndk_root}/ndk-build" \
        NDK_PROJECT_PATH="${TEST_DIR}" \
        APP_BUILD_SCRIPT="${TEST_DIR}/jni/Android.mk" \
        NDK_APPLICATION_MK="${TEST_DIR}/jni/Application.mk" \
        NDK_OUT="${TEST_DIR}/obj" \
        NDK_LIBS_OUT="${TEST_DIR}/libs" \
        OPENCV_VERSION="${OPENCV_VERSION}" \
        -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    echo ""
    echo -e "${CYAN}=== Pushing test binary to device ===${NC}"
    adb push "${TEST_DIR}/libs/arm64-v8a/halide_tests" /data/local/tmp/
    adb shell chmod 755 /data/local/tmp/halide_tests

    echo ""
    echo -e "${CYAN}${BOLD}=== Running tests on device ===${NC}"

    # Build gtest command
    local cmd="/data/local/tmp/halide_tests"
    if [ -n "${GTEST_FILTER}" ]; then
        cmd="${cmd} --gtest_filter=${GTEST_FILTER}"
        echo -e "  Filter: ${YELLOW}${GTEST_FILTER}${NC}"
    fi
    for arg in "${GTEST_EXTRA_ARGS[@]+"${GTEST_EXTRA_ARGS[@]}"}"; do
        cmd="${cmd} ${arg}"
    done
    echo ""

    # Run and capture exit code
    local exit_code=0
    adb shell "${cmd}" || exit_code=$?

    echo ""
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}${BOLD}All tests PASSED.${NC}"
    else
        echo -e "${RED}${BOLD}Some tests FAILED (exit code: ${exit_code}).${NC}"
    fi
    return ${exit_code}
}

# ---------------------------------------------------------------
# Build Halide generators
# ---------------------------------------------------------------
build_generators() {
    echo -e "${CYAN}${BOLD}=== Building Halide AOT generators ===${NC}"
    echo ""
    bash "${PROJECT_ROOT}/halide/build_generators.sh"
    echo ""
    echo -e "${GREEN}${BOLD}Generators built successfully.${NC}"
}

# ---------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------
overall_exit=0

case "${COMMAND}" in
    apk)
        build_apk
        ;;
    run)
        build_apk
        deploy_and_launch
        ;;
    test)
        build_and_run_tests || overall_exit=$?
        ;;
    all)
        build_apk
        deploy_and_launch
        build_and_run_tests || overall_exit=$?
        ;;
    generators|gen)
        build_generators
        ;;
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        echo ""
        usage
        ;;
esac

exit ${overall_exit}
