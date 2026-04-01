#!/bin/bash
# Build OpenCV from source for Android arm64-v8a with libc++ (c++_static).
#
# Pre-built OpenCV 3.4.x Android SDKs use gnustl, which is incompatible
# with NDK r18+. This script rebuilds the static libraries from source
# using libc++, then replaces them in the pre-built SDK directory.
#
# Prerequisites:
#   - Pre-built SDK already downloaded via setup_opencv.sh
#   - cmake (3.10+) on PATH or at ~/.local/bin/cmake
#   - Android NDK (set ANDROID_NDK_HOME or NDK_ROOT)
#
# Usage:
#   OPENCV_VERSION=3.4.16 bash scripts/build_opencv_source.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
OPENCV_DIR="${PROJECT_ROOT}/opencv"
VERSION="${OPENCV_VERSION:-3.4.16}"
SDK_DIR="${OPENCV_DIR}/${VERSION}/OpenCV-android-sdk"

ABI="arm64-v8a"
ANDROID_PLATFORM="android-24"
BUILD_DIR="/tmp/opencv-build-${VERSION}"
SOURCE_DIR="/tmp/opencv-source-${VERSION}"

# ---------------------------------------------------------------
# Verify SDK exists (headers + .mk files will be reused)
# ---------------------------------------------------------------
if [ ! -f "${SDK_DIR}/sdk/native/jni/OpenCV.mk" ]; then
    echo "ERROR: Pre-built SDK not found at ${SDK_DIR}"
    echo "Run first: OPENCV_VERSION=${VERSION} bash scripts/setup_opencv.sh"
    exit 1
fi

# ---------------------------------------------------------------
# Find CMake
# ---------------------------------------------------------------
CMAKE="$(command -v cmake 2>/dev/null || echo "${HOME}/.local/bin/cmake")"
if [ ! -x "${CMAKE}" ]; then
    echo "ERROR: cmake not found. Install with: pip3 install cmake --user"
    exit 1
fi
echo "Using CMake: ${CMAKE} ($(${CMAKE} --version | head -1))"

# ---------------------------------------------------------------
# Find NDK
# ---------------------------------------------------------------
NDK_ROOT="${ANDROID_NDK_HOME:-${NDK_ROOT:-}}"
if [ -z "${NDK_ROOT}" ] || [ ! -f "${NDK_ROOT}/ndk-build" ]; then
    for candidate in \
        "${HOME}/Android/Sdk/ndk/"*"/ndk-build" \
        "${HOME}/Library/Android/sdk/ndk/"*"/ndk-build" \
        "/opt/homebrew/share/android-ndk/ndk-build"; do
        if [ -f "$candidate" ]; then
            NDK_ROOT="$(dirname "$candidate")"
            break
        fi
    done
fi

if [ ! -f "${NDK_ROOT}/build/cmake/android.toolchain.cmake" ]; then
    echo "ERROR: Cannot find NDK toolchain. Set ANDROID_NDK_HOME."
    exit 1
fi
echo "Using NDK: ${NDK_ROOT}"

TOOLCHAIN="${NDK_ROOT}/build/cmake/android.toolchain.cmake"

# ---------------------------------------------------------------
# Download source
# ---------------------------------------------------------------
SOURCE_TARBALL="/tmp/opencv-${VERSION}-source.tar.gz"
SOURCE_URL="https://github.com/opencv/opencv/archive/refs/tags/${VERSION}.tar.gz"

if [ -d "${SOURCE_DIR}/opencv-${VERSION}" ]; then
    echo "Source already exists at ${SOURCE_DIR}/opencv-${VERSION}"
else
    echo "Downloading OpenCV ${VERSION} source..."
    curl -L -o "${SOURCE_TARBALL}" "${SOURCE_URL}"
    mkdir -p "${SOURCE_DIR}"
    tar -xzf "${SOURCE_TARBALL}" -C "${SOURCE_DIR}"
    rm -f "${SOURCE_TARBALL}"
fi

# ---------------------------------------------------------------
# Configure
# ---------------------------------------------------------------
echo ""
echo "=== Configuring OpenCV ${VERSION} for ${ABI} with libc++ ==="
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

"${CMAKE}" \
    -S "${SOURCE_DIR}/opencv-${VERSION}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_ANDROID_EXAMPLES=OFF \
    -DBUILD_ANDROID_PROJECTS=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_FAT_JAVA_LIB=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_IPP=OFF \
    -DWITH_TBB=OFF \
    -DWITH_EIGEN=OFF \
    -DWITH_PROTOBUF=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DENABLE_NEON=ON \
    -DCMAKE_C_FLAGS="-Wno-error=implicit-function-declaration -Wno-error=int-conversion" \
    2>&1 | tail -30

# ---------------------------------------------------------------
# Build
# ---------------------------------------------------------------
echo ""
echo "=== Building OpenCV ${VERSION} (this may take a few minutes) ==="
"${CMAKE}" --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)" 2>&1 | tail -20

# ---------------------------------------------------------------
# Replace pre-built static libraries with libc++ versions
# ---------------------------------------------------------------
echo ""
echo "=== Replacing pre-built static libraries ==="

STATIC_DIR="${SDK_DIR}/sdk/native/staticlibs/${ABI}"
THIRDPARTY_DIR="${SDK_DIR}/sdk/native/3rdparty/libs/${ABI}"

# Backup originals
if [ ! -d "${STATIC_DIR}.gnustl-backup" ]; then
    cp -a "${STATIC_DIR}" "${STATIC_DIR}.gnustl-backup"
    cp -a "${THIRDPARTY_DIR}" "${THIRDPARTY_DIR}.gnustl-backup"
fi

# Clear stale pre-built libraries (modules disabled in source build
# like dnn/protobuf/tbb would otherwise linger as gnustl binaries)
rm -f "${STATIC_DIR}"/libopencv_*.a
rm -f "${THIRDPARTY_DIR}"/lib*.a

# Copy rebuilt OpenCV module libraries
for lib in "${BUILD_DIR}"/lib/${ABI}/*.a; do
    libname="$(basename "$lib")"
    cp -f "$lib" "${STATIC_DIR}/${libname}"
    echo "  Module:   ${libname}"
done

# Copy rebuilt 3rdparty libraries
for lib in "${BUILD_DIR}"/3rdparty/lib/${ABI}/*.a; do
    [ -f "$lib" ] || continue
    libname="$(basename "$lib")"
    cp -f "$lib" "${THIRDPARTY_DIR}/${libname}"
    echo "  3rdparty: ${libname}"
done

# ---------------------------------------------------------------
# Update the arch-specific .mk file to match our build
# ---------------------------------------------------------------
echo ""
echo "=== Updating OpenCV-${ABI}.mk ==="

# Collect 3rdparty component names (strip lib prefix and .a suffix)
COMPONENTS=""
for lib in "${THIRDPARTY_DIR}"/lib*.a; do
    [ -f "$lib" ] || continue
    name="$(basename "$lib" .a)"
    name="${name#lib}"
    COMPONENTS="${COMPONENTS} ${name}"
done
COMPONENTS="$(echo "${COMPONENTS}" | xargs)"

ARCH_MK="${SDK_DIR}/sdk/native/jni/OpenCV-${ABI}.mk"
cat > "${ARCH_MK}" <<EOF
OPENCV_3RDPARTY_COMPONENTS:=${COMPONENTS}
OPENCV_EXTRA_COMPONENTS:=z dl m log
EOF

echo "  Written: ${ARCH_MK}"
cat "${ARCH_MK}"

# ---------------------------------------------------------------
# Update OpenCV.mk module list to match what we built
# ---------------------------------------------------------------
MODULES=""
for lib in "${STATIC_DIR}"/libopencv_*.a; do
    [ -f "$lib" ] || continue
    name="$(basename "$lib" .a)"
    name="${name#libopencv_}"
    MODULES="${MODULES} ${name}"
done
MODULES="$(echo "${MODULES}" | xargs)"

# Patch the OPENCV_MODULES line in OpenCV.mk
sed -i "s/^OPENCV_MODULES:=.*/OPENCV_MODULES:=${MODULES}/" "${SDK_DIR}/sdk/native/jni/OpenCV.mk"
echo "  Updated OPENCV_MODULES: ${MODULES}"

# ---------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------
echo ""
echo "=== Cleanup ==="
rm -rf "${BUILD_DIR}"
echo "Build dir removed."
echo ""
echo "OpenCV ${VERSION} rebuilt with libc++ successfully!"
echo "SDK location: ${SDK_DIR}"
