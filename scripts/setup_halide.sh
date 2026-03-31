#!/bin/bash
# Download and extract Halide v18.0.0 pre-built release for macOS (arm64)
# The pre-built release bundles LLVM internally - no separate LLVM install needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HALIDE_DIR="${SCRIPT_DIR}/../halide"
VERSION="18.0.0"

# Detect host architecture
ARCH=$(uname -m)
if [ "${ARCH}" = "arm64" ]; then
    HALIDE_URL="https://github.com/halide/Halide/releases/download/v${VERSION}/Halide-${VERSION}-arm-64-osx-8c651b459a4e3744b413c23a29b5c5d968702bb7.tar.gz"
else
    HALIDE_URL="https://github.com/halide/Halide/releases/download/v${VERSION}/Halide-${VERSION}-x86-64-osx-8c651b459a4e3744b413c23a29b5c5d968702bb7.tar.gz"
fi

TARBALL="/tmp/halide-${VERSION}.tar.gz"

if [ -d "${HALIDE_DIR}/Halide-${VERSION}" ]; then
    echo "Halide ${VERSION} already exists at ${HALIDE_DIR}/Halide-${VERSION}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading Halide ${VERSION}..."
curl -L -o "${TARBALL}" "${HALIDE_URL}"

echo "Extracting to ${HALIDE_DIR}/Halide-${VERSION}..."
mkdir -p "${HALIDE_DIR}/Halide-${VERSION}"
tar xzf "${TARBALL}" -C "${HALIDE_DIR}/Halide-${VERSION}" --strip-components=1

# Copy GenGen.cpp to tools directory
echo "Copying GenGen.cpp to ${HALIDE_DIR}/tools/..."
cp "${HALIDE_DIR}/Halide-${VERSION}/share/Halide/tools/GenGen.cpp" "${HALIDE_DIR}/tools/GenGen.cpp"

rm -f "${TARBALL}"

echo "Halide ${VERSION} setup complete."
echo "SDK location: ${HALIDE_DIR}/Halide-${VERSION}"
echo "GenGen.cpp: ${HALIDE_DIR}/tools/GenGen.cpp"
