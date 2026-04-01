#!/bin/bash
# Download and extract Halide v21.0.0 pre-built release for macOS or Linux.
# The pre-built release bundles LLVM internally - no separate LLVM install needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HALIDE_DIR="${SCRIPT_DIR}/../halide"
VERSION="21.0.0"
COMMIT_HASH="b629c80de18f1534ec71fddd8b567aa7027a0876"

# Detect host OS
OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Darwin) HL_OS="osx" ;;
    Linux)  HL_OS="linux" ;;
    *)
        echo "ERROR: Unsupported OS: ${OS_NAME}. Only macOS and Linux are supported."
        exit 1
        ;;
esac

# Detect host architecture
ARCH=$(uname -m)
case "${ARCH}" in
    arm64|aarch64) HL_ARCH="arm-64" ;;
    x86_64)        HL_ARCH="x86-64" ;;
    *)
        echo "ERROR: Unsupported architecture: ${ARCH}."
        exit 1
        ;;
esac

HALIDE_URL="https://github.com/halide/Halide/releases/download/v${VERSION}/Halide-${VERSION}-${HL_ARCH}-${HL_OS}-${COMMIT_HASH}.tar.gz"
TARBALL="/tmp/halide-${VERSION}.tar.gz"

if [ -d "${HALIDE_DIR}/Halide-${VERSION}" ]; then
    echo "Halide ${VERSION} already exists at ${HALIDE_DIR}/Halide-${VERSION}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading Halide ${VERSION} (${HL_ARCH}-${HL_OS})..."
curl -L -o "${TARBALL}" "${HALIDE_URL}"

echo "Extracting to ${HALIDE_DIR}/Halide-${VERSION}..."
mkdir -p "${HALIDE_DIR}/Halide-${VERSION}"
tar xzf "${TARBALL}" -C "${HALIDE_DIR}/Halide-${VERSION}" --strip-components=1

# v21+ ships libGenGen.a instead of GenGen.cpp
mkdir -p "${HALIDE_DIR}/tools"
GENGEN_LIB="${HALIDE_DIR}/Halide-${VERSION}/lib/libHalide_GenGen.a"
GENGEN_CPP="${HALIDE_DIR}/Halide-${VERSION}/share/Halide/tools/GenGen.cpp"
if [ -f "${GENGEN_LIB}" ]; then
    echo "Found libGenGen.a (v19+ style) -- no GenGen.cpp copy needed."
elif [ -f "${GENGEN_CPP}" ]; then
    echo "Copying GenGen.cpp to ${HALIDE_DIR}/tools/ (legacy style)..."
    cp "${GENGEN_CPP}" "${HALIDE_DIR}/tools/GenGen.cpp"
else
    echo "WARNING: Neither libGenGen.a nor GenGen.cpp found in SDK."
fi

rm -f "${TARBALL}"

echo "Halide ${VERSION} setup complete."
echo "SDK location: ${HALIDE_DIR}/Halide-${VERSION}"
echo "GenGen.cpp: ${HALIDE_DIR}/tools/GenGen.cpp"
