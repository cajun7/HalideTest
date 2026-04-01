#!/bin/bash
# Download and extract OpenCV Android SDK.
# Supports multiple versions side-by-side via OPENCV_VERSION env var.
#
# Usage:
#   bash scripts/setup_opencv.sh                          # 4.9.0 (default)
#   OPENCV_VERSION=3.4.16 bash scripts/setup_opencv.sh    # 3.4.16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCV_DIR="${SCRIPT_DIR}/../opencv"
VERSION="${OPENCV_VERSION:-4.9.0}"
OPENCV_URL="https://github.com/opencv/opencv/releases/download/${VERSION}/opencv-${VERSION}-android-sdk.zip"
ZIPFILE="/tmp/opencv-${VERSION}-android-sdk.zip"
INSTALL_DIR="${OPENCV_DIR}/${VERSION}"

if [ -d "${INSTALL_DIR}/OpenCV-android-sdk" ]; then
    echo "OpenCV ${VERSION} Android SDK already exists at ${INSTALL_DIR}/OpenCV-android-sdk"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading OpenCV ${VERSION} Android SDK..."
curl -L -o "${ZIPFILE}" "${OPENCV_URL}"

echo "Extracting to ${INSTALL_DIR}/..."
mkdir -p "${INSTALL_DIR}"
unzip -q "${ZIPFILE}" -d "${INSTALL_DIR}"

rm -f "${ZIPFILE}"

# Verify the expected structure
if [ -f "${INSTALL_DIR}/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk" ]; then
    echo "OpenCV ${VERSION} Android SDK setup complete."
    echo "SDK location: ${INSTALL_DIR}/OpenCV-android-sdk"
    echo "OpenCV.mk: ${INSTALL_DIR}/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk"
else
    echo "WARNING: OpenCV.mk not found at expected path. Check the extracted structure."
    ls -la "${INSTALL_DIR}/OpenCV-android-sdk/sdk/native/jni/" 2>/dev/null || echo "Directory not found"
fi
