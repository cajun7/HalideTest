#!/bin/bash
# Download and extract OpenCV 4.9.0 Android SDK
# OpenCV 4.x is built with libc++, compatible with NDK 27+.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCV_DIR="${SCRIPT_DIR}/../opencv"
VERSION="4.9.0"
OPENCV_URL="https://github.com/opencv/opencv/releases/download/${VERSION}/opencv-${VERSION}-android-sdk.zip"
ZIPFILE="/tmp/opencv-${VERSION}-android-sdk.zip"

if [ -d "${OPENCV_DIR}/OpenCV-android-sdk" ]; then
    echo "OpenCV Android SDK already exists at ${OPENCV_DIR}/OpenCV-android-sdk"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading OpenCV ${VERSION} Android SDK..."
curl -L -o "${ZIPFILE}" "${OPENCV_URL}"

echo "Extracting to ${OPENCV_DIR}/..."
mkdir -p "${OPENCV_DIR}"
unzip -q "${ZIPFILE}" -d "${OPENCV_DIR}"

rm -f "${ZIPFILE}"

# Verify the expected structure
if [ -f "${OPENCV_DIR}/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk" ]; then
    echo "OpenCV ${VERSION} Android SDK setup complete."
    echo "SDK location: ${OPENCV_DIR}/OpenCV-android-sdk"
    echo "OpenCV.mk: ${OPENCV_DIR}/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk"
else
    echo "WARNING: OpenCV.mk not found at expected path. Check the extracted structure."
    ls -la "${OPENCV_DIR}/OpenCV-android-sdk/sdk/native/jni/" 2>/dev/null || echo "Directory not found"
fi
