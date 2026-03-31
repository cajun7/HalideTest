#pragma once

#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "HalideBuffer.h"
#include <opencv2/core.hpp>
#include <chrono>

#define LOG_TAG "HalideBenchmark"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using Clock = std::chrono::high_resolution_clock;

// RAII wrapper for Android Bitmap pixel lock.
// Provides zero-copy access as both Halide Buffer and OpenCV Mat.
struct BitmapLock {
    JNIEnv* env;
    jobject bitmap;
    AndroidBitmapInfo info;
    void* pixels;
    bool locked;

    BitmapLock(JNIEnv* e, jobject bmp) : env(e), bitmap(bmp), pixels(nullptr), locked(false) {
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOGE("Failed to get bitmap info");
            return;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOGE("Failed to lock bitmap pixels");
            return;
        }
        locked = true;
    }

    ~BitmapLock() {
        if (locked) {
            AndroidBitmap_unlockPixels(env, bitmap);
        }
    }

    bool is_valid() const { return locked && pixels != nullptr; }
    int width() const { return info.width; }
    int height() const { return info.height; }

    // Wrap as Halide RGBA buffer (Android ARGB_8888 = RGBA in memory on little-endian ARM)
    Halide::Runtime::Buffer<uint8_t> as_halide_rgba() {
        return Halide::Runtime::Buffer<uint8_t>::make_interleaved(
            (uint8_t*)pixels, info.width, info.height, 4);
    }

    // Wrap as Halide RGB buffer (first 3 channels only, stride=4)
    // NOTE: This creates a view with stride 4 on the innermost dimension.
    // For most operations, prefer converting to contiguous RGB first.
    Halide::Runtime::Buffer<uint8_t> as_halide_rgb_view() {
        halide_dimension_t dims[3] = {
            {0, (int)info.width, 4},                     // x, stride=4 (RGBA interleaved)
            {0, (int)info.height, (int)(info.stride)},   // y, stride=row_stride
            {0, 3, 1},                                   // c (R,G,B), stride=1
        };
        return Halide::Runtime::Buffer<uint8_t>((uint8_t*)pixels, 3, dims);
    }

    // Wrap as OpenCV Mat (RGBA)
    cv::Mat as_opencv_rgba() {
        return cv::Mat(info.height, info.width, CV_8UC4, pixels, info.stride);
    }
};

// Extract contiguous RGB from RGBA bitmap (copies data)
inline void rgba_to_rgb(const uint8_t* rgba, uint8_t* rgb, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
}

// Write contiguous RGB back into RGBA bitmap (copies data, sets alpha=255)
inline void rgb_to_rgba(const uint8_t* rgb, uint8_t* rgba, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
}
