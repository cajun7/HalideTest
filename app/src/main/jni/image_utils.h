#pragma once

#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "HalideBuffer.h"
#include <opencv2/core.hpp>
#include <chrono>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

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

// Extract contiguous RGB from RGBA bitmap.
// NEON path: 16 pixels/iteration using vld4q_u8 (deinterleave 4ch) + vst3q_u8 (interleave 3ch).
inline void rgba_to_rgb(const uint8_t* rgba, uint8_t* rgb, int width, int height) {
    int total = width * height;
    int i = 0;
#if HAS_NEON
    // Process 16 RGBA pixels -> 16 RGB pixels per iteration
    for (; i + 16 <= total; i += 16) {
        uint8x16x4_t src = vld4q_u8(rgba + i * 4);  // deinterleave R,G,B,A
        uint8x16x3_t dst;
        dst.val[0] = src.val[0];  // R
        dst.val[1] = src.val[1];  // G
        dst.val[2] = src.val[2];  // B (discard A)
        vst3q_u8(rgb + i * 3, dst);  // interleave R,G,B
    }
#endif
    // Scalar tail
    for (; i < total; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
}

// Write contiguous RGB back into RGBA bitmap (sets alpha=255).
// NEON path: 16 pixels/iteration using vld3q_u8 + vst4q_u8.
inline void rgb_to_rgba(const uint8_t* rgb, uint8_t* rgba, int width, int height) {
    int total = width * height;
    int i = 0;
#if HAS_NEON
    uint8x16_t alpha = vdupq_n_u8(255);
    for (; i + 16 <= total; i += 16) {
        uint8x16x3_t src = vld3q_u8(rgb + i * 3);  // deinterleave R,G,B
        uint8x16x4_t dst;
        dst.val[0] = src.val[0];  // R
        dst.val[1] = src.val[1];  // G
        dst.val[2] = src.val[2];  // B
        dst.val[3] = alpha;       // A = 255
        vst4q_u8(rgba + i * 4, dst);  // interleave R,G,B,A
    }
#endif
    // Scalar tail
    for (; i < total; i++) {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
}

// NEON-optimized interleaved RGB -> planar RGB conversion.
// Used by operations that require planar layout (e.g., rotate_arbitrary).
inline void interleaved_to_planar(const uint8_t* interleaved, uint8_t* planar,
                                  int width, int height) {
    int total = width * height;
    uint8_t* p0 = planar;                    // R plane
    uint8_t* p1 = planar + total;            // G plane
    uint8_t* p2 = planar + total * 2;        // B plane
    int i = 0;
#if HAS_NEON
    for (; i + 16 <= total; i += 16) {
        uint8x16x3_t src = vld3q_u8(interleaved + i * 3);
        vst1q_u8(p0 + i, src.val[0]);
        vst1q_u8(p1 + i, src.val[1]);
        vst1q_u8(p2 + i, src.val[2]);
    }
#endif
    for (; i < total; i++) {
        p0[i] = interleaved[i * 3 + 0];
        p1[i] = interleaved[i * 3 + 1];
        p2[i] = interleaved[i * 3 + 2];
    }
}

// NEON-optimized planar RGB -> interleaved RGB conversion.
inline void planar_to_interleaved(const uint8_t* planar, uint8_t* interleaved,
                                  int width, int height) {
    int total = width * height;
    const uint8_t* p0 = planar;
    const uint8_t* p1 = planar + total;
    const uint8_t* p2 = planar + total * 2;
    int i = 0;
#if HAS_NEON
    for (; i + 16 <= total; i += 16) {
        uint8x16x3_t dst;
        dst.val[0] = vld1q_u8(p0 + i);
        dst.val[1] = vld1q_u8(p1 + i);
        dst.val[2] = vld1q_u8(p2 + i);
        vst3q_u8(interleaved + i * 3, dst);
    }
#endif
    for (; i < total; i++) {
        interleaved[i * 3 + 0] = p0[i];
        interleaved[i * 3 + 1] = p1[i];
        interleaved[i * 3 + 2] = p2[i];
    }
}
