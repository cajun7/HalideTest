#include <jni.h>
#include <android/bitmap.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

#include "image_utils.h"
#include "halide_ops.h"
#include "opencv_ops.h"
#include "benchmark_engine.h"
#include "operation_context.h"
#include "operation_registry.h"
#include "bt709_neon_ref.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using Clock = std::chrono::high_resolution_clock;

// Helper: extract RGB from RGBA bitmap, run Halide op, write back
// Many Halide generators expect 3-channel interleaved RGB, but Android Bitmap is RGBA.

extern "C" {

// -----------------------------------------------------------------------
// RGB <-> BGR conversion
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_rgbBgr(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        // Extract RGB, process, write back RGBA
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::rgb_bgr(ibuf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb, out_bgr;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        opencv_ops::rgb_bgr(in_rgb, out_bgr);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 -> RGB conversion
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ToRgb(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint width, jint height,
    jobject outputBitmap, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    BitmapLock out_lock(env, outputBitmap);
    if (!out_lock.is_valid()) {
        env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
        return -1;
    }

    int w = width, h = height;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + w * h;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
        // UV plane as 2D raw bytes: width x (height/2)
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);

        // Interleaved RGB output — generator now has stride constraints
        std::vector<uint8_t> rgb_out(w * h * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::nv21_to_rgb(y_buf, uv_buf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_to_rgb(nv21_mat, rgb_out);
        cv::Mat rgba_out;
        cv::cvtColor(rgb_out, rgba_out, cv::COLOR_RGB2RGBA);
        rgba_out.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Gaussian Blur
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_gaussianBlur(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint kernelSize, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::gaussian_blur_rgb(ibuf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::gaussian_blur_rgb(in_bgr, out_bgr, kernelSize);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Lens Blur
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_lensBlur(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint radius, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::lens_blur(ibuf, radius, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::lens_blur(in_bgr, out_bgr, radius);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Resize
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resize(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint newWidth, jint newHeight, jboolean useBicubic, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int ow = newWidth, oh = newHeight;
    float sx = (float)ow / w;
    float sy = (float)oh / h;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(ow * oh * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ow, oh, 3);

        if (useBicubic) {
            halide_ops::resize_bicubic(ibuf, sx, sy, obuf);
        } else {
            halide_ops::resize_bilinear(ibuf, sx, sy, obuf);
        }

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, ow, oh);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);

        if (useBicubic) {
            opencv_ops::resize_bicubic(in_bgr, out_bgr, ow, oh);
        } else {
            opencv_ops::resize_bilinear(in_bgr, out_bgr, ow, oh);
        }

        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Rotate
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_rotate(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jfloat angleDegrees, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int ow = out_lock.width(), oh = out_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(ow * oh * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        float angle_rad = angleDegrees * 3.14159265358979f / 180.0f;

        // Use fixed rotation for exact multiples of 90
        if (angleDegrees == 90.0f || angleDegrees == 180.0f || angleDegrees == 270.0f) {
            auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
            auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ow, oh, 3);
            if (angleDegrees == 90.0f) halide_ops::rotate_90cw(ibuf, obuf);
            else if (angleDegrees == 180.0f) halide_ops::rotate_180(ibuf, obuf);
            else halide_ops::rotate_270cw(ibuf, obuf);
        } else {
            // rotate_arbitrary uses planar buffers (constant_exterior + interleaved has issues)
            std::vector<uint8_t> planar_in(w * h * 3);
            interleaved_to_planar(rgb_in.data(), planar_in.data(), w, h);
            Halide::Runtime::Buffer<uint8_t> ibuf(planar_in.data(), w, h, 3);

            Halide::Runtime::Buffer<uint8_t> obuf(ow, oh, 3);
            halide_ops::rotate_angle(ibuf, angle_rad, obuf);

            planar_to_interleaved(obuf.data(), rgb_out.data(), ow, oh);
        }

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, ow, oh);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);

        if (angleDegrees == 90.0f) {
            opencv_ops::rotate_90(in_bgr, out_bgr);
        } else if (angleDegrees == 180.0f) {
            opencv_ops::rotate_180(in_bgr, out_bgr);
        } else if (angleDegrees == 270.0f) {
            opencv_ops::rotate_270(in_bgr, out_bgr);
        } else {
            opencv_ops::rotate_angle(in_bgr, out_bgr, angleDegrees);
        }

        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB -> NV21 conversion
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_rgbToNv21(
    JNIEnv* env, jclass, jobject inputBitmap, jbyteArray nv21Output, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    if (!in_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int y_size = w * h;
    int uv_size = w * (h / 2);

    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Output, nullptr);

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        // Interleaved RGB input — generator now has stride constraints
        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);

        // Y output: w x h
        Halide::Runtime::Buffer<uint8_t> y_buf((uint8_t*)nv21_ptr, w, h);

        // UV output: raw bytes, width x (height/2)
        uint8_t* uv_ptr = (uint8_t*)nv21_ptr + y_size;
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);

        halide_ops::rgb_to_nv21(ibuf, y_buf, uv_buf);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat nv21_mat;
        opencv_ops::rgb_to_nv21(in_rgb, nv21_mat);
        // Copy NV21 data to output array
        memcpy(nv21_ptr, nv21_mat.data, y_size + uv_size);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Output, nv21_ptr, 0);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Resize INTER_AREA
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeArea(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint newWidth, jint newHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int ow = newWidth, oh = newHeight;
    float sx = (float)ow / w;
    float sy = (float)oh / h;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(ow * oh * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ow, oh, 3);

        halide_ops::resize_area(ibuf, sx, sy, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, ow, oh);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);

        opencv_ops::resize_area(in_bgr, out_bgr, ow, oh);

        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Resize Letterbox (aspect-ratio-preserving)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeLetterbox(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

        halide_ops::resize_letterbox(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);

        opencv_ops::resize_letterbox(in_bgr, out_bgr, tw, th);

        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Target-size Resize (Bilinear)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeBilinearTarget(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_bilinear_target(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_bilinear(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Target-size Resize (Bicubic)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeBicubicTarget(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_bicubic_target(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_bicubic(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Target-size Resize (INTER_AREA)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeAreaTarget(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_area_target(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_area(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Flip (Horizontal / Vertical)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_flip(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jboolean horizontal, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);

        if (horizontal) {
            halide_ops::flip_horizontal(ibuf, obuf);
        } else {
            halide_ops::flip_vertical(ibuf, obuf);
        }

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);

        if (horizontal) {
            opencv_ops::flip_horizontal(in_bgr, out_bgr);
        } else {
            opencv_ops::flip_vertical(in_bgr, out_bgr);
        }

        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Rotate -> [Flip] -> Resize -> RGB Pipeline
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21RotateResizeRgb(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint rotationDegreesCW, jint flipCode, jint targetWidth, jint targetHeight,
    jboolean useArea, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    // Allocate output RGBA buffer
    std::vector<uint8_t> rgba_out(tw * th * 4);

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        // Interleaved RGB output
        std::vector<uint8_t> rgb_out(tw * th * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

        if (useArea) {
            halide_ops::nv21_rotate_flip_resize_area_rgb(
                y_buf, uv_buf, rotationDegreesCW, flipCode, tw, th, obuf);
        } else {
            halide_ops::nv21_rotate_flip_resize_rgb(
                y_buf, uv_buf, rotationDegreesCW, flipCode, tw, th, obuf);
        }

        rgb_to_rgba(rgb_out.data(), rgba_out.data(), tw, th);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        int interp = useArea ? cv::INTER_AREA : cv::INTER_LINEAR;
        opencv_ops::nv21_rotate_flip_resize_rgb(
            nv21_mat, rgb_out, rotationDegreesCW, flipCode, tw, th, interp);
        cv::Mat rgba_mat;
        cv::cvtColor(rgb_out, rgba_mat, cv::COLOR_RGB2RGBA);
        memcpy(rgba_out.data(), rgba_mat.data, tw * th * 4);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 -> RGB with bilinear UV upsampling (YUV444)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21Yuv444Rgb(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint width, jint height,
    jobject outputBitmap, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    BitmapLock out_lock(env, outputBitmap);
    if (!out_lock.is_valid()) {
        env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
        return -1;
    }

    int w = width, h = height;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + w * h;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);
        std::vector<uint8_t> rgb_out(w * h * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::nv21_yuv444_rgb(y_buf, uv_buf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_yuv444_rgb(nv21_mat, rgb_out);
        cv::Mat rgba_out;
        cv::cvtColor(rgb_out, rgba_out, cv::COLOR_RGB2RGBA);
        rgba_out.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 -> RGB full-range BT.601
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ToRgbFullRange(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint width, jint height,
    jobject outputBitmap, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    BitmapLock out_lock(env, outputBitmap);
    if (!out_lock.is_valid()) {
        env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
        return -1;
    }

    int w = width, h = height;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + w * h;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);
        std::vector<uint8_t> rgb_out(w * h * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::nv21_to_rgb_full_range(y_buf, uv_buf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_to_rgb_full_range(nv21_mat, rgb_out);
        cv::Mat rgba_out;
        cv::cvtColor(rgb_out, rgba_out, cv::COLOR_RGB2RGBA);
        rgba_out.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Resize -> Pad -> Rotate (ML preprocessing)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizePadRotate(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint rotationDegreesCW, jint targetSize, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int ts = targetSize;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        std::vector<uint8_t> rgb_out(ts * ts * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ts, ts, 3);

        halide_ops::nv21_resize_pad_rotate(y_buf, uv_buf, rotationDegreesCW, ts, obuf);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_resize_pad_rotate(nv21_mat, rgb_out, rotationDegreesCW, ts);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Segmentation Argmax
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_segArgmax(
    JNIEnv* env, jclass, jint width, jint height, jint numClasses, jboolean useHalide)
{
    int w = width, h = height, nc = numClasses;
    int total = w * h * nc;

    // Create synthetic float input data (simulates ML model output)
    std::vector<float> input_data(total);
    for (int i = 0; i < total; i++) {
        input_data[i] = (float)(i % 256) / 255.0f;
    }

    auto start = Clock::now();

    if (useHalide) {
        // Planar float buffer: width x height x numClasses
        Halide::Runtime::Buffer<float> ibuf(input_data.data(), w, h, nc);
        Halide::Runtime::Buffer<uint8_t> obuf(w, h);
        halide_ops::seg_argmax(ibuf, obuf);
    } else {
        // OpenCV reference using same planar layout
        cv::Mat input_mat(h, w, CV_32FC1, input_data.data());
        cv::Mat output_mat;
        opencv_ops::seg_argmax(input_mat, output_mat, nc);
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB <-> BGR Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_rgbBgrOptimized(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::rgb_bgr_optimized(ibuf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb, out_bgr;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        opencv_ops::rgb_bgr_optimized(in_rgb, out_bgr);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 -> RGB Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ToRgbOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint width, jint height,
    jobject outputBitmap, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    BitmapLock out_lock(env, outputBitmap);
    if (!out_lock.is_valid()) {
        env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
        return -1;
    }

    int w = width, h = height;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + w * h;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);

        std::vector<uint8_t> rgb_out(w * h * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
        halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_to_rgb_optimized(nv21_mat, rgb_out);
        cv::Mat rgba_out;
        cv::cvtColor(rgb_out, rgba_out, cv::COLOR_RGB2RGBA);
        rgba_out.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB -> NV21 Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_rgbToNv21Optimized(
    JNIEnv* env, jclass, jobject inputBitmap, jbyteArray nv21Output, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    if (!in_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int y_size = w * h;
    int uv_size = w * (h / 2);

    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Output, nullptr);

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        // Interleaved RGB input — generator now has stride constraints
        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);

        Halide::Runtime::Buffer<uint8_t> y_buf((uint8_t*)nv21_ptr, w, h);
        uint8_t* uv_ptr = (uint8_t*)nv21_ptr + y_size;
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);

        halide_ops::rgb_to_nv21_optimized(ibuf, y_buf, uv_buf);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat nv21_mat;
        opencv_ops::rgb_to_nv21_optimized(in_rgb, nv21_mat);
        memcpy(nv21_ptr, nv21_mat.data, y_size + uv_size);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Output, nv21_ptr, 0);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB Resize Bilinear Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeBilinearOptimized(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_bilinear_optimized(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_bilinear_optimized(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB Resize Area Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeAreaOptimized(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_area_optimized(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_area_optimized(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// RGB Resize Bicubic Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_resizeBicubicOptimized(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::resize_bicubic_optimized(ibuf, tw, th, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, tw, th);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_bgr, out_bgr;
        cv::cvtColor(in_rgba, in_bgr, cv::COLOR_RGBA2BGR);
        opencv_ops::resize_bicubic_optimized(in_bgr, out_bgr, tw, th);
        cv::Mat out_rgba;
        cv::cvtColor(out_bgr, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 Resize Bilinear Optimized (output stays NV21)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeBilinearOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
        Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
        halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat nv21_out;
        opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_LINEAR);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 Resize Area Optimized (output stays NV21)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeAreaOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
        Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
        halide_ops::nv21_resize_area_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat nv21_out;
        opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_AREA);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// NV21 Resize Bicubic Optimized (output stays NV21)
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeBicubicOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
        Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
        halide_ops::nv21_resize_bicubic_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat nv21_out;
        opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_CUBIC);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Resize -> RGB Bilinear Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbBilinearOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        std::vector<uint8_t> rgb_out(tw * th * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::nv21_resize_rgb_bilinear_optimized(y_buf, uv_buf, tw, th, obuf);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_LINEAR);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Resize -> RGB Area Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbAreaOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        std::vector<uint8_t> rgb_out(tw * th * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::nv21_resize_rgb_area_optimized(y_buf, uv_buf, tw, th, obuf);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_AREA);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Resize -> RGB Bicubic Optimized
// -----------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbBicubicOptimized(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight;
    int tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    if (useHalide) {
        uint8_t* y_ptr = (uint8_t*)nv21_ptr;
        uint8_t* uv_ptr = y_ptr + sw * sh;

        Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);

        std::vector<uint8_t> rgb_out(tw * th * 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
        halide_ops::nv21_resize_rgb_bicubic_optimized(y_buf, uv_buf, tw, th, obuf);
    } else {
        cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, (uint8_t*)nv21_ptr);
        cv::Mat rgb_out;
        opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_CUBIC);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Fused NV21 -> Resize -> RGB (BT.709 full-range) — 3 interp variants
// -----------------------------------------------------------------------
// Halide path: the fused nv21_resize_rgb_bt709_{interp} generator
// Baseline  : Halide NV21-domain resize + NEON BT.709 scalar/vector ref
//             (OpenCV has no BT.709 full-range NV21 op, so this is the
//              apples-to-apples "no fusion" comparison — matches
//              bench_main.cpp:272-279).

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbBt709Nearest(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight, tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    uint8_t* y_ptr  = (uint8_t*)nv21_ptr;
    uint8_t* uv_ptr = y_ptr + sw * sh;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);
    std::vector<uint8_t> rgb_out(tw * th * 3);
    auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

    if (useHalide) {
        halide_ops::nv21_resize_rgb_bt709_nearest(y_buf, uv_buf, tw, th, obuf);
    } else {
        Halide::Runtime::Buffer<uint8_t> y_r(tw, th), uv_r(tw, th / 2);
        halide_ops::nv21_resize_nearest_optimized(y_buf, uv_buf, tw, th, y_r, uv_r);
        bt709::nv21_to_rgb_bt709_full_range_neon(
            y_r.data(),  tw,
            uv_r.data(), tw,
            rgb_out.data(), tw * 3, tw, th);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbBt709Bilinear(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight, tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    uint8_t* y_ptr  = (uint8_t*)nv21_ptr;
    uint8_t* uv_ptr = y_ptr + sw * sh;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);
    std::vector<uint8_t> rgb_out(tw * th * 3);
    auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

    if (useHalide) {
        halide_ops::nv21_resize_rgb_bt709_bilinear(y_buf, uv_buf, tw, th, obuf);
    } else {
        Halide::Runtime::Buffer<uint8_t> y_r(tw, th), uv_r(tw, th / 2);
        halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, tw, th, y_r, uv_r);
        bt709::nv21_to_rgb_bt709_full_range_neon(
            y_r.data(),  tw,
            uv_r.data(), tw,
            rgb_out.data(), tw * 3, tw, th);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nv21ResizeRgbBt709Area(
    JNIEnv* env, jclass, jbyteArray nv21Data, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    jbyte* nv21_ptr = env->GetByteArrayElements(nv21Data, nullptr);
    int sw = srcWidth, sh = srcHeight, tw = targetWidth, th = targetHeight;

    auto start = Clock::now();

    uint8_t* y_ptr  = (uint8_t*)nv21_ptr;
    uint8_t* uv_ptr = y_ptr + sw * sh;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, sw, sh / 2);
    std::vector<uint8_t> rgb_out(tw * th * 3);
    auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

    if (useHalide) {
        halide_ops::nv21_resize_rgb_bt709_area(y_buf, uv_buf, tw, th, obuf);
    } else {
        Halide::Runtime::Buffer<uint8_t> y_r(tw, th), uv_r(tw, th / 2);
        halide_ops::nv21_resize_area_optimized(y_buf, uv_buf, tw, th, y_r, uv_r);
        bt709::nv21_to_rgb_bt709_full_range_neon(
            y_r.data(),  tw,
            uv_r.data(), tw,
            rgb_out.data(), tw * 3, tw, th);
    }

    auto end = Clock::now();
    env->ReleaseByteArrayElements(nv21Data, nv21_ptr, JNI_ABORT);
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Segmentation-Guided Pipelines
// -----------------------------------------------------------------------

// Helper: create a synthetic seg mask (centered rectangle as foreground)
static void make_seg_mask_data(uint8_t* mask, int mw, int mh, int fg_class) {
    int x0 = mw * 3 / 10, x1 = mw * 7 / 10;
    int y0 = mh * 3 / 10, y1 = mh * 7 / 10;
    for (int y = 0; y < mh; y++)
        for (int x = 0; x < mw; x++)
            mask[y * mw + x] = (x >= x0 && x < x1 && y >= y0 && y < y1)
                                ? (uint8_t)fg_class : (uint8_t)0;
}

// Helper: create a striped seg mask (alternating classes)
static void make_striped_mask_data(uint8_t* mask, int mw, int mh, int num_classes) {
    int stripe_w = std::max(1, mw / num_classes);
    for (int y = 0; y < mh; y++)
        for (int x = 0; x < mw; x++)
            mask[y * mw + x] = (uint8_t)std::min(x / stripe_w, num_classes - 1);
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_segPortraitBlur(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint blurRadius, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int mw = 256, mh = 256;
    int fg_class = 1;
    float edge_softness = 3.0f;

    // Create synthetic seg mask
    std::vector<uint8_t> mask_data(mw * mh);
    make_seg_mask_data(mask_data.data(), mw, mh, fg_class);

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);

        halide_ops::seg_portrait_blur(ibuf, mbuf, fg_class, blurRadius, edge_softness, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat mask_cv(mh, mw, CV_8UC1, mask_data.data());
        cv::Mat out_rgb;

        opencv_ops::seg_portrait_blur(in_rgb, mask_cv, fg_class, blurRadius,
                                      edge_softness, out_rgb);

        cv::Mat out_rgba;
        cv::cvtColor(out_rgb, out_rgba, cv::COLOR_RGB2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_segBgReplace(
    JNIEnv* env, jclass, jobject inputBitmap, jobject bgBitmap,
    jobject outputBitmap, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock bg_lock(env, bgBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !bg_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int bw = bg_lock.width(), bh = bg_lock.height();
    int mw = 256, mh = 256;
    int fg_class = 1;
    float edge_softness = 3.0f;

    std::vector<uint8_t> mask_data(mw * mh);
    make_seg_mask_data(mask_data.data(), mw, mh, fg_class);

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> fg_rgb(w * h * 3), bg_rgb(bw * bh * 3), out_rgb(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, fg_rgb.data(), w, h);
        rgba_to_rgb((uint8_t*)bg_lock.pixels, bg_rgb.data(), bw, bh);

        auto fg_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(fg_rgb.data(), w, h, 3);
        auto bg_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(bg_rgb.data(), bw, bh, 3);
        Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_rgb.data(), w, h, 3);

        halide_ops::seg_bg_replace(fg_buf, bg_buf, mbuf, fg_class, edge_softness, obuf);

        rgb_to_rgba(out_rgb.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat bg_rgba = bg_lock.as_opencv_rgba();
        cv::Mat in_rgb, bg_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::cvtColor(bg_rgba, bg_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat mask_cv(mh, mw, CV_8UC1, mask_data.data());
        cv::Mat out_rgb;

        opencv_ops::seg_bg_replace(in_rgb, bg_rgb, mask_cv, fg_class,
                                   edge_softness, out_rgb);

        cv::Mat out_rgba;
        cv::cvtColor(out_rgb, out_rgba, cv::COLOR_RGB2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_segColorStyle(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int mw = 256, mh = 256;
    int num_classes = 8;

    // Create striped mask
    std::vector<uint8_t> mask_data(mw * mh);
    make_striped_mask_data(mask_data.data(), mw, mh, num_classes);

    // Create styled LUT: [R_gain, G_gain, B_gain, R_bias, G_bias, B_bias, blend_alpha] per class
    std::vector<float> lut_data(num_classes * 7);
    // Class 0: desaturate (darken)
    float lut_values[] = {
        0.3f, 0.3f, 0.3f, 0.0f, 0.0f, 0.0f, 0.8f,  // 0: background
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // 1: person (unchanged)
        0.8f, 0.9f, 1.2f, 0.0f, 0.0f, 20.0f, 0.9f, // 2: sky (enhance blue)
        0.9f, 1.1f, 0.8f, 0.0f, 10.0f, 0.0f, 0.9f, // 3: vegetation (enhance green)
        1.1f, 1.0f, 0.9f, 12.0f, 0.0f, 0.0f, 0.7f, // 4: warm
        0.9f, 1.0f, 1.1f, 0.0f, 0.0f, 8.0f, 0.7f,  // 5: cool
        1.0f, 1.0f, 1.0f, 15.0f, 0.0f, 10.0f, 0.7f,// 6: magenta tint
        0.8f, 0.8f, 0.8f, 20.0f, 20.0f, 20.0f, 0.7f // 7: gray
    };
    std::copy(lut_values, lut_values + num_classes * 7, lut_data.begin());

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
        Halide::Runtime::Buffer<float> lbuf(lut_data.data(), num_classes, 7);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);

        halide_ops::seg_color_style(ibuf, mbuf, lbuf, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat mask_cv(mh, mw, CV_8UC1, mask_data.data());
        cv::Mat out_rgb;

        opencv_ops::seg_color_style(in_rgb, mask_cv, lut_data, num_classes, out_rgb);

        cv::Mat out_rgba;
        cv::cvtColor(out_rgb, out_rgba, cv::COLOR_RGB2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Native-only benchmark (malloc, no Java Bitmap — supports 200MP+)
// -----------------------------------------------------------------------
// Generates test data in native heap, runs Halide/OpenCV operation, frees.
// opId mapping matches the Java-side string array order.
static void fill_test_rgb(uint8_t* rgb, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = (y * w + x) * 3;
            rgb[i + 0] = (uint8_t)((x * 255) / (w > 1 ? w - 1 : 1));
            rgb[i + 1] = (uint8_t)((y * 255) / (h > 1 ? h - 1 : 1));
            rgb[i + 2] = (uint8_t)(((x + y) * 255) / (w + h > 2 ? w + h - 2 : 1));
        }
    }
}

static void fill_test_nv21(uint8_t* nv21, int w, int h) {
    int y_size = w * h;
    int uv_size = w * (h / 2);
    for (int i = 0; i < y_size; i++) nv21[i] = (uint8_t)(i % 256);
    for (int i = 0; i < uv_size; i++) nv21[y_size + i] = (uint8_t)(128 + (i % 64));
}

// Forward declaration (defined in segDepthBlur JNI section below)
static void make_depth_map_data(uint8_t* data, int w, int h);

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_nativeBenchmark(
    JNIEnv* env, jclass, jint opId, jint srcWidth, jint srcHeight,
    jint targetWidth, jint targetHeight, jboolean useHalide)
{
    int w = srcWidth, h = srcHeight;
    int tw = targetWidth, th = targetHeight;
    float sx = (float)tw / w, sy = (float)th / h;

    auto start = Clock::now();

    switch (opId) {
        case 0: { // RGB BGR
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::rgb_bgr_optimized(ibuf, obuf);
            } else {
                cv::Mat in_rgb(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::rgb_bgr_optimized(in_rgb, out_bgr);
            }
            break;
        }
        case 1: { // NV21 to RGB
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(w * h * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_to_rgb_optimized(nv21_mat, rgb_out);
            }
            break;
        }
        case 2: { // RGB to NV21
            std::vector<uint8_t> rgb_in(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(w, h / 2);
                halide_ops::rgb_to_nv21_optimized(ibuf, y_buf, uv_buf);
            } else {
                cv::Mat in_rgb(h, w, CV_8UC3, rgb_in.data());
                cv::Mat nv21_out;
                opencv_ops::rgb_to_nv21_optimized(in_rgb, nv21_out);
            }
            break;
        }
        case 3: { // NV21 YUV444 RGB (bilinear UV)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(w * h * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::nv21_yuv444_rgb(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_yuv444_rgb(nv21_mat, rgb_out);
            }
            break;
        }
        case 4: { // NV21 to RGB Full-Range
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(w * h * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::nv21_to_rgb_full_range(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_to_rgb_full_range(nv21_mat, rgb_out);
            }
            break;
        }
        case 5: { // Gaussian Blur (5x5)
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::gaussian_blur_rgb(ibuf, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::gaussian_blur_rgb(in_bgr, out_bgr, 5);
            }
            break;
        }
        case 6: { // Lens Blur (r=4)
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::lens_blur(ibuf, 4, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::lens_blur(in_bgr, out_bgr, 4);
            }
            break;
        }
        case 7: { // Resize Bilinear
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bilinear_optimized(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bilinear_optimized(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 8: { // Resize Bicubic
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bicubic_optimized(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bicubic_optimized(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 9: { // Resize Area
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_area_optimized(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_area_optimized(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 10: { // Resize Letterbox (720p)
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_letterbox(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_letterbox(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 11: { // Rotate 90
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(h * w * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), h, w, 3);
                halide_ops::rotate_90cw(ibuf, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::rotate_90(in_bgr, out_bgr);
            }
            break;
        }
        case 12: { // Rotate Arbitrary (45)
            std::vector<uint8_t> rgb_in(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                float angle_rad = 45.0f * 3.14159265358979f / 180.0f;
                std::vector<uint8_t> planar_in(w * h * 3);
                interleaved_to_planar(rgb_in.data(), planar_in.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> ibuf(planar_in.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                halide_ops::rotate_angle(ibuf, angle_rad, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::rotate_angle(in_bgr, out_bgr, 45.0f);
            }
            break;
        }
        case 13: { // Flip
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::flip_horizontal(ibuf, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::flip_horizontal(in_bgr, out_bgr);
            }
            break;
        }
        case 14: { // NV21 Pipeline (rotate+resize)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(tw * th * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::nv21_rotate_flip_resize_rgb(y_buf, uv_buf, 90, 0, tw, th, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_rotate_flip_resize_rgb(nv21_mat, rgb_out, 90, 0, tw, th, cv::INTER_LINEAR);
            }
            break;
        }
        case 15: { // NV21 Resize+Pad+Rotate (384)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            int ts = 384;
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(ts * ts * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ts, ts, 3);
                halide_ops::nv21_resize_pad_rotate(y_buf, uv_buf, 90, ts, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_resize_pad_rotate(nv21_mat, rgb_out, 90, ts);
            }
            break;
        }
        case 16: { // NV21 Resize (stay NV21)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
                Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
                halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat nv21_out;
                opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_LINEAR);
            }
            break;
        }
        case 17: { // NV21 Resize+RGB (fused)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(tw * th * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::nv21_resize_rgb_bilinear_optimized(y_buf, uv_buf, tw, th, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_LINEAR);
            }
            break;
        }
        case 18:   // NV21 Resize+RGB BT.709 Nearest  (fused)
        case 19:   // NV21 Resize+RGB BT.709 Bilinear (fused)
        case 20: { // NV21 Resize+RGB BT.709 Area     (fused)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
            Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
            std::vector<uint8_t> rgb_out(tw * th * 3);
            auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);

            if (useHalide) {
                switch (opId) {
                    case 18: halide_ops::nv21_resize_rgb_bt709_nearest (y_buf, uv_buf, tw, th, obuf); break;
                    case 19: halide_ops::nv21_resize_rgb_bt709_bilinear(y_buf, uv_buf, tw, th, obuf); break;
                    case 20: halide_ops::nv21_resize_rgb_bt709_area    (y_buf, uv_buf, tw, th, obuf); break;
                }
            } else {
                Halide::Runtime::Buffer<uint8_t> y_r(tw, th), uv_r(tw, th / 2);
                switch (opId) {
                    case 18: halide_ops::nv21_resize_nearest_optimized (y_buf, uv_buf, tw, th, y_r, uv_r); break;
                    case 19: halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, tw, th, y_r, uv_r); break;
                    case 20: halide_ops::nv21_resize_area_optimized    (y_buf, uv_buf, tw, th, y_r, uv_r); break;
                }
                bt709::nv21_to_rgb_bt709_full_range_neon(
                    y_r.data(),  tw,
                    uv_r.data(), tw,
                    rgb_out.data(), tw * 3, tw, th);
            }
            break;
        }
        case 21: { // Seg Argmax (8 classes)
            int nc = 8;
            int total = w * h * nc;
            std::vector<float> input_data(total);
            for (int i = 0; i < total; i++) input_data[i] = (float)(i % 256) / 255.0f;
            if (useHalide) {
                Halide::Runtime::Buffer<float> ibuf(input_data.data(), w, h, nc);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h);
                halide_ops::seg_argmax(ibuf, obuf);
            } else {
                // seg_argmax reads raw data as planar: data[c*h*w + y*w + x]
                // Use CV_32FC1 to match the planar layout, not CV_32FC(nc)
                cv::Mat input_mat(h, w, CV_32FC1, input_data.data());
                cv::Mat output_mat;
                opencv_ops::seg_argmax(input_mat, output_mat, nc);
            }
            break;
        }
        case 22: { // Seg Portrait Blur (r=8)
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            int mw = 256, mh = 256;
            std::vector<uint8_t> mask_data(mw * mh);
            int fg_class = 1;
            make_seg_mask_data(mask_data.data(), mw, mh, fg_class);
            int blur_radius = 8;
            float edge_softness = 3.0f;
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::seg_portrait_blur(ibuf, mbuf, fg_class, blur_radius, edge_softness, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat mask_mat(mh, mw, CV_8UC1, mask_data.data());
                cv::Mat out_bgr;
                opencv_ops::seg_portrait_blur(in_bgr, mask_mat, fg_class, blur_radius, edge_softness, out_bgr);
            }
            break;
        }
        case 23: { // Seg Background Replace
            std::vector<uint8_t> fg_rgb(w * h * 3), bg_rgb(w * h * 3), out_rgb(w * h * 3);
            fill_test_rgb(fg_rgb.data(), w, h);
            fill_test_rgb(bg_rgb.data(), w, h); // same size bg for simplicity
            int mw = 256, mh = 256;
            std::vector<uint8_t> mask_data(mw * mh);
            int fg_class = 1;
            make_seg_mask_data(mask_data.data(), mw, mh, fg_class);
            float edge_softness = 3.0f;
            if (useHalide) {
                auto fg_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(fg_rgb.data(), w, h, 3);
                auto bg_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(bg_rgb.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_rgb.data(), w, h, 3);
                halide_ops::seg_bg_replace(fg_buf, bg_buf, mbuf, fg_class, edge_softness, obuf);
            } else {
                cv::Mat fg_mat(h, w, CV_8UC3, fg_rgb.data());
                cv::Mat bg_mat(h, w, CV_8UC3, bg_rgb.data());
                cv::Mat mask_mat(mh, mw, CV_8UC1, mask_data.data());
                cv::Mat out_mat;
                opencv_ops::seg_bg_replace(fg_mat, bg_mat, mask_mat, fg_class, edge_softness, out_mat);
            }
            break;
        }
        case 24: { // Seg Color Style
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            int mw = 256, mh = 256;
            int num_classes = 8;
            std::vector<uint8_t> mask_data(mw * mh);
            make_striped_mask_data(mask_data.data(), mw, mh, num_classes);
            float lut_values[] = {
                0.3f, 0.3f, 0.3f, 0.0f, 0.0f, 0.0f, 0.8f,
                1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.8f, 0.9f, 1.2f, 0.0f, 0.0f, 20.0f, 0.9f,
                0.9f, 1.1f, 0.8f, 0.0f, 10.0f, 0.0f, 0.9f,
                1.1f, 1.0f, 0.9f, 12.0f, 0.0f, 0.0f, 0.7f,
                0.9f, 1.0f, 1.1f, 0.0f, 0.0f, 8.0f, 0.7f,
                1.0f, 1.0f, 1.0f, 15.0f, 0.0f, 10.0f, 0.7f,
                0.8f, 0.8f, 0.8f, 20.0f, 20.0f, 20.0f, 0.7f
            };
            std::vector<float> lut_data(lut_values, lut_values + num_classes * 7);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> mbuf(mask_data.data(), mw, mh);
                Halide::Runtime::Buffer<float> lbuf(lut_data.data(), num_classes, 7);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::seg_color_style(ibuf, mbuf, lbuf, obuf);
            } else {
                cv::Mat in_mat(h, w, CV_8UC3, rgb_in.data());
                cv::Mat mask_mat(mh, mw, CV_8UC1, mask_data.data());
                cv::Mat out_mat;
                opencv_ops::seg_color_style(in_mat, mask_mat, lut_data, num_classes, out_mat);
            }
            break;
        }
        case 25: { // Seg Depth Blur
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            int dw = 256, dh = 256;
            int nk = 3;
            std::vector<uint8_t> depth_data(dw * dh);
            make_depth_map_data(depth_data.data(), dw, dh);
            float config_values[] = {
                0.0f,  0.33f, 0.0f,
                0.33f, 0.66f, 4.0f,
                0.66f, 1.0f,  8.0f,
                0.0f,  0.0f,  0.0f,
                0.0f,  0.0f,  0.0f,
            };
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                Halide::Runtime::Buffer<uint8_t> dbuf(depth_data.data(), dw, dh);
                Halide::Runtime::Buffer<float> cbuf(config_values, 5, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::seg_depth_blur(ibuf, dbuf, cbuf, nk, obuf);
            } else {
                cv::Mat in_mat(h, w, CV_8UC3, rgb_in.data());
                cv::Mat depth_mat(dh, dw, CV_8UC1, depth_data.data());
                std::vector<float> config_vec(config_values, config_values + nk * 3);
                cv::Mat out_mat;
                opencv_ops::seg_depth_blur(in_mat, depth_mat, config_vec, nk, out_mat);
            }
            break;
        }
        default:
            return -1;
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Depth-map guided multi-kernel blur
// -----------------------------------------------------------------------

// Generate a vertical gradient depth map (top=near=0, bottom=far=255)
static void make_depth_map_data(uint8_t* data, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            data[y * w + x] = (uint8_t)(y * 255 / std::max(h - 1, 1));
}

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_segDepthBlur(
    JNIEnv* env, jclass, jobject inputBitmap, jobject outputBitmap,
    jint numKernels, jboolean useHalide)
{
    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int w = in_lock.width(), h = in_lock.height();
    int dw = 256, dh = 256;

    // Create synthetic depth map (vertical gradient)
    std::vector<uint8_t> depth_data(dw * dh);
    make_depth_map_data(depth_data.data(), dw, dh);

    // Default 3-zone kernel config: near(sharp), mid(light blur), far(heavy blur)
    int nk = std::min((int)numKernels, 5);
    float config_values[] = {
        0.0f,  0.33f, 0.0f,   // near: no blur
        0.33f, 0.66f, 4.0f,   // mid: radius 4
        0.66f, 1.0f,  8.0f,   // far: radius 8
        0.0f,  0.0f,  0.0f,   // unused
        0.0f,  0.0f,  0.0f,   // unused
    };

    auto start = Clock::now();

    if (useHalide) {
        std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), w, h);

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        Halide::Runtime::Buffer<uint8_t> dbuf(depth_data.data(), dw, dh);
        Halide::Runtime::Buffer<float> cbuf(config_values, 5, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);

        halide_ops::seg_depth_blur(ibuf, dbuf, cbuf, nk, obuf);

        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, w, h);
    } else {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::Mat in_rgb;
        cv::cvtColor(in_rgba, in_rgb, cv::COLOR_RGBA2RGB);
        cv::Mat depth_cv(dh, dw, CV_8UC1, depth_data.data());

        std::vector<float> config_vec(config_values, config_values + nk * 3);
        cv::Mat out_rgb;

        opencv_ops::seg_depth_blur(in_rgb, depth_cv, config_vec, nk, out_rgb);

        cv::Mat out_rgba;
        cv::cvtColor(out_rgb, out_rgba, cv::COLOR_RGB2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// -----------------------------------------------------------------------
// Utility: Append benchmark result to CSV file on device
// -----------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_example_halidetest_NativeBridge_appendCsv(
    JNIEnv* env, jclass, jstring filePath, jstring csvLine)
{
    const char* path = env->GetStringUTFChars(filePath, nullptr);
    const char* line = env->GetStringUTFChars(csvLine, nullptr);

    std::ofstream ofs(path, std::ios::app);
    if (ofs.is_open()) {
        ofs << line << "\n";
    }

    env->ReleaseStringUTFChars(filePath, path);
    env->ReleaseStringUTFChars(csvLine, line);
}

// -----------------------------------------------------------------------
// Generic Operation Dispatcher (OCP-compliant)
// -----------------------------------------------------------------------
// Dispatches to any registered IOperation by name.
// New operations can be added by creating a single ops/op_*.cpp file
// with self-registration — no modification of this file required.

JNIEXPORT jlong JNICALL
Java_com_example_halidetest_NativeBridge_runOperation(
    JNIEnv* env, jclass, jstring opName,
    jobject inputBitmap, jobject outputBitmap,
    jint targetWidth, jint targetHeight,
    jboolean useHalide)
{
    const char* name = env->GetStringUTFChars(opName, nullptr);
    IOperation* op = OperationRegistry::instance().find(name);
    env->ReleaseStringUTFChars(opName, name);

    if (!op) return -1;

    BitmapLock in_lock(env, inputBitmap);
    BitmapLock out_lock(env, outputBitmap);
    if (!in_lock.is_valid() || !out_lock.is_valid()) return -1;

    int out_w = targetWidth > 0 ? targetWidth : in_lock.width();
    int out_h = targetHeight > 0 ? targetHeight : in_lock.height();

    OperationContext ctx;
    BufferLayout layout = op->halide_layout();
    ctx.prepare_rgb_resize(in_lock, out_lock, out_w, out_h, layout);

    long time_us;
    if (useHalide) {
        time_us = op->run_halide(ctx);
        ctx.write_back(out_lock, layout);
    } else {
        time_us = op->run_opencv(ctx);
        // OpenCV result is in ctx.cv_out (RGB); write back to RGBA
        ctx.write_back(out_lock);
    }

    return time_us;
}

// Query how many operations are registered
JNIEXPORT jint JNICALL
Java_com_example_halidetest_NativeBridge_getRegisteredOpCount(
    JNIEnv* env, jclass)
{
    return (jint)OperationRegistry::instance().size();
}

// Get registered operation name by index
JNIEXPORT jstring JNICALL
Java_com_example_halidetest_NativeBridge_getRegisteredOpName(
    JNIEnv* env, jclass, jint index)
{
    const auto& ops = OperationRegistry::instance().all();
    if (index < 0 || index >= (jint)ops.size()) return nullptr;
    return env->NewStringUTF(ops[index]->name());
}

} // extern "C"
