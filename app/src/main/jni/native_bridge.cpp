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

        // Use planar output (Halide default stride) then copy to RGBA
        Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
        halide_ops::nv21_to_rgb(y_buf, uv_buf, obuf);

        // Copy planar RGB to RGBA pixel by pixel
        uint8_t* dst = (uint8_t*)out_lock.pixels;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int di = (py * w + px) * 4;
                dst[di + 0] = obuf(px, py, 0);
                dst[di + 1] = obuf(px, py, 1);
                dst[di + 2] = obuf(px, py, 2);
                dst[di + 3] = 255;
            }
        }
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
            Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
            Halide::Runtime::Buffer<uint8_t> obuf(ow, oh, 3);
            for (int py = 0; py < h; py++)
                for (int px = 0; px < w; px++)
                    for (int pc = 0; pc < 3; pc++)
                        ibuf(px, py, pc) = rgb_in[(py * w + px) * 3 + pc];
            halide_ops::rotate_angle(ibuf, angle_rad, obuf);
            for (int py = 0; py < oh; py++)
                for (int px = 0; px < ow; px++)
                    for (int pc = 0; pc < 3; pc++)
                        rgb_out[(py * ow + px) * 3 + pc] = obuf(px, py, pc);
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

        // Copy interleaved RGB to planar buffer for Halide
        Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int si = (py * w + px) * 3;
                ibuf(px, py, 0) = rgb_in[si + 0];
                ibuf(px, py, 1) = rgb_in[si + 1];
                ibuf(px, py, 2) = rgb_in[si + 2];
            }
        }

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
        Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
        halide_ops::nv21_yuv444_rgb(y_buf, uv_buf, obuf);

        uint8_t* dst = (uint8_t*)out_lock.pixels;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int di = (py * w + px) * 4;
                dst[di + 0] = obuf(px, py, 0);
                dst[di + 1] = obuf(px, py, 1);
                dst[di + 2] = obuf(px, py, 2);
                dst[di + 3] = 255;
            }
        }
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
        Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
        halide_ops::nv21_to_rgb_full_range(y_buf, uv_buf, obuf);

        uint8_t* dst = (uint8_t*)out_lock.pixels;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int di = (py * w + px) * 4;
                dst[di + 0] = obuf(px, py, 0);
                dst[di + 1] = obuf(px, py, 1);
                dst[di + 2] = obuf(px, py, 2);
                dst[di + 3] = 255;
            }
        }
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

        Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
        halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, obuf);

        uint8_t* dst = (uint8_t*)out_lock.pixels;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int di = (py * w + px) * 4;
                dst[di + 0] = obuf(px, py, 0);
                dst[di + 1] = obuf(px, py, 1);
                dst[di + 2] = obuf(px, py, 2);
                dst[di + 3] = 255;
            }
        }
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

        Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int si = (py * w + px) * 3;
                ibuf(px, py, 0) = rgb_in[si + 0];
                ibuf(px, py, 1) = rgb_in[si + 1];
                ibuf(px, py, 2) = rgb_in[si + 2];
            }
        }

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
        case 0: { // RGB to BGR
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::rgb_bgr(ibuf, obuf);
            } else {
                cv::Mat in_rgb(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::rgb_bgr(in_rgb, out_bgr);
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
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                halide_ops::nv21_to_rgb(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_to_rgb(nv21_mat, rgb_out);
            }
            break;
        }
        case 2: { // Gaussian Blur (5x5)
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
        case 3: { // Lens Blur (r=4)
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
        case 4: { // Resize Bilinear
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bilinear(ibuf, sx, sy, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bilinear(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 5: { // Resize Bicubic
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bicubic(ibuf, sx, sy, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bicubic(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 6: { // Rotate 90
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
        case 7: { // Rotate Arbitrary (45)
            std::vector<uint8_t> rgb_in(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                float angle_rad = 45.0f * 3.14159265358979f / 180.0f;
                Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                for (int py = 0; py < h; py++)
                    for (int px = 0; px < w; px++)
                        for (int pc = 0; pc < 3; pc++)
                            ibuf(px, py, pc) = rgb_in[(py * w + px) * 3 + pc];
                halide_ops::rotate_angle(ibuf, angle_rad, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::rotate_angle(in_bgr, out_bgr, 45.0f);
            }
            break;
        }
        case 8: { // RGB to NV21
            std::vector<uint8_t> rgb_in(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
                for (int py = 0; py < h; py++)
                    for (int px = 0; px < w; px++)
                        for (int pc = 0; pc < 3; pc++)
                            ibuf(px, py, pc) = rgb_in[(py * w + px) * 3 + pc];
                Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(w, h / 2);
                halide_ops::rgb_to_nv21(ibuf, y_buf, uv_buf);
            } else {
                cv::Mat in_rgb(h, w, CV_8UC3, rgb_in.data());
                cv::Mat nv21_out;
                opencv_ops::rgb_to_nv21(in_rgb, nv21_out);
            }
            break;
        }
        case 9: { // Resize Area
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_area(ibuf, sx, sy, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_area(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 10: { // Resize Letterbox
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
        case 11: { // Flip Horizontal
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
        case 12: { // Flip Vertical
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), w, h, 3);
                halide_ops::flip_vertical(ibuf, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::flip_vertical(in_bgr, out_bgr);
            }
            break;
        }
        case 13: { // Resize Bilinear Target
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bilinear_target(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bilinear(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 14: { // Resize Bicubic Target
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_bicubic_target(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_bicubic(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 15: { // Resize Area Target
            std::vector<uint8_t> rgb_in(w * h * 3), rgb_out(tw * th * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::resize_area_target(ibuf, tw, th, obuf);
            } else {
                cv::Mat in_bgr(h, w, CV_8UC3, rgb_in.data());
                cv::Mat out_bgr;
                opencv_ops::resize_area(in_bgr, out_bgr, tw, th);
            }
            break;
        }
        case 16: { // NV21 Pipeline Bilinear (rotate+resize)
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
        case 17: { // NV21 Pipeline Area (rotate+resize)
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(tw * th * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::nv21_rotate_flip_resize_area_rgb(y_buf, uv_buf, 90, 0, tw, th, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_rotate_flip_resize_rgb(nv21_mat, rgb_out, 90, 0, tw, th, cv::INTER_AREA);
            }
            break;
        }
        case 18: { // NV21 YUV444 RGB
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                halide_ops::nv21_yuv444_rgb(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_yuv444_rgb(nv21_mat, rgb_out);
            }
            break;
        }
        case 19: { // NV21 to RGB Full-Range
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                halide_ops::nv21_to_rgb_full_range(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_to_rgb_full_range(nv21_mat, rgb_out);
            }
            break;
        }
        case 20: { // NV21 Resize+Pad+Rotate (384)
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
                cv::Mat input_mat(h, w, CV_32FC(nc), input_data.data());
                cv::Mat output_mat;
                opencv_ops::seg_argmax(input_mat, output_mat, nc);
            }
            break;
        }
        case 22: { // RGB BGR Optimized
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
        case 23: { // NV21 to RGB Optimized
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> obuf(w, h, 3);
                halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_to_rgb_optimized(nv21_mat, rgb_out);
            }
            break;
        }
        case 24: { // RGB to NV21 Optimized
            std::vector<uint8_t> rgb_in(w * h * 3);
            fill_test_rgb(rgb_in.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> ibuf(w, h, 3);
                for (int py = 0; py < h; py++)
                    for (int px = 0; px < w; px++)
                        for (int pc = 0; pc < 3; pc++)
                            ibuf(px, py, pc) = rgb_in[(py * w + px) * 3 + pc];
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
        case 25: { // Resize Bilinear Optimized
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
        case 26: { // Resize Area Optimized
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
        case 27: { // Resize Bicubic Optimized
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
        case 28: { // NV21 Resize Bilinear Optimized
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
        case 29: { // NV21 Resize Area Optimized
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
                Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
                halide_ops::nv21_resize_area_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat nv21_out;
                opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_AREA);
            }
            break;
        }
        case 30: { // NV21 Resize Bicubic Optimized
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
                Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);
                halide_ops::nv21_resize_bicubic_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat nv21_out;
                opencv_ops::nv21_resize_optimized(nv21_mat, nv21_out, tw, th, cv::INTER_CUBIC);
            }
            break;
        }
        case 31: { // NV21 Resize RGB Bilinear Optimized
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
        case 32: { // NV21 Resize RGB Area Optimized
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(tw * th * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::nv21_resize_rgb_area_optimized(y_buf, uv_buf, tw, th, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_AREA);
            }
            break;
        }
        case 33: { // NV21 Resize RGB Bicubic Optimized
            int nv21_size = w * h + w * (h / 2);
            std::vector<uint8_t> nv21(nv21_size);
            fill_test_nv21(nv21.data(), w, h);
            if (useHalide) {
                Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
                Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + w * h, w, h / 2);
                std::vector<uint8_t> rgb_out(tw * th * 3);
                auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), tw, th, 3);
                halide_ops::nv21_resize_rgb_bicubic_optimized(y_buf, uv_buf, tw, th, obuf);
            } else {
                cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
                cv::Mat rgb_out;
                opencv_ops::nv21_resize_rgb_optimized(nv21_mat, rgb_out, tw, th, cv::INTER_CUBIC);
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

} // extern "C"
