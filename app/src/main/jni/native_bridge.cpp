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
        halide_dimension_t uv_dims[3] = {
            {0, w / 2, 2},
            {0, h / 2, w},
            {0, 2, 1},
        };
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, 3, uv_dims);

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

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);
        auto obuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_out.data(), ow, oh, 3);

        float angle_rad = angleDegrees * 3.14159265358979f / 180.0f;

        // Use fixed rotation for exact multiples of 90
        int angle_int = (int)angleDegrees;
        if (angleDegrees == 90.0f) {
            halide_ops::rotate_90(ibuf, obuf);
        } else {
            halide_ops::rotate_angle(ibuf, angle_rad, obuf);
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

        auto ibuf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(rgb_in.data(), w, h, 3);

        // Y output: w x h
        Halide::Runtime::Buffer<uint8_t> y_buf((uint8_t*)nv21_ptr, w, h);

        // UV output: (w/2) x (h/2) x 2 interleaved
        uint8_t* uv_ptr = (uint8_t*)nv21_ptr + y_size;
        halide_dimension_t uv_dims[3] = {
            {0, w / 2, 2},
            {0, h / 2, w},
            {0, 2, 1},
        };
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, 3, uv_dims);

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
