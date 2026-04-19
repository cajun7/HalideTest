#pragma once

#include "image_utils.h"
#include "HalideBuffer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cstdint>

// Centralized buffer management for image processing operations.
// Eliminates the duplicated RGBA<->RGB conversion boilerplate that was
// repeated ~40 times in native_bridge.cpp.
//
// Usage:
//   OperationContext ctx;
//   ctx.prepare_rgb(in_lock, out_lock);  // or prepare_rgb_resize for different output dims
//   // Use ctx.h_in, ctx.h_out for Halide; ctx.cv_in, ctx.cv_out for OpenCV
//   ctx.write_back(out_lock);

enum class BufferLayout {
    INTERLEAVED,  // stride: x=3, c=1 (most generators)
    PLANAR         // stride: x=1, y=w, c=w*h (rotate_arbitrary)
};

struct OperationContext {
    int src_w = 0, src_h = 0;
    int dst_w = 0, dst_h = 0;

    // Raw RGB data (owned)
    std::vector<uint8_t> rgb_in;
    std::vector<uint8_t> rgb_out;

    // Halide buffers (views over rgb_in/rgb_out or planar data)
    Halide::Runtime::Buffer<uint8_t> h_in;
    Halide::Runtime::Buffer<uint8_t> h_out;

    // OpenCV Mats (views over rgb_in/rgb_out)
    cv::Mat cv_in;
    cv::Mat cv_out;

    // Planar data for operations requiring planar layout
    std::vector<uint8_t> planar_in;
    std::vector<uint8_t> planar_out;

    // Prepare buffers for same-size RGB bitmap operation (blur, color convert, etc.)
    void prepare_rgb(BitmapLock& in_lock, BitmapLock& out_lock,
                     BufferLayout layout = BufferLayout::INTERLEAVED) {
        prepare_rgb_resize(in_lock, out_lock,
                           in_lock.width(), in_lock.height(), layout);
    }

    // Prepare buffers for RGB bitmap operation with different output dimensions (resize, rotate)
    void prepare_rgb_resize(BitmapLock& in_lock, BitmapLock& out_lock,
                            int out_w, int out_h,
                            BufferLayout layout = BufferLayout::INTERLEAVED) {
        src_w = in_lock.width();
        src_h = in_lock.height();
        dst_w = out_w;
        dst_h = out_h;

        // Extract RGB from RGBA input
        rgb_in.resize(src_w * src_h * 3);
        rgba_to_rgb((uint8_t*)in_lock.pixels, rgb_in.data(), src_w, src_h);

        // Allocate RGB output
        rgb_out.resize(dst_w * dst_h * 3);

        if (layout == BufferLayout::INTERLEAVED) {
            h_in = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
                rgb_in.data(), src_w, src_h, 3);
            h_out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
                rgb_out.data(), dst_w, dst_h, 3);
        } else {
            // Planar: convert interleaved RGB to planar for Halide
            planar_in.resize(src_w * src_h * 3);
            interleaved_to_planar(rgb_in.data(), planar_in.data(), src_w, src_h);
            h_in = Halide::Runtime::Buffer<uint8_t>(planar_in.data(), src_w, src_h, 3);
            h_out = Halide::Runtime::Buffer<uint8_t>(dst_w, dst_h, 3);
        }

        // OpenCV Mat (wraps RGB data, OpenCV functions handle BGR conversion internally)
        cv_in = cv::Mat(src_h, src_w, CV_8UC3, rgb_in.data());
        cv_out = cv::Mat(dst_h, dst_w, CV_8UC3, rgb_out.data());
    }

    // Prepare OpenCV input with RGBA->BGR conversion (for OpenCV ops that expect BGR)
    void prepare_opencv_bgr(BitmapLock& in_lock) {
        cv::Mat in_rgba = in_lock.as_opencv_rgba();
        cv::cvtColor(in_rgba, cv_in, cv::COLOR_RGBA2BGR);
    }

    // Write results back to RGBA output bitmap
    void write_back(BitmapLock& out_lock, BufferLayout layout = BufferLayout::INTERLEAVED) {
        if (layout == BufferLayout::PLANAR) {
            // Convert planar Halide output back to interleaved RGB
            planar_to_interleaved(h_out.data(), rgb_out.data(), dst_w, dst_h);
        }
        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, dst_w, dst_h);
    }

    // Write OpenCV BGR result back to RGBA output bitmap
    void write_back_opencv(BitmapLock& out_lock) {
        cv::Mat out_rgba;
        cv::cvtColor(cv_out, out_rgba, cv::COLOR_BGR2RGBA);
        out_rgba.copyTo(out_lock.as_opencv_rgba());
    }
};

// NV21-specific context for operations that take NV21 byte arrays
struct NV21Context {
    int src_w = 0, src_h = 0;
    int dst_w = 0, dst_h = 0;

    // Halide Y/UV plane buffers (views over NV21 data)
    Halide::Runtime::Buffer<uint8_t> y_buf;
    Halide::Runtime::Buffer<uint8_t> uv_buf;

    // Halide output (interleaved RGB)
    std::vector<uint8_t> rgb_out;
    Halide::Runtime::Buffer<uint8_t> h_out;

    // OpenCV NV21 Mat
    cv::Mat nv21_mat;

    void prepare(uint8_t* nv21_data, int w, int h, int out_w, int out_h) {
        src_w = w;
        src_h = h;
        dst_w = out_w;
        dst_h = out_h;

        // Halide: separate Y and UV plane views
        y_buf = Halide::Runtime::Buffer<uint8_t>(nv21_data, w, h);
        uv_buf = Halide::Runtime::Buffer<uint8_t>(nv21_data + w * h, w, h / 2);

        // Halide output: interleaved RGB
        rgb_out.resize(out_w * out_h * 3);
        h_out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
            rgb_out.data(), out_w, out_h, 3);

        // OpenCV: single-channel NV21 frame
        nv21_mat = cv::Mat(h + h / 2, w, CV_8UC1, nv21_data);
    }

    void write_back(BitmapLock& out_lock) {
        rgb_to_rgba(rgb_out.data(), (uint8_t*)out_lock.pixels, dst_w, dst_h);
    }

    void write_back_opencv(cv::Mat& rgb_result, BitmapLock& out_lock) {
        cv::Mat rgba_out;
        cv::cvtColor(rgb_result, rgba_out, cv::COLOR_RGB2RGBA);
        rgba_out.copyTo(out_lock.as_opencv_rgba());
    }
};
