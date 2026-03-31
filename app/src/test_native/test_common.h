#pragma once

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "HalideBuffer.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

// Maximum allowed per-pixel difference for accuracy comparison.
// Rounding differences between Halide (integer/fixed-point) and OpenCV (float) are expected.
constexpr int DEFAULT_TOLERANCE = 2;

// ---------------------------------------------------------------------------
// Pixel comparison utilities
// ---------------------------------------------------------------------------

// Compare 3-channel Halide buffer (RGB interleaved) against OpenCV Mat.
// OpenCV default is BGR; set opencv_is_bgr=true to swap channels during comparison.
inline void compare_buffers_rgb(
    const Halide::Runtime::Buffer<uint8_t>& halide_buf,
    const cv::Mat& opencv_mat,
    int tolerance = DEFAULT_TOLERANCE,
    bool opencv_is_bgr = true)
{
    ASSERT_EQ(halide_buf.width(), opencv_mat.cols);
    ASSERT_EQ(halide_buf.height(), opencv_mat.rows);
    ASSERT_EQ(halide_buf.channels(), opencv_mat.channels());

    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < halide_buf.height(); y++) {
        for (int x = 0; x < halide_buf.width(); x++) {
            for (int c = 0; c < halide_buf.channels(); c++) {
                uint8_t h_val = halide_buf(x, y, c);
                int cv_c = opencv_is_bgr ? (2 - c) : c;
                uint8_t cv_val = opencv_mat.at<cv::Vec3b>(y, x)[cv_c];
                int diff = std::abs((int)h_val - (int)cv_val);
                if (diff > tolerance) {
                    mismatches++;
                }
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float total_pixels = (float)(halide_buf.width() * halide_buf.height() * halide_buf.channels());
    float mismatch_pct = 100.0f * mismatches / total_pixels;
    EXPECT_LT(mismatch_pct, 1.0f)
        << "Too many pixel mismatches: " << mismatches
        << " (" << mismatch_pct << "%), max_diff=" << max_diff;
}

// Compare single-channel Halide buffer against OpenCV Mat (CV_8UC1).
inline void compare_buffers_gray(
    const Halide::Runtime::Buffer<uint8_t>& halide_buf,
    const cv::Mat& opencv_mat,
    int tolerance = DEFAULT_TOLERANCE)
{
    ASSERT_EQ(halide_buf.width(), opencv_mat.cols);
    ASSERT_EQ(halide_buf.height(), opencv_mat.rows);
    ASSERT_EQ(1, opencv_mat.channels());

    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < halide_buf.height(); y++) {
        for (int x = 0; x < halide_buf.width(); x++) {
            uint8_t h_val = halide_buf(x, y);
            uint8_t cv_val = opencv_mat.at<uint8_t>(y, x);
            int diff = std::abs((int)h_val - (int)cv_val);
            if (diff > tolerance) {
                mismatches++;
            }
            max_diff = std::max(max_diff, diff);
        }
    }
    float total_pixels = (float)(halide_buf.width() * halide_buf.height());
    float mismatch_pct = 100.0f * mismatches / total_pixels;
    EXPECT_LT(mismatch_pct, 1.0f)
        << "Too many pixel mismatches: " << mismatches
        << " (" << mismatch_pct << "%), max_diff=" << max_diff;
}

// ---------------------------------------------------------------------------
// Synthetic test data generators
// ---------------------------------------------------------------------------

// Generate a gradient test image (3-channel BGR for OpenCV).
inline cv::Mat make_test_image_bgr(int width, int height) {
    cv::Mat img(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(x * 255 / std::max(width - 1, 1)),
                static_cast<uint8_t>(y * 255 / std::max(height - 1, 1)),
                static_cast<uint8_t>((x + y) * 255 / std::max(width + height - 2, 1))
            );
        }
    }
    return img;
}

// Generate a gradient test image (single-channel grayscale).
inline cv::Mat make_test_image_gray(int width, int height) {
    cv::Mat img(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img.at<uint8_t>(y, x) = static_cast<uint8_t>((x + y) % 256);
        }
    }
    return img;
}

// Generate synthetic NV21 data: Y plane (width*height) + interleaved VU plane (width*(height/2)).
// NV21 layout: Y full-res, then interleaved V,U at half resolution per row.
inline void make_nv21_data(int width, int height,
                           std::vector<uint8_t>& y_plane,
                           std::vector<uint8_t>& uv_interleaved) {
    y_plane.resize(width * height);
    // UV plane: for each 2x2 block, one V byte and one U byte -> width bytes per UV row, height/2 rows
    uv_interleaved.resize(width * (height / 2));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            // Gradient Y values
            y_plane[j * width + i] = static_cast<uint8_t>((i + j * 2) % 256);
        }
    }
    for (int j = 0; j < height / 2; j++) {
        for (int i = 0; i < width / 2; i++) {
            // Interleaved VU: V at even index, U at odd index
            uv_interleaved[j * width + 2 * i + 0] = static_cast<uint8_t>(128 + (i % 64));  // V
            uv_interleaved[j * width + 2 * i + 1] = static_cast<uint8_t>(128 + (j % 64));  // U
        }
    }
}

// Assemble NV21 data into a single contiguous buffer (as Android camera provides).
inline std::vector<uint8_t> make_nv21_contiguous(int width, int height) {
    std::vector<uint8_t> y_plane, uv_plane;
    make_nv21_data(width, height, y_plane, uv_plane);
    std::vector<uint8_t> nv21(y_plane.size() + uv_plane.size());
    std::copy(y_plane.begin(), y_plane.end(), nv21.begin());
    std::copy(uv_plane.begin(), uv_plane.end(), nv21.begin() + y_plane.size());
    return nv21;
}

// ---------------------------------------------------------------------------
// Buffer conversion helpers
// ---------------------------------------------------------------------------

// Wrap cv::Mat (interleaved) as Halide Runtime Buffer (zero-copy).
// For 3-channel: returns Buffer<uint8_t> with dimensions (width, height, channels).
inline Halide::Runtime::Buffer<uint8_t> mat_to_halide_interleaved(cv::Mat& mat) {
    return Halide::Runtime::Buffer<uint8_t>::make_interleaved(
        mat.data, mat.cols, mat.rows, mat.channels());
}

// Copy Halide Buffer (x, y, c) data into an OpenCV Mat for display/comparison.
inline cv::Mat halide_to_mat_rgb(const Halide::Runtime::Buffer<uint8_t>& buf) {
    cv::Mat mat(buf.height(), buf.width(), CV_8UC3);
    for (int y = 0; y < buf.height(); y++) {
        for (int x = 0; x < buf.width(); x++) {
            // Halide is RGB, OpenCV Mat is BGR
            mat.at<cv::Vec3b>(y, x) = cv::Vec3b(
                buf(x, y, 2),  // B
                buf(x, y, 1),  // G
                buf(x, y, 0)   // R
            );
        }
    }
    return mat;
}

// Standard test resolutions including odd sizes for edge-case testing.
inline std::vector<std::pair<int, int>> get_test_resolutions() {
    return {
        {320, 240},
        {640, 480},
        {641, 481},    // odd resolution
        {1280, 720},
        {1279, 719},   // odd resolution
        {1920, 1080},
    };
}
