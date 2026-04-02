#pragma once

#include <opencv2/core.hpp>
#include <cstdint>

// OpenCV operation wrappers for benchmark comparison.
namespace opencv_ops {

// RGB <-> BGR conversion
void rgb_bgr(const cv::Mat& input, cv::Mat& output);

// NV21 -> RGB conversion (input is single-channel h*3/2 x w NV21 frame)
void nv21_to_rgb(const cv::Mat& nv21, cv::Mat& output);

// NV21 -> RGB with bilinear UV upsampling (timing reference; OpenCV uses nearest-neighbor)
void nv21_yuv444_rgb(const cv::Mat& nv21, cv::Mat& output);

// NV21 -> RGB full-range BT.601 (timing reference; OpenCV uses limited-range)
void nv21_to_rgb_full_range(const cv::Mat& nv21, cv::Mat& output);

// Gaussian blur on single-channel
void gaussian_blur_gray(const cv::Mat& input, cv::Mat& output, int kernel_size);

// Gaussian blur on 3-channel
void gaussian_blur_rgb(const cv::Mat& input, cv::Mat& output, int kernel_size);

// Lens blur (disc kernel) on 3-channel
void lens_blur(const cv::Mat& input, cv::Mat& output, int radius);

// Bilinear resize
void resize_bilinear(const cv::Mat& input, cv::Mat& output, int out_w, int out_h);

// Bicubic resize
void resize_bicubic(const cv::Mat& input, cv::Mat& output, int out_w, int out_h);

// Fixed 90-degree CW rotation
void rotate_90(const cv::Mat& input, cv::Mat& output);

// Fixed 180-degree rotation
void rotate_180(const cv::Mat& input, cv::Mat& output);

// Fixed 270-degree CW rotation
void rotate_270(const cv::Mat& input, cv::Mat& output);

// Arbitrary angle rotation (degrees)
void rotate_angle(const cv::Mat& input, cv::Mat& output, float angle_degrees);

// RGB to NV21 conversion
void rgb_to_nv21(const cv::Mat& rgb, cv::Mat& nv21_output);

// INTER_AREA resize (optimal downsampling)
void resize_area(const cv::Mat& input, cv::Mat& output, int out_w, int out_h);

// Letterbox resize (aspect-ratio-preserving with black padding)
void resize_letterbox(const cv::Mat& input, cv::Mat& output, int target_w, int target_h);

// Flip operations
void flip_horizontal(const cv::Mat& input, cv::Mat& output);
void flip_vertical(const cv::Mat& input, cv::Mat& output);

// Fused NV21 -> Rotate -> Resize -> RGB (reference implementation using separate steps)
// rotation_degrees_cw: 0, 90, 180, 270
// flip_code: 0=none, 1=horizontal, 2=vertical
void nv21_rotate_flip_resize_rgb(const cv::Mat& nv21, cv::Mat& output,
                                 int rotation_degrees_cw, int flip_code,
                                 int target_w, int target_h, int interp);

// NV21 -> Resize -> Pad -> Rotate for ML preprocessing (chained OpenCV steps)
void nv21_resize_pad_rotate(const cv::Mat& nv21, cv::Mat& output,
                            int rotation_degrees_cw, int target_size);

// Segmentation argmax across channels (per-pixel class selection)
void seg_argmax(const cv::Mat& input, cv::Mat& output, int num_classes);

} // namespace opencv_ops
