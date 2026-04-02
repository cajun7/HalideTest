#pragma once

#include "HalideBuffer.h"
#include <cstdint>

// Halide operation wrappers.
// Each function takes Halide Runtime Buffers and calls the AOT-generated function.
namespace halide_ops {

// RGB <-> BGR channel swap (3-channel interleaved)
int rgb_bgr(Halide::Runtime::Buffer<uint8_t>& input,
            Halide::Runtime::Buffer<uint8_t>& output);

// NV21 -> RGB conversion
int nv21_to_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                Halide::Runtime::Buffer<uint8_t>& output);

// Gaussian blur on single channel (Y plane)
int gaussian_blur_gray(Halide::Runtime::Buffer<uint8_t>& input,
                       Halide::Runtime::Buffer<uint8_t>& output);

// Gaussian blur on 3-channel RGB
int gaussian_blur_rgb(Halide::Runtime::Buffer<uint8_t>& input,
                      Halide::Runtime::Buffer<uint8_t>& output);

// Lens blur (disc kernel) on 3-channel RGB
int lens_blur(Halide::Runtime::Buffer<uint8_t>& input,
              int radius,
              Halide::Runtime::Buffer<uint8_t>& output);

// Bilinear resize on 3-channel RGB
int resize_bilinear(Halide::Runtime::Buffer<uint8_t>& input,
                    float scale_x, float scale_y,
                    Halide::Runtime::Buffer<uint8_t>& output);

// Bicubic resize on 3-channel RGB
int resize_bicubic(Halide::Runtime::Buffer<uint8_t>& input,
                   float scale_x, float scale_y,
                   Halide::Runtime::Buffer<uint8_t>& output);

// Fixed rotation — all directions
int rotate_90cw(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& output);
int rotate_180(Halide::Runtime::Buffer<uint8_t>& input,
               Halide::Runtime::Buffer<uint8_t>& output);
int rotate_270cw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output);

// Counter-clockwise rotation convenience aliases
int rotate_90ccw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output);
int rotate_270ccw(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output);

// Backward-compatible alias for rotate_90cw
int rotate_90(Halide::Runtime::Buffer<uint8_t>& input,
              Halide::Runtime::Buffer<uint8_t>& output);

// Arbitrary angle rotation on 3-channel RGB
int rotate_angle(Halide::Runtime::Buffer<uint8_t>& input,
                 float angle_rad,
                 Halide::Runtime::Buffer<uint8_t>& output);

// RGB to NV21 conversion (produces Y plane + interleaved VU)
int rgb_to_nv21(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& y_output,
                Halide::Runtime::Buffer<uint8_t>& uv_output);

// INTER_AREA resize on 3-channel RGB (box-filter downsampling)
int resize_area(Halide::Runtime::Buffer<uint8_t>& input,
                float scale_x, float scale_y,
                Halide::Runtime::Buffer<uint8_t>& output);

// Letterbox resize on 3-channel RGB (aspect-ratio-preserving with black padding)
int resize_letterbox(Halide::Runtime::Buffer<uint8_t>& input,
                     int target_w, int target_h,
                     Halide::Runtime::Buffer<uint8_t>& output);

// Target-size resize APIs (exact pixel dimensions, no scale float)
int resize_bilinear_target(Halide::Runtime::Buffer<uint8_t>& input,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output);

int resize_bicubic_target(Halide::Runtime::Buffer<uint8_t>& input,
                          int target_w, int target_h,
                          Halide::Runtime::Buffer<uint8_t>& output);

int resize_area_target(Halide::Runtime::Buffer<uint8_t>& input,
                       int target_w, int target_h,
                       Halide::Runtime::Buffer<uint8_t>& output);

// Flip operations on 3-channel RGB
int flip_horizontal(Halide::Runtime::Buffer<uint8_t>& input,
                    Halide::Runtime::Buffer<uint8_t>& output);

int flip_vertical(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Resize (Bilinear) -> RGB pipeline
// rotation_degrees_cw: 0, 90, 180, 270
int nv21_rotate_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                           Halide::Runtime::Buffer<uint8_t>& uv_plane,
                           int rotation_degrees_cw,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Flip -> Resize (Bilinear) -> RGB pipeline
// flip_code: 0=none, 1=horizontal, 2=vertical
int nv21_rotate_flip_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw, int flip_code,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Resize (INTER_AREA) -> RGB pipeline
int nv21_rotate_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Flip -> Resize (INTER_AREA) -> RGB pipeline
int nv21_rotate_flip_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                     Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                     int rotation_degrees_cw, int flip_code,
                                     int target_w, int target_h,
                                     Halide::Runtime::Buffer<uint8_t>& output);

// Segmentation post-processing: argmax across class logits (planar float32 -> uint8 class mask)
int seg_argmax(Halide::Runtime::Buffer<float>& input,
               Halide::Runtime::Buffer<uint8_t>& output);

} // namespace halide_ops
