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

// Fixed 90-degree CW rotation on 3-channel RGB
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

} // namespace halide_ops
