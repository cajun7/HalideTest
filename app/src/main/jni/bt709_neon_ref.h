#pragma once

#include <cstdint>

// =============================================================================
// BT.709 Full-Range NV21 -> RGB Reference (hand-rolled ARM NEON + portable scalar).
//
// This is the "OpenCV baseline" that Halide is being benchmarked against:
// OpenCV 3.4.x has no BT.709 NV21 cvtColor path, so the production code we are
// replacing is a hand-written NEON routine. The scalar sibling is the oracle
// used to self-test the NEON one.
//
// Coefficients (Q8 fixed-point, scaled by 256):
//   R = Y + 1.5748 * (V-128)   ->  403*(V-128)
//   G = Y - 0.1873 * (U-128) - 0.4681 * (V-128)  ->  -48*(U-128) - 120*(V-128)
//   B = Y + 1.8556 * (U-128)   ->  475*(U-128)
//
//   y_scaled = Y*256 + 128                      (+128 = round-to-nearest bias)
//   R = clamp_u8((y_scaled + 403*v) >> 8)
//   G = clamp_u8((y_scaled -  48*u - 120*v) >> 8)
//   B = clamp_u8((y_scaled + 475*u) >> 8)
//
// NV21 chroma layout: UV plane is width x (height/2) bytes of V,U,V,U,...
//   V at even byte offsets, U at odd. Each VU pair covers a 2x2 Y block.
//
// Output layout: interleaved RGB (R0 G0 B0 R1 G1 B1 ...).
// Channel order is RGB (matches every other generator in this repo).
// Binary-compatible with cv::Mat(h, w, CV_8UC3, rgb, rgb_stride).
// =============================================================================

namespace bt709 {

// Portable scalar reference. Works everywhere. Used as the equivalence oracle.
void nv21_to_rgb_bt709_full_range_scalar(
    const uint8_t* y,   int y_stride,
    const uint8_t* uv,  int uv_stride,
    uint8_t*       rgb, int rgb_stride,
    int width, int height);

// ARM NEON implementation. On non-ARM hosts, falls through to the scalar path.
// Processes 16 Y pixels per inner iteration and reuses each UV row across
// two output rows (NV21 4:2:0 chroma subsampling).
void nv21_to_rgb_bt709_full_range_neon(
    const uint8_t* y,   int y_stride,
    const uint8_t* uv,  int uv_stride,
    uint8_t*       rgb, int rgb_stride,
    int width, int height);

}  // namespace bt709
