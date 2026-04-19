// =============================================================================
// NV21 to RGB Conversion Generator (BT.709 Full-Range / Rec. 709 JPEG variant)
// =============================================================================
//
// Twin of nv21_to_rgb_full_range_generator.cpp but with BT.709 coefficients
// instead of BT.601. Samsung Camera2 on modern Exynos / Snapdragon devices
// emits full-range YUV using BT.709 primaries for HD+ capture resolutions;
// this generator is the correct conversion for that input.
//
// ## Full-Range BT.709 formulas (reference, floating-point)
//
//   R = Y + 1.5748   * (V-128)
//   G = Y - 0.1873   * (U-128) - 0.4681 * (V-128)
//   B = Y + 1.8556   * (U-128)
//   Y: [0, 255]  UV: [0, 255]   (128 = neutral chroma, no Y offset)
//
// ## Fixed-Point Coefficients (Q8, scaled by 256)
//
//   1.5748  * 256 = 403.1488  -> 403
//   0.1873  * 256 =  47.9488  ->  48
//   0.4681  * 256 = 119.8336  -> 120
//   1.8556  * 256 = 475.0336  -> 475
//
//   y_scaled = Y * 256 + 128     (+128 = round-to-nearest-even bias for >>8)
//   R = clamp_u8((y_scaled + 403 * (V-128)) >> 8)
//   G = clamp_u8((y_scaled -  48 * (U-128) - 120 * (V-128)) >> 8)
//   B = clamp_u8((y_scaled + 475 * (U-128)) >> 8)
//
// The hand-rolled NEON reference at app/src/main/jni/bt709_neon_ref.cpp
// uses identical Q8 coefficients, identical +128 bias pre-added into y_scaled,
// and a plain (non-rounding) shift via vqshrun_n_s32 — giving bit-exact
// equivalence with this generator's output.
// =============================================================================

#include "Halide.h"

using namespace Halide;

class Nv21ToRgbBt709FullRange : public Generator<Nv21ToRgbBt709FullRange> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};     // width x height x 3 (RGB interleaved)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // NV21 UV byte layout: V at even offsets, U at odd; each VU pair
        // covers a 2x2 Y block (4:2:0 chroma subsampling).
        Expr uv_x = (x / 2) * 2;
        Expr uv_y = y / 2;
        Expr v_val = cast<int32_t>(uv_plane(uv_x, uv_y))     - 128;
        Expr u_val = cast<int32_t>(uv_plane(uv_x + 1, uv_y)) - 128;

        // y_scaled carries the +128 rounding bias so the final >>8 needs no
        // further adjustment (matches NEON `vqshrun_n_s32`, not `vqrshrun_n_s32`).
        Expr y_scaled = y_val * 256 + 128;

        Expr r = (y_scaled + 403 * v_val) >> 8;
        Expr g = (y_scaled -  48 * u_val - 120 * v_val) >> 8;
        Expr b = (y_scaled + 475 * u_val) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));
    }

    void schedule() {
        // Interleaved RGB output: c is the innermost dim (stride 1), then x (stride 3).
        // Same schedule as nv21_to_rgb_full_range_generator.cpp — proven on ARM64 NEON.
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ToRgbBt709FullRange, nv21_to_rgb_bt709_full_range)
