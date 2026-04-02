// =============================================================================
// NV21 to RGB Conversion Generator (BT.601 Full-Range / JFIF)
// =============================================================================
//
// Same as nv21_to_rgb_generator.cpp but uses FULL-RANGE BT.601 coefficients.
// This is the correct variant for Android Camera API, which outputs full-range
// YUV by default (also known as the JFIF/JPEG variant of BT.601).
//
// ## Full-Range vs Limited-Range
//
// Limited-range (nv21_to_rgb_generator.cpp):
//   - Y:  [16, 235]  (16 = black, 235 = white)
//   - UV: [16, 240]
//   - Used by: broadcast TV, older camera APIs
//   - Conversion: R = 1.164*(Y-16) + 1.596*(V-128)
//
// Full-range (this file):
//   - Y:  [0, 255]   (0 = black, 255 = white — full byte range)
//   - UV: [0, 255]   (128 = neutral)
//   - Used by: Android Camera API, JFIF/JPEG, most modern mobile cameras
//   - Conversion: R = Y + 1.402*(V-128)
//
// Key differences in the formulas:
//   - No Y offset (Y maps directly, no subtraction of 16)
//   - Different scaling coefficients (no 1.164 multiplier needed)
//
// ## Fixed-Point Coefficients (scaled by 256)
//
//   1.402   * 256 = 358.912  -> 359
//   0.34414 * 256 = 88.10    -> 88
//   0.71414 * 256 = 182.82   -> 183
//   1.772   * 256 = 453.63   -> 454
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class Nv21ToRgbFullRange : public Generator<Nv21ToRgbFullRange> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // UV sampling: same NV21 byte layout as limited-range version.
        // See nv21_to_rgb_generator.cpp for detailed NV21 layout explanation.
        Expr uv_x = (x / 2) * 2;
        Expr uv_y = y / 2;
        Expr v_val = cast<int32_t>(uv_plane(uv_x, uv_y)) - 128;
        Expr u_val = cast<int32_t>(uv_plane(uv_x + 1, uv_y)) - 128;

        // Full-range BT.601 (fixed-point, shift by 8).
        //
        // Unlike limited-range, there is no Y-16 offset because Y already
        // spans [0, 255]. We scale Y by 256 (identity after >>8) and add
        // 128 as rounding bias.
        //
        // y_scaled = Y * 256 + 128
        // R = (y_scaled + 359 * V) >> 8  ≈ Y + 1.402 * V
        // G = (y_scaled - 88 * U - 183 * V) >> 8  ≈ Y - 0.344 * U - 0.714 * V
        // B = (y_scaled + 454 * U) >> 8  ≈ Y + 1.772 * U
        Expr y_scaled = y_val * 256 + 128;

        Expr r = (y_scaled + 359 * v_val) >> 8;
        Expr g = (y_scaled - 88 * u_val - 183 * v_val) >> 8;
        Expr b = (y_scaled + 454 * u_val) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));
    }

    void schedule() {
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ToRgbFullRange, nv21_to_rgb_full_range)
