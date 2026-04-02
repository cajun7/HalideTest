#include "Halide.h"

using namespace Halide;

// NV21 to RGB conversion using full-range BT.601 coefficients (JFIF/Android Camera).
//
// Android Camera API outputs full-range YUV by default:
//   Y:  0-255  (no offset, scale = 1.0)
//   UV: 0-255  (bias at 128)
//
// Full-range BT.601 (ITU-R BT.601, JFIF variant):
//   R = Y + 1.402 * (V - 128)
//   G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
//   B = Y + 1.772 * (U - 128)
//
// Fixed-point coefficients (scaled by 256):
//   1.402 * 256 = 358.912 -> 359
//   0.34414 * 256 = 88.10 -> 88
//   0.71414 * 256 = 182.82 -> 183
//   1.772 * 256 = 453.63 -> 454
//
// NV21 memory layout: same as nv21_to_rgb_generator.cpp
class Nv21ToRgbFullRange : public Generator<Nv21ToRgbFullRange> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr y_val = cast<int32_t>(y_plane(x, y));

        Expr uv_x = (x / 2) * 2;
        Expr uv_y = y / 2;
        Expr v_val = cast<int32_t>(uv_plane(uv_x, uv_y)) - 128;
        Expr u_val = cast<int32_t>(uv_plane(uv_x + 1, uv_y)) - 128;

        // Full-range BT.601 (fixed-point, shift by 8)
        // No Y offset (full range: Y 0-255 maps directly)
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
