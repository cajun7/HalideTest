#include "Halide.h"

using namespace Halide;

// NV21 to RGB conversion using BT.601 coefficients (fixed-point integer arithmetic).
//
// NV21 memory layout:
//   Y plane:  width x height (full resolution), uint8
//   UV plane: width x (height/2) raw bytes, interleaved V,U pairs at half resolution
//             Stored as: V0 U0 V1 U1 ... (width bytes per row, height/2 rows)
//
// We model the UV plane as a 2D buffer: width x (height/2) raw bytes.
// V is at even byte offsets, U at odd byte offsets within each row.
class Nv21ToRgb : public Generator<Nv21ToRgb> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // Sample Y at full resolution (use int32 to avoid overflow in BT.601 arithmetic)
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // Sample UV at half resolution from raw byte buffer
        // NV21: V at even offset, U at odd offset within each row
        Expr uv_x = (x / 2) * 2;   // byte offset of the V,U pair
        Expr uv_y = y / 2;          // row in UV plane
        Expr v_val = cast<int32_t>(uv_plane(uv_x, uv_y)) - 128;
        Expr u_val = cast<int32_t>(uv_plane(uv_x + 1, uv_y)) - 128;

        // BT.601 YUV to RGB conversion (fixed-point, shift by 8)
        // Y' = (Y - 16) * 298
        Expr y_scaled = (y_val - 16) * 298 + 128;

        Expr r = (y_scaled + 409 * v_val) >> 8;
        Expr g = (y_scaled - 100 * u_val - 208 * v_val) >> 8;
        Expr b = (y_scaled + 516 * u_val) >> 8;

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

HALIDE_REGISTER_GENERATOR(Nv21ToRgb, nv21_to_rgb)
