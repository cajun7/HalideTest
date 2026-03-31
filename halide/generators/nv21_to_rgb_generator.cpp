#include "Halide.h"

using namespace Halide;

// NV21 to RGB conversion using BT.601 coefficients (fixed-point integer arithmetic).
//
// NV21 memory layout:
//   Y plane:  width x height (full resolution), uint8
//   UV plane: width x (height/2) bytes, interleaved V,U pairs at half resolution
//             For each 2x2 pixel block: one V byte, one U byte
//             Stored as: V0 U0 V1 U1 ... (width/2 pairs per row, height/2 rows)
//
// We model the UV plane as a 3D buffer: (width/2) x (height/2) x 2 (interleaved)
// where channel 0 = V, channel 1 = U.
class Nv21ToRgb : public Generator<Nv21ToRgb> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 3>> uv_plane{"uv_plane"};  // (width/2) x (height/2) x 2

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // Sample Y at full resolution
        Expr y_val = cast<int16_t>(y_plane(x, y));

        // Sample UV at half resolution (NV21: channel 0 = V, channel 1 = U)
        Expr v_val = cast<int16_t>(uv_plane(x / 2, y / 2, 0)) - 128;
        Expr u_val = cast<int16_t>(uv_plane(x / 2, y / 2, 1)) - 128;

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
        uv_plane.dim(0).set_stride(2);  // interleaved VU
        uv_plane.dim(2).set_bounds(0, 2);
        uv_plane.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ToRgb, nv21_to_rgb)
