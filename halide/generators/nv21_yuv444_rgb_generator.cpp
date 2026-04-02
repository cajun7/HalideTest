#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// NV21 to RGB via YUV444 upsampling: bilinear-interpolate UV from 4:2:0 to
// full resolution before BT.601 conversion.
//
// Compared to nv21_to_rgb_generator.cpp (nearest-neighbor UV lookup), this
// produces smoother chroma transitions at the cost of more computation.
//
// UV sample centers in NV21 4:2:0: each UV sample at half-res position (i, j)
// corresponds to the center of the 2x2 Y block. In full-res coordinates,
// the UV sample center is at (2*i + 0.5, 2*j + 0.5). For a full-res pixel
// at (x, y), the continuous UV coordinate is:
//   uv_x = (x + 0.5) / 2.0 - 0.5 = x / 2.0
//   uv_y = (y + 0.5) / 2.0 - 0.5 = y / 2.0
//
// Uses BT.601 limited-range coefficients (same as nv21_to_rgb_generator.cpp).
class Nv21Yuv444Rgb : public Generator<Nv21Yuv444Rgb> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        // Y at full resolution
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // Bilinear-interpolate V and U from half-res UV plane
        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_sx = cast<float>(x) / 2.0f;
        Expr uv_sy = cast<float>(y) / 2.0f;

        Expr uv_ix = cast<int>(floor(uv_sx));
        Expr uv_iy = cast<int>(floor(uv_sy));
        Expr uv_fx = uv_sx - cast<float>(uv_ix);
        Expr uv_fy = uv_sy - cast<float>(uv_iy);

        // Clamp UV pixel indices to valid range before computing byte offsets.
        // This avoids repeat_edge clamping byte offsets, which could pick up
        // U instead of V (or vice versa) at the right edge of the UV plane.
        Expr uv_w_half = uv_plane.dim(0).extent() / 2;
        Expr uv_h_dim = uv_plane.dim(1).extent();
        Expr ix0 = clamp(uv_ix, 0, uv_w_half - 1);
        Expr ix1 = clamp(uv_ix + 1, 0, uv_w_half - 1);
        Expr iy0 = clamp(uv_iy, 0, uv_h_dim - 1);
        Expr iy1 = clamp(uv_iy + 1, 0, uv_h_dim - 1);

        // V at even byte offsets within each UV row
        Expr v00 = uv_float(ix0 * 2, iy0);
        Expr v10 = uv_float(ix1 * 2, iy0);
        Expr v01 = uv_float(ix0 * 2, iy1);
        Expr v11 = uv_float(ix1 * 2, iy1);
        Expr v_interp = v00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        v10 * uv_fx * (1.0f - uv_fy) +
                        v01 * (1.0f - uv_fx) * uv_fy +
                        v11 * uv_fx * uv_fy;

        // U at odd byte offsets within each UV row
        Expr u00 = uv_float(ix0 * 2 + 1, iy0);
        Expr u10 = uv_float(ix1 * 2 + 1, iy0);
        Expr u01 = uv_float(ix0 * 2 + 1, iy1);
        Expr u11 = uv_float(ix1 * 2 + 1, iy1);
        Expr u_interp = u00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        u10 * uv_fx * (1.0f - uv_fy) +
                        u01 * (1.0f - uv_fx) * uv_fy +
                        u11 * uv_fx * uv_fy;

        // BT.601 limited-range conversion
        Expr v_int = cast<int32_t>(clamp(v_interp, 0.0f, 255.0f)) - 128;
        Expr u_int = cast<int32_t>(clamp(u_interp, 0.0f, 255.0f)) - 128;

        Expr y_scaled = (y_val - 16) * 298 + 128;
        Expr r = (y_scaled + 409 * v_int) >> 8;
        Expr g = (y_scaled - 100 * u_int - 208 * v_int) >> 8;
        Expr b = (y_scaled + 516 * u_int) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));

        // Schedule
        uv_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21Yuv444Rgb, nv21_yuv444_rgb)
