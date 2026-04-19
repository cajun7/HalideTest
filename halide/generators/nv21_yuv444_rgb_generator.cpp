// =============================================================================
// NV21 to RGB via YUV444 Upsampling Generator
// =============================================================================
//
// Converts NV21 to RGB with bilinear-interpolated UV channels.
//
// ## Why This Generator Exists
//
// The basic nv21_to_rgb_generator.cpp uses nearest-neighbor UV lookup:
// each 2x2 Y block shares exactly one UV sample, causing blocky chroma
// transitions (visible as color stepping on diagonal edges).
//
// This generator bilinearly interpolates the UV plane from half-resolution
// to full-resolution before the BT.601 conversion, producing smooth chroma
// transitions at the cost of additional computation.
//
// ## UV Sample Center Geometry
//
// In NV21 4:2:0, each UV sample at half-res position (i, j) represents the
// center of a 2x2 Y block. In full-resolution coordinates, the UV sample
// center is at (2*i + 0.5, 2*j + 0.5). For a full-res pixel at (x, y):
//
//   uv_x = (x + 0.5) / 2.0 - 0.5 = x / 2.0
//   uv_y = (y + 0.5) / 2.0 - 0.5 = y / 2.0
//
// The fractional part drives the bilinear interpolation weights.
//
// ## Critical UV Clamping Pattern
//
// When bilinearly sampling the UV plane, we must clamp PIXEL INDICES before
// converting to byte offsets. If we clamped byte offsets directly, repeat_edge
// might clamp an odd index (U position) to an even index (V position) or
// vice versa, causing V/U channel mixup at image edges.
//
// Uses BT.601 limited-range coefficients (same as nv21_to_rgb_generator.cpp).
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class Nv21Yuv444Rgb : public Generator<Nv21Yuv444Rgb> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        // Y at full resolution (same as basic NV21 converter)
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // Bilinear-interpolate V and U from half-res UV plane.
        //
        // repeat_edge: boundary condition that clamps out-of-bounds coordinates
        // to the nearest edge pixel. This is safer than letting Halide read
        // garbage data outside the buffer.
        Func uv_clamped = repeat_edge(uv_plane);

        // Convert UV bytes to float for interpolation arithmetic.
        // This intermediate Func will be scheduled (compute_at) to avoid
        // redundant uint8->float conversions.
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        // Map full-res pixel coordinate to half-res UV coordinate.
        // See "UV Sample Center Geometry" in the header comment.
        Expr uv_sx = cast<float>(x) / 2.0f;
        Expr uv_sy = cast<float>(y) / 2.0f;

        // Integer and fractional parts for bilinear interpolation.
        Expr uv_ix = cast<int>(floor(uv_sx));
        Expr uv_iy = cast<int>(floor(uv_sy));
        Expr uv_fx = uv_sx - cast<float>(uv_ix);  // horizontal weight
        Expr uv_fy = uv_sy - cast<float>(uv_iy);  // vertical weight

        // *** CRITICAL: Clamp UV PIXEL indices before computing byte offsets ***
        //
        // The UV plane stores interleaved V,U pairs: V0 U0 V1 U1 ...
        // V is at byte offset (pixel_index * 2), U at (pixel_index * 2 + 1).
        //
        // If we didn't clamp pixel indices and instead relied on repeat_edge
        // to clamp the final byte offset, the boundary condition might map
        // a V byte offset to a U byte offset (or vice versa). For example:
        //   - pixel_index = -1 -> byte_offset = -2
        //   - repeat_edge clamps byte_offset to 0 -> reads V(0), correct!
        //   - But byte_offset = -1 -> clamps to 0 -> reads V(0), WRONG (should be U)
        //
        // By clamping pixel indices first, we guarantee V reads always get V
        // and U reads always get U.
        Expr uv_w_half = uv_plane.dim(0).extent() / 2;  // number of UV pixel pairs
        Expr uv_h_dim = uv_plane.dim(1).extent();        // UV plane height
        Expr ix0 = clamp(uv_ix, 0, uv_w_half - 1);      // left pixel index
        Expr ix1 = clamp(uv_ix + 1, 0, uv_w_half - 1);  // right pixel index
        Expr iy0 = clamp(uv_iy, 0, uv_h_dim - 1);       // top row index
        Expr iy1 = clamp(uv_iy + 1, 0, uv_h_dim - 1);   // bottom row index

        // Bilinear interpolation of V (at even byte offsets: pixel_index * 2)
        Expr v00 = uv_float(ix0 * 2, iy0);  // top-left V
        Expr v10 = uv_float(ix1 * 2, iy0);  // top-right V
        Expr v01 = uv_float(ix0 * 2, iy1);  // bottom-left V
        Expr v11 = uv_float(ix1 * 2, iy1);  // bottom-right V
        Expr v_interp = v00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        v10 * uv_fx * (1.0f - uv_fy) +
                        v01 * (1.0f - uv_fx) * uv_fy +
                        v11 * uv_fx * uv_fy;

        // Bilinear interpolation of U (at odd byte offsets: pixel_index * 2 + 1)
        Expr u00 = uv_float(ix0 * 2 + 1, iy0);
        Expr u10 = uv_float(ix1 * 2 + 1, iy0);
        Expr u01 = uv_float(ix0 * 2 + 1, iy1);
        Expr u11 = uv_float(ix1 * 2 + 1, iy1);
        Expr u_interp = u00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        u10 * uv_fx * (1.0f - uv_fy) +
                        u01 * (1.0f - uv_fx) * uv_fy +
                        u11 * uv_fx * uv_fy;

        // BT.601 limited-range conversion (same as nv21_to_rgb_generator.cpp)
        Expr v_int = cast<int32_t>(clamp(v_interp, 0.0f, 255.0f)) - 128;
        Expr u_int = cast<int32_t>(clamp(u_interp, 0.0f, 255.0f)) - 128;

        Expr y_scaled = (y_val - 16) * 298 + 128;
        Expr r = (y_scaled + 409 * v_int) >> 8;
        Expr g = (y_scaled - 100 * u_int - 208 * v_int) >> 8;
        Expr b = (y_scaled + 516 * u_int) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));

        // Schedule
        //
        // compute_at(output, yi): Compute uv_float values within each output
        // tile (32-row strip). This means uv_float values are reused across
        // the 2x2 Y blocks within a tile, but freed between tiles.
        uv_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        // Output is 3-channel interleaved RGB (stride: x=3, c=1).
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21Yuv444Rgb, nv21_yuv444_rgb)
