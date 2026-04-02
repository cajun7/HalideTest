#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Fused NV21 -> Resize -> RGB -> Pad -> Rotate Pipeline
//
// Single Halide generator for ML preprocessing: takes NV21 camera input and
// produces a square RGB output with aspect-ratio-preserving resize, black
// padding (letterbox), and optional rotation.
//
// Target use cases:
//   1920x1080 -> 384x384   (Full HD camera -> ML input)
//   4000x3000 -> 1408x1408 (Samsung 12MP -> ML input)
//
// Operation order (conceptual):
//   NV21 -> Resize (bilinear) -> RGB (full-range BT.601) -> Pad -> Rotate
//
// Implementation: inverse mapping from output pixel back to NV21 source.
//   For each output (x, y):
//   1. Inverse rotate -> (rx, ry) in padded square
//   2. Check if in padding region -> black if yes
//   3. Inverse pad (subtract offset) -> (img_x, img_y) in resized image
//   4. Inverse resize -> (src_x, src_y) in NV21 source
//   5. Bilinear sample Y + UV, apply full-range BT.601
//
// Uses full-range BT.601 coefficients (Samsung/Android Camera default).
//
// rotation_code (GeneratorParam, compile-time):
//   0 = no rotation, 1 = 90 CW, 2 = 180, 3 = 270 CW
// ---------------------------------------------------------------------------
class NV21ResizePadRotate : public Generator<NV21ResizePadRotate> {
public:
    GeneratorParam<int> rotation_code{"rotation_code", 0};

    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};     // src_w x src_h
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};   // src_w x (src_h/2) raw bytes
    Input<int32_t> target_size{"target_size"};         // square output dimension

    Output<Buffer<uint8_t, 3>> output{"output"};       // target_size x target_size x 3

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int code = rotation_code;

        Expr src_w = y_plane.dim(0).extent();
        Expr src_h = y_plane.dim(1).extent();
        Expr src_wf = cast<float>(src_w);
        Expr src_hf = cast<float>(src_h);
        Expr ts = cast<float>(target_size);

        // ================================================================
        // Compute aspect-ratio-preserving resize dimensions
        // ================================================================
        Expr scale_x = ts / src_wf;
        Expr scale_y = ts / src_hf;
        Expr uniform_scale = min(scale_x, scale_y);

        Expr scaled_w = cast<int>(round(src_wf * uniform_scale));
        Expr scaled_h = cast<int>(round(src_hf * uniform_scale));
        Expr scaled_wf = cast<float>(scaled_w);
        Expr scaled_hf = cast<float>(scaled_h);

        // Centering offsets for padding
        Expr pad_x = (target_size - scaled_w) / 2;
        Expr pad_y = (target_size - scaled_h) / 2;
        Expr pad_xf = cast<float>(pad_x);
        Expr pad_yf = cast<float>(pad_y);

        // ================================================================
        // Step 1: Inverse rotation — output pixel to padded-square coord
        // ================================================================
        // The output is target_size x target_size regardless of rotation.
        // After rotation, the padded square is still target_size x target_size.
        Expr rx, ry;
        if (code == 0) {
            rx = cast<float>(x);
            ry = cast<float>(y);
        } else if (code == 1) {
            // Inverse of 90 CW: (x,y) -> (y, ts-1-x)
            rx = cast<float>(y);
            ry = ts - 1.0f - cast<float>(x);
        } else if (code == 2) {
            // Inverse of 180: (x,y) -> (ts-1-x, ts-1-y)
            rx = ts - 1.0f - cast<float>(x);
            ry = ts - 1.0f - cast<float>(y);
        } else {
            // Inverse of 270 CW: (x,y) -> (ts-1-y, x)
            rx = ts - 1.0f - cast<float>(y);
            ry = cast<float>(x);
        }

        // ================================================================
        // Step 2: Check if in image region (not padding)
        // ================================================================
        Expr in_region = (rx >= pad_xf) && (rx < pad_xf + scaled_wf) &&
                         (ry >= pad_yf) && (ry < pad_yf + scaled_hf);

        // ================================================================
        // Step 3: Inverse pad — get coordinate in resized image
        // ================================================================
        Expr img_x = rx - pad_xf;
        Expr img_y = ry - pad_yf;

        // ================================================================
        // Step 4: Inverse resize — resized image to NV21 source
        // ================================================================
        Expr sx = (img_x + 0.5f) * src_wf / scaled_wf - 0.5f;
        Expr sy = (img_y + 0.5f) * src_hf / scaled_hf - 0.5f;

        // Clamp to valid source range
        Expr sx_clamped = clamp(sx, 0.0f, src_wf - 1.0f);
        Expr sy_clamped = clamp(sy, 0.0f, src_hf - 1.0f);

        // ================================================================
        // Step 5: Bilinear sample Y at full resolution
        // ================================================================
        Func y_clamped = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped(x, y));

        Expr y_ix = cast<int>(floor(sx_clamped));
        Expr y_iy = cast<int>(floor(sy_clamped));
        Expr y_fx = sx_clamped - cast<float>(y_ix);
        Expr y_fy = sy_clamped - cast<float>(y_iy);

        Expr y_ix_s = unsafe_promise_clamped(y_ix, 0, src_w - 1);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, 0, src_h - 1);

        Expr y_val = y_float(y_ix_s, y_iy_s) * (1.0f - y_fx) * (1.0f - y_fy) +
                     y_float(y_ix_s + 1, y_iy_s) * y_fx * (1.0f - y_fy) +
                     y_float(y_ix_s, y_iy_s + 1) * (1.0f - y_fx) * y_fy +
                     y_float(y_ix_s + 1, y_iy_s + 1) * y_fx * y_fy;

        // ================================================================
        // Step 6: Bilinear sample UV at half resolution
        // ================================================================
        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_sx = sx_clamped / 2.0f;
        Expr uv_sy = sy_clamped / 2.0f;

        Expr uv_ix = cast<int>(floor(uv_sx));
        Expr uv_iy = cast<int>(floor(uv_sy));
        Expr uv_fx = uv_sx - cast<float>(uv_ix);
        Expr uv_fy = uv_sy - cast<float>(uv_iy);

        // Clamp UV pixel indices to valid range before computing byte offsets.
        // This avoids repeat_edge clamping byte offsets, which could pick up
        // U instead of V (or vice versa) at the right edge of the UV plane.
        Expr uv_w_half = uv_plane.dim(0).extent() / 2;
        Expr uv_h_dim = uv_plane.dim(1).extent();
        Expr uv_ix0 = clamp(uv_ix, 0, uv_w_half - 1);
        Expr uv_ix1 = clamp(uv_ix + 1, 0, uv_w_half - 1);
        Expr uv_iy0 = clamp(uv_iy, 0, uv_h_dim - 1);
        Expr uv_iy1 = clamp(uv_iy + 1, 0, uv_h_dim - 1);

        // V at even byte offsets within each UV row
        Expr v00 = uv_float(uv_ix0 * 2, uv_iy0);
        Expr v10 = uv_float(uv_ix1 * 2, uv_iy0);
        Expr v01 = uv_float(uv_ix0 * 2, uv_iy1);
        Expr v11 = uv_float(uv_ix1 * 2, uv_iy1);
        Expr v_interp = v00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        v10 * uv_fx * (1.0f - uv_fy) +
                        v01 * (1.0f - uv_fx) * uv_fy +
                        v11 * uv_fx * uv_fy;

        // U at odd byte offsets within each UV row
        Expr u00 = uv_float(uv_ix0 * 2 + 1, uv_iy0);
        Expr u10 = uv_float(uv_ix1 * 2 + 1, uv_iy0);
        Expr u01 = uv_float(uv_ix0 * 2 + 1, uv_iy1);
        Expr u11 = uv_float(uv_ix1 * 2 + 1, uv_iy1);
        Expr u_interp = u00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        u10 * uv_fx * (1.0f - uv_fy) +
                        u01 * (1.0f - uv_fx) * uv_fy +
                        u11 * uv_fx * uv_fy;

        // ================================================================
        // Step 7: Full-range BT.601 YUV to RGB
        // ================================================================
        Expr y_int = cast<int32_t>(clamp(y_val, 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_interp, 0.0f, 255.0f)) - 128;
        Expr u_int = cast<int32_t>(clamp(u_interp, 0.0f, 255.0f)) - 128;

        // Full-range: no Y-16 offset
        Expr y_scaled = y_int * 256 + 128;
        Expr r = (y_scaled + 359 * v_int) >> 8;
        Expr g = (y_scaled - 88 * u_int - 183 * v_int) >> 8;
        Expr b = (y_scaled + 454 * u_int) >> 8;

        // ================================================================
        // Output: image pixel or black padding
        // ================================================================
        Expr r_clamped = cast<uint8_t>(clamp(r, 0, 255));
        Expr g_clamped = cast<uint8_t>(clamp(g, 0, 255));
        Expr b_clamped = cast<uint8_t>(clamp(b, 0, 255));

        output(x, y, c) = select(in_region,
            cast<uint8_t>(clamp(mux(c, {r, g, b}), 0, 255)),
            cast<uint8_t>(0));

        // ================================================================
        // Schedule
        // ================================================================
        y_float.compute_at(output, yi);
        uv_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        // Output: interleaved RGB
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(NV21ResizePadRotate, nv21_resize_pad_rotate)
