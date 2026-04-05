// =============================================================================
// NV21-Domain Resize Optimized Generators (Bilinear, INTER_AREA, Bicubic)
// =============================================================================
//
// Resizes an NV21 image directly in YUV space without converting to RGB first.
// This avoids the NV21→RGB→resize roundtrip, processing ~40% less data:
//   - Y plane resized at full resolution
//   - UV plane resized at half resolution
//
// The UV plane uses interleaved V,U byte pairs:
//   V at even byte offsets (0, 2, 4, ...), U at odd (1, 3, 5, ...)
// Each V,U pair corresponds to a 2x2 block of Y pixels (4:2:0 subsampling).
//
// When resizing UV, V and U channels must be interpolated independently —
// never interpolate across the V/U byte boundary.
//
// All methods use pixel-center alignment matching OpenCV conventions.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// NV21 Bilinear Resize Optimized
// ---------------------------------------------------------------------------
class Nv21ResizeBilinearOptimized : public Generator<Nv21ResizeBilinearOptimized> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // src_w x src_h
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // src_w x (src_h/2)
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};

    Output<Buffer<uint8_t, 2>> y_output{"y_output"};   // target_w x target_h
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"}; // target_w x (target_h/2)

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        // --- Y plane: hybrid float-coord / integer-interp bilinear ---
        // Float for coordinate computation (cheap per pixel, no int64 div)
        // Integer for interpolation (faster than float multiply-accumulate)
        Func y_clamped = repeat_edge(y_plane);

        Expr src_w = y_plane.dim(0).extent();
        Expr src_h = y_plane.dim(1).extent();
        Expr src_wf = cast<float>(src_w);
        Expr src_hf = cast<float>(src_h);
        Expr twf = cast<float>(target_w);
        Expr thf = cast<float>(target_h);

        // Float source coordinates
        Expr y_src_x = (cast<float>(x) + 0.5f) * src_wf / twf - 0.5f;
        Expr y_src_y = (cast<float>(y) + 0.5f) * src_hf / thf - 0.5f;

        Expr y_ix = cast<int>(floor(y_src_x));
        Expr y_iy = cast<int>(floor(y_src_y));

        // Convert fractional part to 11-bit fixed-point for integer interp
        Expr y_fx = cast<int32_t>(clamp((y_src_x - cast<float>(y_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr y_fy = cast<int32_t>(clamp((y_src_y - cast<float>(y_iy)) * 2048.0f, 0.0f, 2048.0f));

        Expr y_ix_s = unsafe_promise_clamped(y_ix, -1, src_w);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, -1, src_h);

        // Integer bilinear interpolation (11-bit weights)
        Expr p00 = cast<int32_t>(y_clamped(y_ix_s, y_iy_s));
        Expr p10 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s));
        Expr p01 = cast<int32_t>(y_clamped(y_ix_s, y_iy_s + 1));
        Expr p11 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s + 1));

        Expr top = p00 * (2048 - y_fx) + p10 * y_fx;
        Expr bot = p01 * (2048 - y_fx) + p11 * y_fx;
        Expr y_val = top * (2048 - y_fy) + bot * y_fy;

        y_output(x, y) = cast<uint8_t>(clamp((y_val + (1 << 21)) >> 22, 0, 255));

        // --- UV plane: bilinear at half resolution ---
        // UV output dimensions: target_w x (target_h/2)
        // UV pixel coordinate maps to a chroma sample covering a 2x2 Y block.
        //
        // For UV output byte at (x_uv, y_uv):
        //   is_v = (x_uv % 2 == 0)  — V at even bytes, U at odd
        //   uv_pixel_index = x_uv / 2
        //   Source UV dimensions: (src_w/2) pixels wide, (src_h/2) rows
        //   Target UV dimensions: (target_w/2) pixels wide, (target_h/2) rows
        Func uv_clamped = repeat_edge(uv_plane);

        // UV plane: hybrid float-coord / integer-interp bilinear
        Expr uv_src_wf = src_wf / 2.0f;
        Expr uv_src_hf = src_hf / 2.0f;
        Expr uv_dst_wf = twf / 2.0f;
        Expr uv_dst_hf = thf / 2.0f;

        Expr uv_px = x / 2;
        Expr is_v = (x % 2) == 0;

        // Float source UV coordinates
        Expr uv_src_px = (cast<float>(uv_px) + 0.5f) * uv_src_wf / uv_dst_wf - 0.5f;
        Expr uv_src_row = (cast<float>(y) + 0.5f) * uv_src_hf / uv_dst_hf - 0.5f;

        Expr uv_ix = cast<int>(floor(uv_src_px));
        Expr uv_iy = cast<int>(floor(uv_src_row));

        // Convert to 11-bit fixed-point for integer interp
        Expr uv_fx = cast<int32_t>(clamp((uv_src_px - cast<float>(uv_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr uv_fy = cast<int32_t>(clamp((uv_src_row - cast<float>(uv_iy)) * 2048.0f, 0.0f, 2048.0f));

        Expr uv_w_half = src_w / 2;
        Expr uv_h_half = uv_plane.dim(1).extent();
        Expr ix0 = unsafe_promise_clamped(uv_ix, 0, uv_w_half - 1);
        Expr ix1 = unsafe_promise_clamped(uv_ix + 1, 0, uv_w_half);
        Expr iy0 = unsafe_promise_clamped(uv_iy, 0, uv_h_half - 1);
        Expr iy1 = unsafe_promise_clamped(uv_iy + 1, 0, uv_h_half);

        // Integer bilinear for V (even byte offsets)
        Expr v00 = cast<int32_t>(uv_clamped(ix0 * 2, iy0));
        Expr v10 = cast<int32_t>(uv_clamped(ix1 * 2, iy0));
        Expr v01 = cast<int32_t>(uv_clamped(ix0 * 2, iy1));
        Expr v11 = cast<int32_t>(uv_clamped(ix1 * 2, iy1));
        Expr v_top = v00 * (2048 - uv_fx) + v10 * uv_fx;
        Expr v_bot = v01 * (2048 - uv_fx) + v11 * uv_fx;
        Expr v_val = (v_top * (2048 - uv_fy) + v_bot * uv_fy + (1 << 21)) >> 22;

        // Integer bilinear for U (odd byte offsets)
        Expr u00 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy0));
        Expr u10 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy0));
        Expr u01 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy1));
        Expr u11 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy1));
        Expr u_top = u00 * (2048 - uv_fx) + u10 * uv_fx;
        Expr u_bot = u01 * (2048 - uv_fx) + u11 * uv_fx;
        Expr u_val = (u_top * (2048 - uv_fy) + u_bot * uv_fy + (1 << 21)) >> 22;

        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, v_val, u_val), 0, 255));

        // --- Schedule ---
        y_output.split(y, y, yi, 64)
                .parallel(y)
                .vectorize(x, 16, TailStrategy::GuardWithIf);
        Var uv_yi("uv_yi");
        uv_output.split(y, y, uv_yi, 32)
                 .parallel(y)
                 .vectorize(x, 8, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        y_output.dim(0).set_stride(1);
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeBilinearOptimized, nv21_resize_bilinear_optimized)

// ---------------------------------------------------------------------------
// NV21 INTER_AREA Resize Optimized (Separable Box Filter)
// ---------------------------------------------------------------------------
class Nv21ResizeAreaOptimized : public Generator<Nv21ResizeAreaOptimized> {
public:
    GeneratorParam<int> max_kernel{"max_kernel", 8};

    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};

    Output<Buffer<uint8_t, 2>> y_output{"y_output"};
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"};

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int mk = max_kernel;

        Expr src_w = cast<float>(y_plane.dim(0).extent());
        Expr src_h = cast<float>(y_plane.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // ===================== Y PLANE (full resolution) =====================

        Func y_clamped = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped(x, y));

        // Horizontal pass
        Expr y_inv_sx = src_w / tw;
        Expr y_src_left = cast<float>(x) * src_w / tw;
        Expr y_src_right = (cast<float>(x) + 1.0f) * src_w / tw;
        Expr y_base_h = cast<int>(floor(y_src_left));

        RDom y_rh(0, mk);
        Expr y_src_px = y_base_h + y_rh.x;
        Expr y_ol = max(cast<float>(y_src_px), y_src_left);
        Expr y_or = min(cast<float>(y_src_px) + 1.0f, y_src_right);
        Expr y_wh = max(y_or - y_ol, 0.0f);
        Expr y_in_range_h = y_rh.x < cast<int>(ceil(y_inv_sx)) + 1;

        Func y_h_sum("y_h_sum"), y_h_wsum("y_h_wsum");
        y_h_sum(x, y) = 0.0f;
        y_h_wsum(x, y) = 0.0f;
        y_h_sum(x, y) += select(y_in_range_h, y_wh * y_float(y_src_px, y), 0.0f);
        y_h_wsum(x, y) += select(y_in_range_h, y_wh, 0.0f);

        Func y_h_result("y_h_result");
        y_h_result(x, y) = y_h_sum(x, y) / max(y_h_wsum(x, y), 0.0001f);

        // Vertical pass
        Expr y_inv_sy = src_h / th;
        Expr y_src_top = cast<float>(y) * src_h / th;
        Expr y_src_bot = (cast<float>(y) + 1.0f) * src_h / th;
        Expr y_base_v = cast<int>(floor(y_src_top));

        RDom y_rv(0, mk);
        Expr y_src_py = y_base_v + y_rv.x;
        Expr y_ot = max(cast<float>(y_src_py), y_src_top);
        Expr y_ob = min(cast<float>(y_src_py) + 1.0f, y_src_bot);
        Expr y_wv = max(y_ob - y_ot, 0.0f);
        Expr y_in_range_v = y_rv.x < cast<int>(ceil(y_inv_sy)) + 1;

        Func y_v_sum("y_v_sum"), y_v_wsum("y_v_wsum");
        y_v_sum(x, y) = 0.0f;
        y_v_wsum(x, y) = 0.0f;
        y_v_sum(x, y) += select(y_in_range_v, y_wv * y_h_result(x, y_src_py), 0.0f);
        y_v_wsum(x, y) += select(y_in_range_v, y_wv, 0.0f);

        y_output(x, y) = cast<uint8_t>(clamp(
            y_v_sum(x, y) / max(y_v_wsum(x, y), 0.0001f), 0.0f, 255.0f));

        // ===================== UV PLANE (half resolution) =====================
        // UV is resized in the chroma pixel domain. V and U are interpolated
        // separately to avoid cross-channel contamination.

        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_src_w = src_w / 2.0f;
        Expr uv_src_h = src_h / 2.0f;
        Expr uv_dst_w = tw / 2.0f;
        Expr uv_dst_h = th / 2.0f;
        Expr uv_px = x / 2;
        Expr is_v = (x % 2) == 0;

        // Horizontal pass for V and U independently
        Expr uv_inv_sx = uv_src_w / uv_dst_w;
        Expr uv_src_left = cast<float>(uv_px) * uv_src_w / uv_dst_w;
        Expr uv_src_right = (cast<float>(uv_px) + 1.0f) * uv_src_w / uv_dst_w;
        Expr uv_base_h = cast<int>(floor(uv_src_left));

        RDom uv_rh(0, mk);
        Expr uv_src_px_h = uv_base_h + uv_rh.x;
        Expr uv_ol_h = max(cast<float>(uv_src_px_h), uv_src_left);
        Expr uv_or_h = min(cast<float>(uv_src_px_h) + 1.0f, uv_src_right);
        Expr uv_wh = max(uv_or_h - uv_ol_h, 0.0f);
        Expr uv_in_range_h = uv_rh.x < cast<int>(ceil(uv_inv_sx)) + 1;

        // V accumulator (even byte offsets: pixel_index * 2)
        Func v_h_sum("v_h_sum"), v_h_wsum("v_h_wsum");
        v_h_sum(x, y) = 0.0f;
        v_h_wsum(x, y) = 0.0f;
        v_h_sum(x, y) += select(uv_in_range_h, uv_wh * uv_float(uv_src_px_h * 2, y), 0.0f);
        v_h_wsum(x, y) += select(uv_in_range_h, uv_wh, 0.0f);

        // U accumulator (odd byte offsets: pixel_index * 2 + 1)
        Func u_h_sum("u_h_sum"), u_h_wsum("u_h_wsum");
        u_h_sum(x, y) = 0.0f;
        u_h_wsum(x, y) = 0.0f;
        u_h_sum(x, y) += select(uv_in_range_h, uv_wh * uv_float(uv_src_px_h * 2 + 1, y), 0.0f);
        u_h_wsum(x, y) += select(uv_in_range_h, uv_wh, 0.0f);

        Func v_h_result("v_h_result"), u_h_result("u_h_result");
        v_h_result(x, y) = v_h_sum(x, y) / max(v_h_wsum(x, y), 0.0001f);
        u_h_result(x, y) = u_h_sum(x, y) / max(u_h_wsum(x, y), 0.0001f);

        // Vertical pass for V and U
        Expr uv_inv_sy = uv_src_h / uv_dst_h;
        Expr uv_src_top = cast<float>(y) * uv_src_h / uv_dst_h;
        Expr uv_src_bot = (cast<float>(y) + 1.0f) * uv_src_h / uv_dst_h;
        Expr uv_base_v = cast<int>(floor(uv_src_top));

        RDom uv_rv(0, mk);
        Expr uv_src_py_v = uv_base_v + uv_rv.x;
        Expr uv_ot_v = max(cast<float>(uv_src_py_v), uv_src_top);
        Expr uv_ob_v = min(cast<float>(uv_src_py_v) + 1.0f, uv_src_bot);
        Expr uv_wv = max(uv_ob_v - uv_ot_v, 0.0f);
        Expr uv_in_range_v = uv_rv.x < cast<int>(ceil(uv_inv_sy)) + 1;

        Func v_v_sum("v_v_sum"), v_v_wsum("v_v_wsum");
        v_v_sum(x, y) = 0.0f;
        v_v_wsum(x, y) = 0.0f;
        v_v_sum(x, y) += select(uv_in_range_v, uv_wv * v_h_result(x, uv_src_py_v), 0.0f);
        v_v_wsum(x, y) += select(uv_in_range_v, uv_wv, 0.0f);

        Func u_v_sum("u_v_sum"), u_v_wsum("u_v_wsum");
        u_v_sum(x, y) = 0.0f;
        u_v_wsum(x, y) = 0.0f;
        u_v_sum(x, y) += select(uv_in_range_v, uv_wv * u_h_result(x, uv_src_py_v), 0.0f);
        u_v_wsum(x, y) += select(uv_in_range_v, uv_wv, 0.0f);

        Expr v_final = v_v_sum(x, y) / max(v_v_wsum(x, y), 0.0001f);
        Expr u_final = u_v_sum(x, y) / max(u_v_wsum(x, y), 0.0001f);

        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, v_final, u_final), 0.0f, 255.0f));

        // --- Schedule ---
        // Y plane
        Var y_yi("y_yi");
        y_h_result.compute_at(y_output, y_yi)
                  .vectorize(x, 16, TailStrategy::GuardWithIf);
        y_h_sum.compute_at(y_h_result, x);
        y_h_sum.update();
        y_h_wsum.compute_at(y_h_result, x);
        y_h_wsum.update();

        y_v_sum.compute_at(y_output, x);
        y_v_sum.update();
        y_v_wsum.compute_at(y_output, x);
        y_v_wsum.update();

        y_output.split(y, y, y_yi, 64)
                .parallel(y)
                .vectorize(x, 16, TailStrategy::GuardWithIf);

        // UV plane
        Var uv_yi("uv_yi");
        v_h_result.compute_at(uv_output, uv_yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);
        u_h_result.compute_at(uv_output, uv_yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);

        v_h_sum.compute_at(v_h_result, x);
        v_h_sum.update();
        v_h_wsum.compute_at(v_h_result, x);
        v_h_wsum.update();
        u_h_sum.compute_at(u_h_result, x);
        u_h_sum.update();
        u_h_wsum.compute_at(u_h_result, x);
        u_h_wsum.update();

        v_v_sum.compute_at(uv_output, x);
        v_v_sum.update();
        v_v_wsum.compute_at(uv_output, x);
        v_v_wsum.update();
        u_v_sum.compute_at(uv_output, x);
        u_v_sum.update();
        u_v_wsum.compute_at(uv_output, x);
        u_v_wsum.update();

        uv_output.split(y, y, uv_yi, 32)
                 .parallel(y)
                 .vectorize(x, 8, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        y_output.dim(0).set_stride(1);
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeAreaOptimized, nv21_resize_area_optimized)

// ---------------------------------------------------------------------------
// NV21 Bicubic Resize Optimized (a=-0.75 matching OpenCV)
// ---------------------------------------------------------------------------
class Nv21ResizeBicubicOptimized : public Generator<Nv21ResizeBicubicOptimized> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};

    Output<Buffer<uint8_t, 2>> y_output{"y_output"};
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"};

    Var x{"x"}, y{"y"}, yi{"yi"};

    // OpenCV cubic kernel (a = -0.75)
    static Expr cubic_weight(Expr t) {
        Expr at = abs(t);
        Expr at2 = at * at;
        Expr at3 = at2 * at;
        return select(
            at <= 1.0f,
            1.25f * at3 - 2.25f * at2 + 1.0f,
            -0.75f * at3 + 3.75f * at2 - 6.0f * at + 3.0f
        );
    }

    void generate() {
        Expr src_w = cast<float>(y_plane.dim(0).extent());
        Expr src_h = cast<float>(y_plane.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // ===================== Y PLANE =====================
        Func y_clamped = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped(x, y));

        // Horizontal 4-tap
        Expr y_src_x = (cast<float>(x) + 0.5f) * src_w / tw - 0.5f;
        Expr y_ix = cast<int>(floor(y_src_x));
        Expr y_fx = y_src_x - cast<float>(y_ix);
        Expr y_ix_s = unsafe_promise_clamped(y_ix, -1, y_plane.dim(0).extent());

        Func y_h_interp("y_h_interp");
        Expr y_h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            y_h_val += y_float(y_ix_s + dx, y) * cubic_weight(y_fx - cast<float>(dx));
        }
        y_h_interp(x, y) = y_h_val;

        // Vertical 4-tap
        Expr y_src_y = (cast<float>(y) + 0.5f) * src_h / th - 0.5f;
        Expr y_iy = cast<int>(floor(y_src_y));
        Expr y_fy = y_src_y - cast<float>(y_iy);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, -1, y_plane.dim(1).extent());

        Expr y_v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            y_v_val += y_h_interp(x, y_iy_s + dy) * cubic_weight(y_fy - cast<float>(dy));
        }
        y_output(x, y) = cast<uint8_t>(clamp(y_v_val, 0.0f, 255.0f));

        // ===================== UV PLANE =====================
        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_src_w = src_w / 2.0f;
        Expr uv_src_h = src_h / 2.0f;
        Expr uv_dst_w = tw / 2.0f;
        Expr uv_dst_h = th / 2.0f;
        Expr uv_px = x / 2;
        Expr is_v = (x % 2) == 0;

        // UV horizontal 4-tap (V and U independently)
        Expr uv_src_px = (cast<float>(uv_px) + 0.5f) * uv_src_w / uv_dst_w - 0.5f;
        Expr uv_ix = cast<int>(floor(uv_src_px));
        Expr uv_fx = uv_src_px - cast<float>(uv_ix);

        Expr uv_w_half = y_plane.dim(0).extent() / 2;
        // Clamp UV pixel indices for 4-tap
        Expr uv_ix_m1 = clamp(uv_ix - 1, 0, uv_w_half - 1);
        Expr uv_ix_0  = clamp(uv_ix,     0, uv_w_half - 1);
        Expr uv_ix_1  = clamp(uv_ix + 1, 0, uv_w_half - 1);
        Expr uv_ix_2  = clamp(uv_ix + 2, 0, uv_w_half - 1);

        Expr v_w_m1 = cubic_weight(uv_fx + 1.0f);
        Expr v_w_0  = cubic_weight(uv_fx);
        Expr v_w_1  = cubic_weight(uv_fx - 1.0f);
        Expr v_w_2  = cubic_weight(uv_fx - 2.0f);

        // Horizontal V interp
        Func v_h_interp("v_h_interp");
        v_h_interp(x, y) = uv_float(uv_ix_m1 * 2, y) * v_w_m1 +
                            uv_float(uv_ix_0 * 2, y)  * v_w_0 +
                            uv_float(uv_ix_1 * 2, y)  * v_w_1 +
                            uv_float(uv_ix_2 * 2, y)  * v_w_2;

        // Horizontal U interp
        Func u_h_interp("u_h_interp");
        u_h_interp(x, y) = uv_float(uv_ix_m1 * 2 + 1, y) * v_w_m1 +
                            uv_float(uv_ix_0 * 2 + 1, y)  * v_w_0 +
                            uv_float(uv_ix_1 * 2 + 1, y)  * v_w_1 +
                            uv_float(uv_ix_2 * 2 + 1, y)  * v_w_2;

        // UV vertical 4-tap
        Expr uv_src_row = (cast<float>(y) + 0.5f) * uv_src_h / uv_dst_h - 0.5f;
        Expr uv_iy = cast<int>(floor(uv_src_row));
        Expr uv_fy = uv_src_row - cast<float>(uv_iy);

        Expr uv_h_half = uv_plane.dim(1).extent();
        Expr uv_iy_m1 = clamp(uv_iy - 1, 0, uv_h_half - 1);
        Expr uv_iy_0  = clamp(uv_iy,     0, uv_h_half - 1);
        Expr uv_iy_1  = clamp(uv_iy + 1, 0, uv_h_half - 1);
        Expr uv_iy_2  = clamp(uv_iy + 2, 0, uv_h_half - 1);

        Expr vy_w_m1 = cubic_weight(uv_fy + 1.0f);
        Expr vy_w_0  = cubic_weight(uv_fy);
        Expr vy_w_1  = cubic_weight(uv_fy - 1.0f);
        Expr vy_w_2  = cubic_weight(uv_fy - 2.0f);

        Expr v_final = v_h_interp(x, uv_iy_m1) * vy_w_m1 +
                       v_h_interp(x, uv_iy_0)  * vy_w_0 +
                       v_h_interp(x, uv_iy_1)  * vy_w_1 +
                       v_h_interp(x, uv_iy_2)  * vy_w_2;

        Expr u_final = u_h_interp(x, uv_iy_m1) * vy_w_m1 +
                       u_h_interp(x, uv_iy_0)  * vy_w_0 +
                       u_h_interp(x, uv_iy_1)  * vy_w_1 +
                       u_h_interp(x, uv_iy_2)  * vy_w_2;

        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, v_final, u_final), 0.0f, 255.0f));

        // --- Schedule ---
        Var y_yi("y_yi");
        y_h_interp.compute_at(y_output, y_yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);
        y_float.compute_at(y_output, y_yi);

        y_output.split(y, y, y_yi, 16)
                .parallel(y)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        Var uv_yi("uv_yi");
        v_h_interp.compute_at(uv_output, uv_yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);
        u_h_interp.compute_at(uv_output, uv_yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);
        uv_float.compute_at(uv_output, uv_yi);

        uv_output.split(y, y, uv_yi, 16)
                 .parallel(y)
                 .vectorize(x, 8, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        y_output.dim(0).set_stride(1);
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeBicubicOptimized, nv21_resize_bicubic_optimized)
