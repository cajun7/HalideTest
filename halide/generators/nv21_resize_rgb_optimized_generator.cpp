// =============================================================================
// Fused NV21 Resize → RGB Optimized Generators
// =============================================================================
//
// These pipelines resize NV21 data (Y + UV planes) and convert to RGB in a
// single fused pass. This avoids the multi-step OpenCV approach:
//   OpenCV: NV21 → RGB → resize → output  (2 full passes + color conversion)
//   Fused:  NV21 → resize(Y) + resize(UV) → BT.601 → RGB  (1 pass, 40% less data)
//
// Three interpolation methods:
//   - Bilinear: 2×2 neighborhood, fastest
//   - INTER_AREA: box-filter downsampling, optimal quality for downscale
//   - Bicubic: 4×4 neighborhood (a=-0.75 matching OpenCV), sharpest
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Fused NV21 → Bilinear Resize → RGB
// ---------------------------------------------------------------------------
class Nv21ResizeRgbBilinearOptimized : public Generator<Nv21ResizeRgbBilinearOptimized> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        Expr src_w = y_plane.dim(0).extent();
        Expr src_h = y_plane.dim(1).extent();
        Expr src_wf = cast<float>(src_w);
        Expr src_hf = cast<float>(src_h);
        Expr twf = cast<float>(target_w);
        Expr thf = cast<float>(target_h);

        // --- Resize Y: hybrid float-coord / integer-interp bilinear ---
        Func y_clamped = repeat_edge(y_plane);

        Expr y_src_x = (cast<float>(x) + 0.5f) * src_wf / twf - 0.5f;
        Expr y_src_y = (cast<float>(y) + 0.5f) * src_hf / thf - 0.5f;
        Expr y_ix = cast<int>(floor(y_src_x));
        Expr y_iy = cast<int>(floor(y_src_y));
        Expr y_fx = cast<int32_t>(clamp((y_src_x - cast<float>(y_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr y_fy = cast<int32_t>(clamp((y_src_y - cast<float>(y_iy)) * 2048.0f, 0.0f, 2048.0f));
        Expr y_ix_s = unsafe_promise_clamped(y_ix, -1, src_w);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, -1, src_h);

        Expr yp00 = cast<int32_t>(y_clamped(y_ix_s, y_iy_s));
        Expr yp10 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s));
        Expr yp01 = cast<int32_t>(y_clamped(y_ix_s, y_iy_s + 1));
        Expr yp11 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s + 1));
        Expr y_top = yp00 * (2048 - y_fx) + yp10 * y_fx;
        Expr y_bot = yp01 * (2048 - y_fx) + yp11 * y_fx;

        Func y_resized("y_resized");
        y_resized(x, y) = (y_top * (2048 - y_fy) + y_bot * y_fy + (1 << 21)) >> 22;

        // --- Resize UV: hybrid float-coord / integer-interp bilinear ---
        Func uv_clamped = repeat_edge(uv_plane);

        Expr uv_src_wf = src_wf / 2.0f;
        Expr uv_src_hf = src_hf / 2.0f;
        Expr uv_dst_wf = twf / 2.0f;
        Expr uv_dst_hf = thf / 2.0f;

        Expr uv_px = x / 2;
        Expr uv_row = y / 2;

        Expr uv_src_px = (cast<float>(uv_px) + 0.5f) * uv_src_wf / uv_dst_wf - 0.5f;
        Expr uv_src_row = (cast<float>(uv_row) + 0.5f) * uv_src_hf / uv_dst_hf - 0.5f;
        Expr uv_ix = cast<int>(floor(uv_src_px));
        Expr uv_iy = cast<int>(floor(uv_src_row));
        Expr uv_fx = cast<int32_t>(clamp((uv_src_px - cast<float>(uv_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr uv_fy = cast<int32_t>(clamp((uv_src_row - cast<float>(uv_iy)) * 2048.0f, 0.0f, 2048.0f));

        Expr uv_w_half = src_w / 2;
        Expr uv_h_dim = uv_plane.dim(1).extent();
        Expr ix0 = unsafe_promise_clamped(uv_ix, 0, uv_w_half - 1);
        Expr ix1 = unsafe_promise_clamped(uv_ix + 1, 0, uv_w_half);
        Expr iy0 = unsafe_promise_clamped(uv_iy, 0, uv_h_dim - 1);
        Expr iy1 = unsafe_promise_clamped(uv_iy + 1, 0, uv_h_dim);

        // V (even bytes) — integer bilinear
        Expr v00 = cast<int32_t>(uv_clamped(ix0 * 2, iy0));
        Expr v10 = cast<int32_t>(uv_clamped(ix1 * 2, iy0));
        Expr v01 = cast<int32_t>(uv_clamped(ix0 * 2, iy1));
        Expr v11 = cast<int32_t>(uv_clamped(ix1 * 2, iy1));
        Expr v_top = v00 * (2048 - uv_fx) + v10 * uv_fx;
        Expr v_bot = v01 * (2048 - uv_fx) + v11 * uv_fx;
        Expr v_val = (v_top * (2048 - uv_fy) + v_bot * uv_fy + (1 << 21)) >> 22;

        // U (odd bytes) — integer bilinear
        Expr u00 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy0));
        Expr u10 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy0));
        Expr u01 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy1));
        Expr u11 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy1));
        Expr u_top = u00 * (2048 - uv_fx) + u10 * uv_fx;
        Expr u_bot = u01 * (2048 - uv_fx) + u11 * uv_fx;
        Expr u_val = (u_top * (2048 - uv_fy) + u_bot * uv_fy + (1 << 21)) >> 22;

        // --- BT.601 limited-range conversion (already integer) ---
        Expr y_int = clamp(y_resized(x, y), 0, 255);
        Expr v_int = clamp(v_val, 0, 255) - 128;
        Expr u_int = clamp(u_val, 0, 255) - 128;

        Expr y_s = (y_int - 16) * 298 + 128;
        Expr r = (y_s + 409 * v_int) >> 8;
        Expr g = (y_s - 100 * u_int - 208 * v_int) >> 8;
        Expr b = (y_s + 516 * u_int) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(mux(c, {r, g, b}), 0, 255));

        // --- Schedule ---
        y_resized.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbBilinearOptimized, nv21_resize_rgb_bilinear_optimized)

// ---------------------------------------------------------------------------
// Fused NV21 → INTER_AREA Resize → RGB
// ---------------------------------------------------------------------------
class Nv21ResizeRgbAreaOptimized : public Generator<Nv21ResizeRgbAreaOptimized> {
public:
    GeneratorParam<int> max_kernel{"max_kernel", 8};

    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mk = max_kernel;

        Expr src_w = cast<float>(y_plane.dim(0).extent());
        Expr src_h = cast<float>(y_plane.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // --- Y plane: separable area resize ---
        Func y_clamped = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped(x, y));

        // Y horizontal
        Expr y_inv_sx = src_w / tw;
        Expr y_sl = cast<float>(x) * src_w / tw;
        Expr y_sr = (cast<float>(x) + 1.0f) * src_w / tw;
        Expr y_bh = cast<int>(floor(y_sl));

        RDom y_rh(0, mk);
        Expr y_spx = y_bh + y_rh.x;
        Expr y_wh = max(min(cast<float>(y_spx) + 1.0f, y_sr) - max(cast<float>(y_spx), y_sl), 0.0f);
        Expr y_irh = y_rh.x < cast<int>(ceil(y_inv_sx)) + 1;

        Func y_hs("y_hs"), y_hw("y_hw");
        y_hs(x, y) = 0.0f; y_hw(x, y) = 0.0f;
        y_hs(x, y) += select(y_irh, y_wh * y_float(y_spx, y), 0.0f);
        y_hw(x, y) += select(y_irh, y_wh, 0.0f);

        Func y_hr("y_hr");
        y_hr(x, y) = y_hs(x, y) / max(y_hw(x, y), 0.0001f);

        // Y vertical
        Expr y_inv_sy = src_h / th;
        Expr y_st = cast<float>(y) * src_h / th;
        Expr y_sb = (cast<float>(y) + 1.0f) * src_h / th;
        Expr y_bv = cast<int>(floor(y_st));

        RDom y_rv(0, mk);
        Expr y_spy = y_bv + y_rv.x;
        Expr y_wv = max(min(cast<float>(y_spy) + 1.0f, y_sb) - max(cast<float>(y_spy), y_st), 0.0f);
        Expr y_irv = y_rv.x < cast<int>(ceil(y_inv_sy)) + 1;

        Func y_vs("y_vs"), y_vw("y_vw");
        y_vs(x, y) = 0.0f; y_vw(x, y) = 0.0f;
        y_vs(x, y) += select(y_irv, y_wv * y_hr(x, y_spy), 0.0f);
        y_vw(x, y) += select(y_irv, y_wv, 0.0f);

        Func y_resized("y_resized");
        y_resized(x, y) = y_vs(x, y) / max(y_vw(x, y), 0.0001f);

        // --- UV plane: area resize at half resolution ---
        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_src_w = src_w / 2.0f;
        Expr uv_src_h = src_h / 2.0f;
        Expr uv_dst_w = tw / 2.0f;
        Expr uv_dst_h = th / 2.0f;
        Expr uv_px_idx = x / 2;

        // UV horizontal (V and U independently)
        Expr uv_inv_sx = uv_src_w / uv_dst_w;
        Expr uv_sl = cast<float>(uv_px_idx) * uv_src_w / uv_dst_w;
        Expr uv_sr = (cast<float>(uv_px_idx) + 1.0f) * uv_src_w / uv_dst_w;
        Expr uv_bh = cast<int>(floor(uv_sl));

        RDom uv_rh(0, mk);
        Expr uv_spx = uv_bh + uv_rh.x;
        Expr uv_wh = max(min(cast<float>(uv_spx) + 1.0f, uv_sr) - max(cast<float>(uv_spx), uv_sl), 0.0f);
        Expr uv_irh = uv_rh.x < cast<int>(ceil(uv_inv_sx)) + 1;

        Func v_hs("v_hs"), v_hw("v_hw"), u_hs("u_hs"), u_hw("u_hw");
        v_hs(x, y) = 0.0f; v_hw(x, y) = 0.0f;
        u_hs(x, y) = 0.0f; u_hw(x, y) = 0.0f;
        v_hs(x, y) += select(uv_irh, uv_wh * uv_float(uv_spx * 2, y), 0.0f);
        v_hw(x, y) += select(uv_irh, uv_wh, 0.0f);
        u_hs(x, y) += select(uv_irh, uv_wh * uv_float(uv_spx * 2 + 1, y), 0.0f);
        u_hw(x, y) += select(uv_irh, uv_wh, 0.0f);

        Func v_hr("v_hr"), u_hr("u_hr");
        v_hr(x, y) = v_hs(x, y) / max(v_hw(x, y), 0.0001f);
        u_hr(x, y) = u_hs(x, y) / max(u_hw(x, y), 0.0001f);

        // UV vertical
        Expr uv_row_idx = y / 2;
        Expr uv_inv_sy = uv_src_h / uv_dst_h;
        Expr uv_st = cast<float>(uv_row_idx) * uv_src_h / uv_dst_h;
        Expr uv_sb_val = (cast<float>(uv_row_idx) + 1.0f) * uv_src_h / uv_dst_h;
        Expr uv_bv = cast<int>(floor(uv_st));

        RDom uv_rv(0, mk);
        Expr uv_spy = uv_bv + uv_rv.x;
        Expr uv_wv = max(min(cast<float>(uv_spy) + 1.0f, uv_sb_val) - max(cast<float>(uv_spy), uv_st), 0.0f);
        Expr uv_irv = uv_rv.x < cast<int>(ceil(uv_inv_sy)) + 1;

        Func v_vs("v_vs"), v_vw("v_vw"), u_vs("u_vs"), u_vw("u_vw");
        v_vs(x, y) = 0.0f; v_vw(x, y) = 0.0f;
        u_vs(x, y) = 0.0f; u_vw(x, y) = 0.0f;
        v_vs(x, y) += select(uv_irv, uv_wv * v_hr(x, uv_spy), 0.0f);
        v_vw(x, y) += select(uv_irv, uv_wv, 0.0f);
        u_vs(x, y) += select(uv_irv, uv_wv * u_hr(x, uv_spy), 0.0f);
        u_vw(x, y) += select(uv_irv, uv_wv, 0.0f);

        Expr v_resized = v_vs(x, y) / max(v_vw(x, y), 0.0001f);
        Expr u_resized = u_vs(x, y) / max(u_vw(x, y), 0.0001f);

        // --- BT.601 conversion ---
        Expr y_int = cast<int32_t>(clamp(y_resized(x, y), 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_resized, 0.0f, 255.0f)) - 128;
        Expr u_int_val = cast<int32_t>(clamp(u_resized, 0.0f, 255.0f)) - 128;

        Expr y_s = (y_int - 16) * 298 + 128;
        Expr r = (y_s + 409 * v_int) >> 8;
        Expr g = (y_s - 100 * u_int_val - 208 * v_int) >> 8;
        Expr b = (y_s + 516 * u_int_val) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(mux(c, {r, g, b}), 0, 255));

        // --- Schedule ---
        y_hr.compute_at(output, yi)
            .vectorize(x, 16, TailStrategy::GuardWithIf);
        y_hs.compute_at(y_hr, x);
        y_hs.update();
        y_hw.compute_at(y_hr, x);
        y_hw.update();

        y_vs.compute_at(y_resized, x);
        y_vs.update();
        y_vw.compute_at(y_resized, x);
        y_vw.update();
        y_resized.compute_at(output, yi);

        v_hr.compute_at(output, yi).vectorize(x, 8, TailStrategy::GuardWithIf);
        u_hr.compute_at(output, yi).vectorize(x, 8, TailStrategy::GuardWithIf);
        v_hs.compute_at(v_hr, x); v_hs.update();
        v_hw.compute_at(v_hr, x); v_hw.update();
        u_hs.compute_at(u_hr, x); u_hs.update();
        u_hw.compute_at(u_hr, x); u_hw.update();

        v_vs.compute_at(output, x); v_vs.update();
        v_vw.compute_at(output, x); v_vw.update();
        u_vs.compute_at(output, x); u_vs.update();
        u_vw.compute_at(output, x); u_vw.update();

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbAreaOptimized, nv21_resize_rgb_area_optimized)

// ---------------------------------------------------------------------------
// Fused NV21 → Bicubic Resize → RGB (a=-0.75 matching OpenCV)
// ---------------------------------------------------------------------------
class Nv21ResizeRgbBicubicOptimized : public Generator<Nv21ResizeRgbBicubicOptimized> {
public:
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

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

        // --- Y bicubic resize ---
        Func y_clamped = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped(x, y));

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

        Expr y_src_y = (cast<float>(y) + 0.5f) * src_h / th - 0.5f;
        Expr y_iy = cast<int>(floor(y_src_y));
        Expr y_fy = y_src_y - cast<float>(y_iy);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, -1, y_plane.dim(1).extent());

        Func y_resized("y_resized");
        Expr y_v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            y_v_val += y_h_interp(x, y_iy_s + dy) * cubic_weight(y_fy - cast<float>(dy));
        }
        y_resized(x, y) = y_v_val;

        // --- UV bicubic resize ---
        Func uv_clamped = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped(x, y));

        Expr uv_src_w = src_w / 2.0f;
        Expr uv_src_h = src_h / 2.0f;
        Expr uv_dst_w = tw / 2.0f;
        Expr uv_dst_h = th / 2.0f;

        Expr uv_dst_px = cast<float>(x / 2);
        Expr uv_dst_row = cast<float>(y / 2);

        Expr uv_src_px = (uv_dst_px + 0.5f) * uv_src_w / uv_dst_w - 0.5f;
        Expr uv_src_row = (uv_dst_row + 0.5f) * uv_src_h / uv_dst_h - 0.5f;

        Expr uv_ix = cast<int>(floor(uv_src_px));
        Expr uv_iy = cast<int>(floor(uv_src_row));
        Expr uv_fx = uv_src_px - cast<float>(uv_ix);
        Expr uv_fy = uv_src_row - cast<float>(uv_iy);

        Expr uv_w_half = y_plane.dim(0).extent() / 2;
        Expr uv_h_dim = uv_plane.dim(1).extent();

        // Horizontal 4-tap for V and U
        Expr v_h_val = cast<float>(0);
        Expr u_h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            Expr cix = clamp(uv_ix + dx, 0, uv_w_half - 1);
            Expr w = cubic_weight(uv_fx - cast<float>(dx));
            v_h_val += uv_float(cix * 2, uv_iy) * w;      // placeholder for vertical
            u_h_val += uv_float(cix * 2 + 1, uv_iy) * w;
        }
        // For separable: need h_interp per row, then vertical pass
        // Simpler approach: direct 2D interpolation for UV (small kernel)

        // Use direct 4x4 interpolation for UV since it's at half resolution
        // and the kernel is small. This avoids the complexity of separable UV.
        Expr v_result = cast<float>(0);
        Expr u_result = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            Expr ciy = clamp(uv_iy + dy, 0, uv_h_dim - 1);
            Expr wy = cubic_weight(uv_fy - cast<float>(dy));
            for (int dx = -1; dx <= 2; dx++) {
                Expr cix = clamp(uv_ix + dx, 0, uv_w_half - 1);
                Expr wx = cubic_weight(uv_fx - cast<float>(dx));
                Expr w = wx * wy;
                v_result += uv_float(cix * 2, ciy) * w;
                u_result += uv_float(cix * 2 + 1, ciy) * w;
            }
        }

        // --- BT.601 conversion ---
        Expr y_int = cast<int32_t>(clamp(y_resized(x, y), 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_result, 0.0f, 255.0f)) - 128;
        Expr u_int_val = cast<int32_t>(clamp(u_result, 0.0f, 255.0f)) - 128;

        Expr y_s = (y_int - 16) * 298 + 128;
        Expr r = (y_s + 409 * v_int) >> 8;
        Expr g = (y_s - 100 * u_int_val - 208 * v_int) >> 8;
        Expr b = (y_s + 516 * u_int_val) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(mux(c, {r, g, b}), 0, 255));

        // --- Schedule ---
        y_h_interp.compute_at(output, yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);
        y_float.compute_at(output, yi);
        y_resized.compute_at(output, yi);
        uv_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbBicubicOptimized, nv21_resize_rgb_bicubic_optimized)
