// =============================================================================
// Fused NV21 → Resize → RGB (BT.709 Full-Range)
// =============================================================================
//
// Three interpolation variants emitted as three separate AOT pipelines:
//   - Nearest  (OpenCV INTER_NEAREST convention: sx = floor(dst_x * src_w / dst_w))
//   - Bilinear (Q11 fixed-point integer, matches nv21_resize_rgb_bilinear_optimized)
//   - Area     (separable box filter, matches nv21_resize_rgb_area_optimized)
//
// Color-space tail is BT.709 full-range (Samsung Camera2 HD+ default), Q8:
//     R = clamp_u8((y_scaled + 403 * (V-128)) >> 8)
//     G = clamp_u8((y_scaled -  48 * (U-128) - 120 * (V-128)) >> 8)
//     B = clamp_u8((y_scaled + 475 * (U-128)) >> 8)
// with y_scaled = Y*256 + 128 (+128 pre-bias for round-to-nearest shift).
// Coefficients match the standalone Nv21ToRgbBt709FullRange generator AND the
// hand-rolled NEON reference in app/src/main/jni/bt709_neon_ref.cpp — so for
// identical NV21 inputs the non-fused and fused paths agree on color (up to
// the 1-LSB rounding of the resize stage).
//
// Fusion: y_resized / u / v are compute_at(output, yi), eliminating the
// intermediate RGB buffer a 2-stage OpenCV pipeline would materialize.
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

namespace {

// Shared BT.709 full-range Q8 color-space tail. `y_val`, `u_val`, `v_val` are
// already-clamped [0, 255] integer or float Exprs for the three channels.
// `c` is the channel dim Var. Returns an Expr producing a uint8 interleaved
// RGB pixel — the caller assigns `output(x, y, c) = ...`.
Expr bt709_full_range_rgb(Expr y_val, Expr u_val, Expr v_val, Var c) {
    Expr y_i = cast<int32_t>(y_val);
    Expr u_c = cast<int32_t>(u_val) - 128;
    Expr v_c = cast<int32_t>(v_val) - 128;

    Expr y_scaled = y_i * 256 + 128;
    Expr r = (y_scaled + 403 * v_c) >> 8;
    Expr g = (y_scaled -  48 * u_c - 120 * v_c) >> 8;
    Expr b = (y_scaled + 475 * u_c) >> 8;

    return cast<uint8_t>(clamp(mux(c, {r, g, b}), 0, 255));
}

// Schedule constants — "narrow-parallel" from the empirical sweep that won
// the majority of cells across all 3 interps on SM8850 / Exynos 2600. See
// docs/schedule_experiments.md for the comparison data that led here.
//
// - PARALLEL_SPLIT=16: rows per thread task (trades scheduler overhead vs.
//   load balance). 16 beats 32 (old default), 64, and the autoscheduler
//   choices on this SoC's 4-wide big-core cluster.
// - VECTOR_X for nearest/bilinear = 16 (uint8 NEON lane × 1 unroll).
// - VECTOR_X for area             = 8  (float NEON lane; wider saturates).
constexpr int PARALLEL_SPLIT = 16;
constexpr int VECTOR_X_U8    = 16;
constexpr int VECTOR_X_AREA  = 8;

}  // namespace

// ---------------------------------------------------------------------------
// Fused NV21 → Nearest-Neighbor Resize → BT.709 RGB
// OpenCV INTER_NEAREST rule: sx = floor(dst_x * src_w / dst_w).
// ---------------------------------------------------------------------------
class Nv21ResizeRgbBt709Nearest : public Generator<Nv21ResizeRgbBt709Nearest> {
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

        // --- Nearest Y ---
        Expr y_sxf = cast<float>(x) * cast<float>(src_w) / cast<float>(target_w);
        Expr y_syf = cast<float>(y) * cast<float>(src_h) / cast<float>(target_h);
        Expr y_ix  = clamp(cast<int32_t>(floor(y_sxf)), 0, src_w - 1);
        Expr y_iy  = clamp(cast<int32_t>(floor(y_syf)), 0, src_h - 1);
        Expr y_val = y_plane(y_ix, y_iy);

        // --- Nearest UV at chroma resolution ---
        Expr uv_src_w_half = src_w / 2;
        Expr uv_src_h_half = uv_plane.dim(1).extent();
        Expr uv_dst_w_half = target_w / 2;
        Expr uv_dst_h_half = target_h / 2;

        Expr uv_px = x / 2;
        Expr uv_row = y / 2;
        Expr uv_sxf = cast<float>(uv_px)  * cast<float>(uv_src_w_half) / cast<float>(uv_dst_w_half);
        Expr uv_syf = cast<float>(uv_row) * cast<float>(uv_src_h_half) / cast<float>(uv_dst_h_half);
        Expr uv_ix  = clamp(cast<int32_t>(floor(uv_sxf)), 0, uv_src_w_half - 1);
        Expr uv_iy  = clamp(cast<int32_t>(floor(uv_syf)), 0, uv_src_h_half - 1);

        // V at even bytes, U at odd bytes
        Expr v_val = uv_plane(uv_ix * 2,     uv_iy);
        Expr u_val = uv_plane(uv_ix * 2 + 1, uv_iy);

        output(x, y, c) = bt709_full_range_rgb(y_val, u_val, v_val, c);

        // Layout constraints.
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, PARALLEL_SPLIT, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, VECTOR_X_U8, TailStrategy::GuardWithIf);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbBt709Nearest, nv21_resize_rgb_bt709_nearest)

// ---------------------------------------------------------------------------
// Fused NV21 → Bilinear Resize → BT.709 RGB
// Q11 integer weights, pixel-center aligned (matches OpenCV INTER_LINEAR).
// ---------------------------------------------------------------------------
class Nv21ResizeRgbBt709Bilinear : public Generator<Nv21ResizeRgbBt709Bilinear> {
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

        // --- Y bilinear (Q11) ---
        Func y_clamped = repeat_edge(y_plane);

        Expr y_src_x = (cast<float>(x) + 0.5f) * src_wf / twf - 0.5f;
        Expr y_src_y = (cast<float>(y) + 0.5f) * src_hf / thf - 0.5f;
        Expr y_ix = cast<int>(floor(y_src_x));
        Expr y_iy = cast<int>(floor(y_src_y));
        Expr y_fx = cast<int32_t>(clamp((y_src_x - cast<float>(y_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr y_fy = cast<int32_t>(clamp((y_src_y - cast<float>(y_iy)) * 2048.0f, 0.0f, 2048.0f));
        Expr y_ix_s = unsafe_promise_clamped(y_ix, -1, src_w);
        Expr y_iy_s = unsafe_promise_clamped(y_iy, -1, src_h);

        Expr yp00 = cast<int32_t>(y_clamped(y_ix_s,     y_iy_s));
        Expr yp10 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s));
        Expr yp01 = cast<int32_t>(y_clamped(y_ix_s,     y_iy_s + 1));
        Expr yp11 = cast<int32_t>(y_clamped(y_ix_s + 1, y_iy_s + 1));
        Expr y_top = yp00 * (2048 - y_fx) + yp10 * y_fx;
        Expr y_bot = yp01 * (2048 - y_fx) + yp11 * y_fx;

        Func y_resized("y_resized");
        y_resized(x, y) = (y_top * (2048 - y_fy) + y_bot * y_fy + (1 << 21)) >> 22;

        // --- UV bilinear (Q11) at chroma resolution ---
        Func uv_clamped = repeat_edge(uv_plane);

        Expr uv_src_wf = src_wf / 2.0f;
        Expr uv_src_hf = src_hf / 2.0f;
        Expr uv_dst_wf = twf / 2.0f;
        Expr uv_dst_hf = thf / 2.0f;

        Expr uv_px = x / 2;
        Expr uv_row = y / 2;

        Expr uv_src_px  = (cast<float>(uv_px)  + 0.5f) * uv_src_wf / uv_dst_wf - 0.5f;
        Expr uv_src_row = (cast<float>(uv_row) + 0.5f) * uv_src_hf / uv_dst_hf - 0.5f;
        Expr uv_ix = cast<int>(floor(uv_src_px));
        Expr uv_iy = cast<int>(floor(uv_src_row));
        Expr uv_fx = cast<int32_t>(clamp((uv_src_px  - cast<float>(uv_ix)) * 2048.0f, 0.0f, 2048.0f));
        Expr uv_fy = cast<int32_t>(clamp((uv_src_row - cast<float>(uv_iy)) * 2048.0f, 0.0f, 2048.0f));

        Expr uv_w_half = src_w / 2;
        Expr uv_h_dim  = uv_plane.dim(1).extent();
        Expr ix0 = unsafe_promise_clamped(uv_ix,     -1, uv_w_half - 1);
        Expr ix1 = unsafe_promise_clamped(uv_ix + 1,  0, uv_w_half);
        Expr iy0 = unsafe_promise_clamped(uv_iy,     -1, uv_h_dim  - 1);
        Expr iy1 = unsafe_promise_clamped(uv_iy + 1,  0, uv_h_dim);

        Expr v00 = cast<int32_t>(uv_clamped(ix0 * 2,     iy0));
        Expr v10 = cast<int32_t>(uv_clamped(ix1 * 2,     iy0));
        Expr v01 = cast<int32_t>(uv_clamped(ix0 * 2,     iy1));
        Expr v11 = cast<int32_t>(uv_clamped(ix1 * 2,     iy1));
        Expr v_top = v00 * (2048 - uv_fx) + v10 * uv_fx;
        Expr v_bot = v01 * (2048 - uv_fx) + v11 * uv_fx;
        Expr v_val = (v_top * (2048 - uv_fy) + v_bot * uv_fy + (1 << 21)) >> 22;

        Expr u00 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy0));
        Expr u10 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy0));
        Expr u01 = cast<int32_t>(uv_clamped(ix0 * 2 + 1, iy1));
        Expr u11 = cast<int32_t>(uv_clamped(ix1 * 2 + 1, iy1));
        Expr u_top = u00 * (2048 - uv_fx) + u10 * uv_fx;
        Expr u_bot = u01 * (2048 - uv_fx) + u11 * uv_fx;
        Expr u_val = (u_top * (2048 - uv_fy) + u_bot * uv_fy + (1 << 21)) >> 22;

        // BT.709 conversion on resized samples (all three are already [0..255])
        Expr y_clamped_v = clamp(y_resized(x, y), 0, 255);
        Expr v_clamped_v = clamp(v_val, 0, 255);
        Expr u_clamped_v = clamp(u_val, 0, 255);

        output(x, y, c) = bt709_full_range_rgb(y_clamped_v, u_clamped_v, v_clamped_v, c);

        // Layout constraints.
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);

        y_resized.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, PARALLEL_SPLIT, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, VECTOR_X_U8, TailStrategy::GuardWithIf);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbBt709Bilinear, nv21_resize_rgb_bt709_bilinear)

// ---------------------------------------------------------------------------
// Fused NV21 → INTER_AREA Resize → BT.709 RGB
// Separable float box-filter; matches nv21_resize_rgb_area_optimized structure.
// ---------------------------------------------------------------------------
class Nv21ResizeRgbBt709Area : public Generator<Nv21ResizeRgbBt709Area> {
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
        Func y_clamped_f = repeat_edge(y_plane);
        Func y_float("y_float");
        y_float(x, y) = cast<float>(y_clamped_f(x, y));

        Expr y_inv_sx = src_w / tw;
        Expr y_sl = cast<float>(x) * src_w / tw;
        Expr y_sr = (cast<float>(x) + 1.0f) * src_w / tw;
        Expr y_bh = cast<int>(floor(y_sl));

        RDom y_rh(0, mk);
        Expr y_spx = y_bh + y_rh.x;
        Expr y_wh  = max(min(cast<float>(y_spx) + 1.0f, y_sr) - max(cast<float>(y_spx), y_sl), 0.0f);
        Expr y_irh = y_rh.x < cast<int>(ceil(y_inv_sx)) + 1;

        Func y_hs("y_hs"), y_hw("y_hw");
        y_hs(x, y) = 0.0f; y_hw(x, y) = 0.0f;
        y_hs(x, y) += select(y_irh, y_wh * y_float(y_spx, y), 0.0f);
        y_hw(x, y) += select(y_irh, y_wh, 0.0f);

        Func y_hr("y_hr");
        y_hr(x, y) = y_hs(x, y) / max(y_hw(x, y), 0.0001f);

        Expr y_inv_sy = src_h / th;
        Expr y_st = cast<float>(y) * src_h / th;
        Expr y_sb = (cast<float>(y) + 1.0f) * src_h / th;
        Expr y_bv = cast<int>(floor(y_st));

        RDom y_rv(0, mk);
        Expr y_spy = y_bv + y_rv.x;
        Expr y_wv  = max(min(cast<float>(y_spy) + 1.0f, y_sb) - max(cast<float>(y_spy), y_st), 0.0f);
        Expr y_irv = y_rv.x < cast<int>(ceil(y_inv_sy)) + 1;

        Func y_vs("y_vs"), y_vw("y_vw");
        y_vs(x, y) = 0.0f; y_vw(x, y) = 0.0f;
        y_vs(x, y) += select(y_irv, y_wv * y_hr(x, y_spy), 0.0f);
        y_vw(x, y) += select(y_irv, y_wv, 0.0f);

        Func y_resized("y_resized");
        y_resized(x, y) = y_vs(x, y) / max(y_vw(x, y), 0.0001f);

        // --- UV plane: separable area resize at chroma resolution ---
        Func uv_clamped_f = repeat_edge(uv_plane);
        Func uv_float("uv_float");
        uv_float(x, y) = cast<float>(uv_clamped_f(x, y));

        Expr uv_src_w = src_w / 2.0f;
        Expr uv_src_h = src_h / 2.0f;
        Expr uv_dst_w = tw / 2.0f;
        Expr uv_dst_h = th / 2.0f;
        Expr uv_px_idx = x / 2;

        Expr uv_inv_sx = uv_src_w / uv_dst_w;
        Expr uv_sl = cast<float>(uv_px_idx) * uv_src_w / uv_dst_w;
        Expr uv_sr = (cast<float>(uv_px_idx) + 1.0f) * uv_src_w / uv_dst_w;
        Expr uv_bh = cast<int>(floor(uv_sl));

        RDom uv_rh(0, mk);
        Expr uv_spx = uv_bh + uv_rh.x;
        Expr uv_wh  = max(min(cast<float>(uv_spx) + 1.0f, uv_sr) - max(cast<float>(uv_spx), uv_sl), 0.0f);
        Expr uv_irh = uv_rh.x < cast<int>(ceil(uv_inv_sx)) + 1;

        Func v_hs("v_hs"), v_hw("v_hw"), u_hs("u_hs"), u_hw("u_hw");
        v_hs(x, y) = 0.0f; v_hw(x, y) = 0.0f;
        u_hs(x, y) = 0.0f; u_hw(x, y) = 0.0f;
        v_hs(x, y) += select(uv_irh, uv_wh * uv_float(uv_spx * 2,     y), 0.0f);
        v_hw(x, y) += select(uv_irh, uv_wh, 0.0f);
        u_hs(x, y) += select(uv_irh, uv_wh * uv_float(uv_spx * 2 + 1, y), 0.0f);
        u_hw(x, y) += select(uv_irh, uv_wh, 0.0f);

        Func v_hr("v_hr"), u_hr("u_hr");
        v_hr(x, y) = v_hs(x, y) / max(v_hw(x, y), 0.0001f);
        u_hr(x, y) = u_hs(x, y) / max(u_hw(x, y), 0.0001f);

        Expr uv_row_idx = y / 2;
        Expr uv_inv_sy = uv_src_h / uv_dst_h;
        Expr uv_st = cast<float>(uv_row_idx) * uv_src_h / uv_dst_h;
        Expr uv_sb_val = (cast<float>(uv_row_idx) + 1.0f) * uv_src_h / uv_dst_h;
        Expr uv_bv = cast<int>(floor(uv_st));

        RDom uv_rv(0, mk);
        Expr uv_spy = uv_bv + uv_rv.x;
        Expr uv_wv  = max(min(cast<float>(uv_spy) + 1.0f, uv_sb_val) - max(cast<float>(uv_spy), uv_st), 0.0f);
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

        // BT.709 tail (clamp float to [0,255] then quantize)
        Expr y_int = cast<int32_t>(clamp(y_resized(x, y), 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_resized,        0.0f, 255.0f));
        Expr u_int = cast<int32_t>(clamp(u_resized,        0.0f, 255.0f));
        output(x, y, c) = bt709_full_range_rgb(y_int, u_int, v_int, c);

        // Layout constraints.
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);

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
              .split(y, y, yi, PARALLEL_SPLIT, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, VECTOR_X_AREA, TailStrategy::GuardWithIf);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ResizeRgbBt709Area, nv21_resize_rgb_bt709_area)
