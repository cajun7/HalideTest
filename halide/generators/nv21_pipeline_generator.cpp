#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Fused NV21 -> Rotate -> [Flip] -> Resize (Bilinear) -> RGB Pipeline
//
// Single Halide generator that composes inverse-resize, inverse-flip, and
// inverse-rotation into one coordinate transform. Samples NV21 Y/UV once
// with bilinear interpolation and applies BT.601 to produce RGB.
//
// Benefits:
//   - Single interpolation: no halo/bleeding from double-interpolation
//   - Single memory pass: read NV21 once, write RGB once
//   - Maximum fusion: Halide optimizer schedules the entire pipeline
//
// rotation_code (GeneratorParam, compile-time):
//   0 = no rotation, 1 = 90 CW, 2 = 180, 3 = 270 CW
//
// flip_code (runtime Input):
//   0 = no flip, 1 = horizontal flip, 2 = vertical flip
// ---------------------------------------------------------------------------
class NV21PipelineBilinear : public Generator<NV21PipelineBilinear> {
public:
    GeneratorParam<int> rotation_code{"rotation_code", 0};

    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // src_w x src_h
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // src_w x (src_h/2) raw bytes
    Input<int32_t> flip_code{"flip_code"};            // 0=none, 1=horizontal, 2=vertical
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};      // target_w x target_h x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int code = rotation_code;

        Expr src_w = y_plane.dim(0).extent();
        Expr src_h = y_plane.dim(1).extent();
        Expr src_wf = cast<float>(src_w);
        Expr src_hf = cast<float>(src_h);

        // After rotation, dimensions may swap
        Expr rotated_wf = (code == 1 || code == 3) ? src_hf : src_wf;
        Expr rotated_hf = (code == 1 || code == 3) ? src_wf : src_hf;

        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // Step 1: Inverse resize — output pixel to rotated-space coordinate
        Expr rx = (cast<float>(x) + 0.5f) * rotated_wf / tw - 0.5f;
        Expr ry = (cast<float>(y) + 0.5f) * rotated_hf / th - 0.5f;

        // Step 2: Inverse flip (runtime — near-zero cost, single select)
        Expr fx = select(flip_code == 1, rotated_wf - 1.0f - rx, rx);
        Expr fy = select(flip_code == 2, rotated_hf - 1.0f - ry, ry);

        // Step 3: Inverse rotation — rotated-space to NV21 source coordinate
        Expr sx, sy;
        if (code == 0) {
            sx = fx;
            sy = fy;
        } else if (code == 1) {
            // Inverse of 90 CW: (x,y) -> (y, h-1-x)
            sx = fy;
            sy = src_hf - 1.0f - fx;
        } else if (code == 2) {
            // Inverse of 180: (x,y) -> (w-1-x, h-1-y)
            sx = src_wf - 1.0f - fx;
            sy = src_hf - 1.0f - fy;
        } else {
            // Inverse of 270 CW: (x,y) -> (w-1-y, x)
            sx = src_wf - 1.0f - fy;
            sy = fx;
        }

        // Boundary handling: clamp source coordinates to valid range
        Expr sx_clamped = clamp(sx, 0.0f, src_wf - 1.0f);
        Expr sy_clamped = clamp(sy, 0.0f, src_hf - 1.0f);

        // Step 4: Bilinear sample Y at (sx, sy)
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

        // Step 5: Bilinear sample UV at half resolution
        // NV21: V at even byte offset, U at odd byte offset within each row
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

        // Step 6: BT.601 YUV to RGB conversion (fixed-point, shift by 8)
        Expr y_int = cast<int32_t>(clamp(y_val, 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_interp, 0.0f, 255.0f)) - 128;
        Expr u_int = cast<int32_t>(clamp(u_interp, 0.0f, 255.0f)) - 128;

        Expr y_scaled = (y_int - 16) * 298 + 128;
        Expr r = (y_scaled + 409 * v_int) >> 8;
        Expr g = (y_scaled - 100 * u_int - 208 * v_int) >> 8;
        Expr b = (y_scaled + 516 * u_int) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));

        // Schedule
        y_float.compute_at(output, yi);
        uv_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Input planes: contiguous rows
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        // Output: interleaved RGB (channel stride = 1, x stride = 3)
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(NV21PipelineBilinear, nv21_pipeline_bilinear)

// ---------------------------------------------------------------------------
// Fused NV21 -> Rotate -> [Flip] -> Resize (INTER_AREA) -> RGB Pipeline
//
// Separable 2-pass area filtering: horizontal pass averages along NV21
// x-axis, vertical pass averages along NV21 y-axis. The 2D box-filter
// weight w(col,row) = w_x(col) * w_y(row) factorizes, so separability
// is exact. For fixed rotations the source footprint stays axis-aligned.
//
// Compared to the non-separable version: O(2*mk) vs O(mk^2) per pixel.
//
// max_pool GeneratorParam bounds each 1-D RDom (default 8 = up to 8x).
// ---------------------------------------------------------------------------
class NV21PipelineArea : public Generator<NV21PipelineArea> {
public:
    GeneratorParam<int> rotation_code{"rotation_code", 0};
    GeneratorParam<int> max_pool{"max_pool", 8};

    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};
    Input<int32_t> flip_code{"flip_code"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int code = rotation_code;
        int mk = max_pool;

        Expr src_w = y_plane.dim(0).extent();
        Expr src_h = y_plane.dim(1).extent();
        Expr src_wf = cast<float>(src_w);
        Expr src_hf = cast<float>(src_h);
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        Func y_clamped = repeat_edge(y_plane);
        Func uv_clamped = repeat_edge(uv_plane);
        Expr uv_w_dim = uv_plane.dim(0).extent();
        Expr uv_h_dim = uv_plane.dim(1).extent();

        // Rotated-space dimensions
        Expr rotated_wf = (code == 1 || code == 3) ? src_hf : src_wf;
        Expr rotated_hf = (code == 1 || code == 3) ? src_wf : src_hf;

        // ================================================================
        // NV21 footprint bounds in terms of (x, y) — for final
        // normalization and for the vertical pass.
        // ================================================================
        Expr rot_left  = cast<float>(x) * rotated_wf / tw;
        Expr rot_right = (cast<float>(x) + 1.0f) * rotated_wf / tw;
        Expr rot_top   = cast<float>(y) * rotated_hf / th;
        Expr rot_bot   = (cast<float>(y) + 1.0f) * rotated_hf / th;

        Expr fl = select(flip_code == 1, rotated_wf - rot_right, rot_left);
        Expr fr = select(flip_code == 1, rotated_wf - rot_left, rot_right);
        Expr ft = select(flip_code == 2, rotated_hf - rot_bot, rot_top);
        Expr fb = select(flip_code == 2, rotated_hf - rot_top, rot_bot);

        Expr nv21_left, nv21_right, nv21_top, nv21_bot;
        if (code == 0) {
            nv21_left = fl; nv21_right = fr;
            nv21_top = ft; nv21_bot = fb;
        } else if (code == 1) {
            nv21_left = ft; nv21_right = fb;
            nv21_top = src_hf - fr; nv21_bot = src_hf - fl;
        } else if (code == 2) {
            nv21_left = src_wf - fr; nv21_right = src_wf - fl;
            nv21_top = src_hf - fb; nv21_bot = src_hf - ft;
        } else {
            nv21_left = src_wf - fb; nv21_right = src_wf - ft;
            nv21_top = fl; nv21_bot = fr;
        }
        nv21_left  = clamp(nv21_left, 0.0f, src_wf);
        nv21_right = clamp(nv21_right, 0.0f, src_wf);
        nv21_top   = clamp(nv21_top, 0.0f, src_hf);
        nv21_bot   = clamp(nv21_bot, 0.0f, src_hf);

        // Analytical total weights (no accumulators needed)
        Expr total_h = max(nv21_right - nv21_left, 0.0001f);
        Expr total_v = max(nv21_bot - nv21_top, 0.0001f);

        // ================================================================
        // HORIZONTAL PASS — 1-D area-average along NV21 x-axis
        //
        // h_y(h_idx, sr) = weighted sum of Y along source columns
        //   code 0/2: h_idx = output x  (nv21 x-range depends on x)
        //   code 1/3: h_idx = output y  (nv21 x-range depends on y)
        // ================================================================
        Var h_idx{"h_idx"}, sr{"sr"};
        Expr hf = cast<float>(h_idx);

        // NV21 horizontal bounds as function of h_idx only
        Expr nv21_h_left, nv21_h_right;
        if (code == 0) {
            Expr rl = hf * src_wf / tw;
            Expr rr = (hf + 1.0f) * src_wf / tw;
            nv21_h_left  = select(flip_code == 1, src_wf - rr, rl);
            nv21_h_right = select(flip_code == 1, src_wf - rl, rr);
        } else if (code == 1) {
            Expr rt = hf * src_wf / th;
            Expr rb = (hf + 1.0f) * src_wf / th;
            nv21_h_left  = select(flip_code == 2, src_wf - rb, rt);
            nv21_h_right = select(flip_code == 2, src_wf - rt, rb);
        } else if (code == 2) {
            Expr rl = hf * src_wf / tw;
            Expr rr = (hf + 1.0f) * src_wf / tw;
            Expr fl_h = select(flip_code == 1, src_wf - rr, rl);
            Expr fr_h = select(flip_code == 1, src_wf - rl, rr);
            nv21_h_left  = src_wf - fr_h;
            nv21_h_right = src_wf - fl_h;
        } else { // code == 3
            Expr rt = hf * src_wf / th;
            Expr rb = (hf + 1.0f) * src_wf / th;
            Expr ft_h = select(flip_code == 2, src_wf - rb, rt);
            Expr fb_h = select(flip_code == 2, src_wf - rt, rb);
            nv21_h_left  = src_wf - fb_h;
            nv21_h_right = src_wf - ft_h;
        }
        nv21_h_left  = clamp(nv21_h_left, 0.0f, src_wf);
        nv21_h_right = clamp(nv21_h_right, 0.0f, src_wf);

        Expr h_base = cast<int>(floor(nv21_h_left));
        Expr h_fw = cast<int>(ceil(nv21_h_right - nv21_h_left)) + 1;

        // --- H pass: Y ---
        RDom rh(0, mk);
        Expr h_px = h_base + rh.x;
        Expr h_ol = max(cast<float>(h_px), nv21_h_left);
        Expr h_or = min(cast<float>(h_px) + 1.0f, nv21_h_right);
        Expr h_w  = max(h_or - h_ol, 0.0f);
        Expr h_in = rh.x < h_fw;

        Func h_y("h_y");
        h_y(h_idx, sr) = 0.0f;
        h_y(h_idx, sr) += select(h_in,
            h_w * cast<float>(y_clamped(clamp(h_px, 0, src_w - 1),
                                         clamp(sr, 0, src_h - 1))),
            0.0f);

        // --- H pass: UV (half-res) ---
        Expr uv_h_left  = nv21_h_left / 2.0f;
        Expr uv_h_right = nv21_h_right / 2.0f;
        Expr uv_h_base  = cast<int>(floor(uv_h_left));
        Expr uv_h_fw    = cast<int>(ceil(uv_h_right - uv_h_left)) + 1;

        int uv_mk = mk / 2 + 2;
        RDom rh_uv(0, uv_mk);
        Expr uv_h_px = uv_h_base + rh_uv.x;
        Expr uv_h_ol = max(cast<float>(uv_h_px), uv_h_left);
        Expr uv_h_or = min(cast<float>(uv_h_px) + 1.0f, uv_h_right);
        Expr uv_h_w  = max(uv_h_or - uv_h_ol, 0.0f);
        Expr uv_h_in = rh_uv.x < uv_h_fw;
        Expr uv_h_px_c = clamp(uv_h_px, 0, uv_w_dim / 2 - 1);

        Func h_v("h_v"), h_u("h_u");
        h_v(h_idx, sr) = 0.0f;
        h_u(h_idx, sr) = 0.0f;
        h_v(h_idx, sr) += select(uv_h_in,
            uv_h_w * cast<float>(uv_clamped(uv_h_px_c * 2,
                                             clamp(sr, 0, uv_h_dim - 1))),
            0.0f);
        h_u(h_idx, sr) += select(uv_h_in,
            uv_h_w * cast<float>(uv_clamped(uv_h_px_c * 2 + 1,
                                             clamp(sr, 0, uv_h_dim - 1))),
            0.0f);

        // ================================================================
        // VERTICAL PASS — 1-D area-average along NV21 y-axis
        //
        // nv21_v_top/bot already computed above (depend on x or y
        // depending on rotation, via the full (x,y)-based bounds).
        // ================================================================
        Expr v_base = cast<int>(floor(nv21_top));
        Expr v_fw   = cast<int>(ceil(nv21_bot - nv21_top)) + 1;

        // --- V pass: Y ---
        RDom rv(0, mk);
        Expr v_py_raw = v_base + rv.x;
        // Clamp to valid source range — bounds the h_y allocation and
        // makes the pipeline robust for any downscale factor (even > max_pool).
        Expr v_py = clamp(v_py_raw, 0, src_h - 1);
        Expr v_ol = max(cast<float>(v_py_raw), nv21_top);
        Expr v_ob = min(cast<float>(v_py_raw) + 1.0f, nv21_bot);
        Expr v_w  = max(v_ob - v_ol, 0.0f);
        Expr v_in = rv.x < v_fw;

        // h_y first index: x for code 0/2, y for code 1/3
        Func v_y_sum("v_y_sum");
        v_y_sum(x, y) = 0.0f;
        if (code == 0 || code == 2)
            v_y_sum(x, y) += select(v_in, v_w * h_y(x, v_py), 0.0f);
        else
            v_y_sum(x, y) += select(v_in, v_w * h_y(y, v_py), 0.0f);

        // --- V pass: UV (half-res) ---
        Expr uv_v_top  = nv21_top / 2.0f;
        Expr uv_v_bot  = nv21_bot / 2.0f;
        Expr uv_v_base = cast<int>(floor(uv_v_top));
        Expr uv_v_fw   = cast<int>(ceil(uv_v_bot - uv_v_top)) + 1;

        RDom rv_uv(0, uv_mk);
        Expr uv_v_py_raw = uv_v_base + rv_uv.x;
        Expr uv_v_py = clamp(uv_v_py_raw, 0, uv_h_dim - 1);
        Expr uv_v_ol = max(cast<float>(uv_v_py_raw), uv_v_top);
        Expr uv_v_ob = min(cast<float>(uv_v_py_raw) + 1.0f, uv_v_bot);
        Expr uv_v_w  = max(uv_v_ob - uv_v_ol, 0.0f);
        Expr uv_v_in = rv_uv.x < uv_v_fw;

        Func v_v_sum("v_v_sum"), v_u_sum("v_u_sum");
        v_v_sum(x, y) = 0.0f;
        v_u_sum(x, y) = 0.0f;
        if (code == 0 || code == 2) {
            v_v_sum(x, y) += select(uv_v_in, uv_v_w * h_v(x, uv_v_py), 0.0f);
            v_u_sum(x, y) += select(uv_v_in, uv_v_w * h_u(x, uv_v_py), 0.0f);
        } else {
            v_v_sum(x, y) += select(uv_v_in, uv_v_w * h_v(y, uv_v_py), 0.0f);
            v_u_sum(x, y) += select(uv_v_in, uv_v_w * h_u(y, uv_v_py), 0.0f);
        }

        // ================================================================
        // NORMALIZATION + BT.601
        // ================================================================
        // Total weight = h_extent * v_extent (the 2D box weight factorizes)
        Expr y_area  = total_h * total_v;
        Expr uv_area = max(total_h / 2.0f, 0.0001f)
                     * max(total_v / 2.0f, 0.0001f);

        Expr y_avg = v_y_sum(x, y) / y_area;
        Expr v_avg = v_v_sum(x, y) / uv_area;
        Expr u_avg = v_u_sum(x, y) / uv_area;

        // BT.601 YUV to RGB conversion (fixed-point, shift by 8)
        Expr y_int = cast<int32_t>(clamp(y_avg, 0.0f, 255.0f));
        Expr v_int = cast<int32_t>(clamp(v_avg, 0.0f, 255.0f)) - 128;
        Expr u_int = cast<int32_t>(clamp(u_avg, 0.0f, 255.0f)) - 128;

        Expr y_scaled = (y_int - 16) * 298 + 128;
        Expr r_val = (y_scaled + 409 * v_int) >> 8;
        Expr g_val = (y_scaled - 100 * u_int - 208 * v_int) >> 8;
        Expr b_val = (y_scaled + 516 * u_int) >> 8;

        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r_val, g_val, b_val}), 0, 255));

        // ================================================================
        // SCHEDULE
        // ================================================================
        // Horizontal intermediates: store per-tile, compute per-row.
        // Sliding-window reuse for code 0/2 where h_y is y-invariant.
        h_y.store_at(output, y).compute_at(output, yi);
        h_v.store_at(output, y).compute_at(output, yi);
        h_u.store_at(output, y).compute_at(output, yi);

        // Vertical reductions: per-pixel (inherits output vectorization)
        v_y_sum.compute_at(output, x);
        v_y_sum.update();
        v_v_sum.compute_at(output, x);
        v_v_sum.update();
        v_u_sum.compute_at(output, x);
        v_u_sum.update();

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Input planes: contiguous rows
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        // Output: interleaved RGB (channel stride = 1, x stride = 3)
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(NV21PipelineArea, nv21_pipeline_area)
