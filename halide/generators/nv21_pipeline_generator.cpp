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

        // Bilinear interpolation of V (at even byte offsets: uv_ix*2)
        Expr v00 = uv_float(uv_ix * 2, uv_iy);
        Expr v10 = uv_float((uv_ix + 1) * 2, uv_iy);
        Expr v01 = uv_float(uv_ix * 2, uv_iy + 1);
        Expr v11 = uv_float((uv_ix + 1) * 2, uv_iy + 1);
        Expr v_interp = v00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                        v10 * uv_fx * (1.0f - uv_fy) +
                        v01 * (1.0f - uv_fx) * uv_fy +
                        v11 * uv_fx * uv_fy;

        // Bilinear interpolation of U (at odd byte offsets: uv_ix*2+1)
        Expr u00 = uv_float(uv_ix * 2 + 1, uv_iy);
        Expr u10 = uv_float((uv_ix + 1) * 2 + 1, uv_iy);
        Expr u01 = uv_float(uv_ix * 2 + 1, uv_iy + 1);
        Expr u11 = uv_float((uv_ix + 1) * 2 + 1, uv_iy + 1);
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
              .split(y, y, yi, 32)
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
// Same transform composition as bilinear variant, but uses non-separable
// area filtering (box filter) for the resize step. Optimal for downscaling.
//
// For each output pixel, computes the 2D source footprint in NV21 space
// and averages all Y/UV pixels within it with overlap weights.
//
// max_pool GeneratorParam bounds the RDom (default 8 = up to 8x downscale).
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

        Expr rotated_wf = (code == 1 || code == 3) ? src_hf : src_wf;
        Expr rotated_hf = (code == 1 || code == 3) ? src_wf : src_hf;

        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // Source footprint in rotated space for output pixel (x, y)
        Expr rot_left  = cast<float>(x) * rotated_wf / tw;
        Expr rot_right = (cast<float>(x) + 1.0f) * rotated_wf / tw;
        Expr rot_top   = cast<float>(y) * rotated_hf / th;
        Expr rot_bot   = (cast<float>(y) + 1.0f) * rotated_hf / th;

        // Apply inverse flip to footprint bounds
        Expr fl, fr, ft, fb;
        // Horizontal flip reverses left/right
        fl = select(flip_code == 1, rotated_wf - rot_right, rot_left);
        fr = select(flip_code == 1, rotated_wf - rot_left, rot_right);
        // Vertical flip reverses top/bottom
        ft = select(flip_code == 2, rotated_hf - rot_bot, rot_top);
        fb = select(flip_code == 2, rotated_hf - rot_top, rot_bot);

        // Apply inverse rotation to get footprint in NV21 source space
        // For fixed rotations, axis-aligned rectangles stay axis-aligned
        Expr nv21_left, nv21_right, nv21_top, nv21_bot;
        if (code == 0) {
            nv21_left = fl; nv21_right = fr;
            nv21_top = ft; nv21_bot = fb;
        } else if (code == 1) {
            // Inverse of 90 CW: sx=fy, sy=src_h-fx
            nv21_left = ft; nv21_right = fb;
            nv21_top = src_hf - fr; nv21_bot = src_hf - fl;
        } else if (code == 2) {
            // Inverse of 180: sx=src_w-fx, sy=src_h-fy
            nv21_left = src_wf - fr; nv21_right = src_wf - fl;
            nv21_top = src_hf - fb; nv21_bot = src_hf - ft;
        } else {
            // Inverse of 270 CW: sx=src_w-fy, sy=fx
            nv21_left = src_wf - fb; nv21_right = src_wf - ft;
            nv21_top = fl; nv21_bot = fr;
        }

        // Clamp footprint to valid source range
        nv21_left = clamp(nv21_left, 0.0f, src_wf);
        nv21_right = clamp(nv21_right, 0.0f, src_wf);
        nv21_top = clamp(nv21_top, 0.0f, src_hf);
        nv21_bot = clamp(nv21_bot, 0.0f, src_hf);

        Expr base_x = cast<int>(floor(nv21_left));
        Expr base_y = cast<int>(floor(nv21_top));

        // Boundary-safe Y plane access
        Func y_clamped = repeat_edge(y_plane);

        // --- Area-average Y over 2D source footprint ---
        RDom r(0, mk, 0, mk);
        Expr px = base_x + r.x;
        Expr py = base_y + r.y;

        // Compute overlap weights
        Expr oleft = max(cast<float>(px), nv21_left);
        Expr oright = min(cast<float>(px) + 1.0f, nv21_right);
        Expr otop = max(cast<float>(py), nv21_top);
        Expr obot = min(cast<float>(py) + 1.0f, nv21_bot);
        Expr w_x = max(oright - oleft, 0.0f);
        Expr w_y = max(obot - otop, 0.0f);
        Expr weight = w_x * w_y;

        // Kernel extent guard
        Expr footprint_w = cast<int>(ceil(nv21_right - nv21_left)) + 1;
        Expr footprint_h = cast<int>(ceil(nv21_bot - nv21_top)) + 1;
        Expr in_range = r.x < footprint_w && r.y < footprint_h;

        Func y_sum("y_sum"), wt_sum("wt_sum");
        y_sum(x, y) = 0.0f;
        wt_sum(x, y) = 0.0f;
        y_sum(x, y) += select(in_range,
            weight * cast<float>(y_clamped(clamp(px, 0, src_w - 1),
                                           clamp(py, 0, src_h - 1))),
            0.0f);
        wt_sum(x, y) += select(in_range, weight, 0.0f);

        Expr y_avg = y_sum(x, y) / max(wt_sum(x, y), 0.0001f);

        // --- Area-average UV at half resolution ---
        // UV footprint is half the Y footprint
        Expr uv_left = nv21_left / 2.0f;
        Expr uv_right = nv21_right / 2.0f;
        Expr uv_top = nv21_top / 2.0f;
        Expr uv_bot = nv21_bot / 2.0f;
        Expr uv_base_x = cast<int>(floor(uv_left));
        Expr uv_base_y = cast<int>(floor(uv_top));

        Func uv_clamped = repeat_edge(uv_plane);

        // UV reduction: smaller footprint since it's half-res
        // max_pool/2 + 1 is sufficient for UV
        int uv_mk = mk / 2 + 2;
        RDom ruv(0, uv_mk, 0, uv_mk);
        Expr uv_px = uv_base_x + ruv.x;
        Expr uv_py = uv_base_y + ruv.y;

        Expr uv_oleft = max(cast<float>(uv_px), uv_left);
        Expr uv_oright = min(cast<float>(uv_px) + 1.0f, uv_right);
        Expr uv_otop = max(cast<float>(uv_py), uv_top);
        Expr uv_obot = min(cast<float>(uv_py) + 1.0f, uv_bot);
        Expr uv_wx = max(uv_oright - uv_oleft, 0.0f);
        Expr uv_wy = max(uv_obot - uv_otop, 0.0f);
        Expr uv_weight = uv_wx * uv_wy;

        Expr uv_fw = cast<int>(ceil(uv_right - uv_left)) + 1;
        Expr uv_fh = cast<int>(ceil(uv_bot - uv_top)) + 1;
        Expr uv_in_range = ruv.x < uv_fw && ruv.y < uv_fh;

        // Clamp UV pixel coords
        Expr uv_w = uv_plane.dim(0).extent();
        Expr uv_h = uv_plane.dim(1).extent();
        Expr uv_px_c = clamp(uv_px, 0, uv_w / 2 - 1);
        Expr uv_py_c = clamp(uv_py, 0, uv_h - 1);

        Func v_sum("v_sum"), u_sum("u_sum"), uv_wt_sum("uv_wt_sum");
        v_sum(x, y) = 0.0f;
        u_sum(x, y) = 0.0f;
        uv_wt_sum(x, y) = 0.0f;
        v_sum(x, y) += select(uv_in_range,
            uv_weight * cast<float>(uv_clamped(uv_px_c * 2, uv_py_c)),
            0.0f);
        u_sum(x, y) += select(uv_in_range,
            uv_weight * cast<float>(uv_clamped(uv_px_c * 2 + 1, uv_py_c)),
            0.0f);
        uv_wt_sum(x, y) += select(uv_in_range, uv_weight, 0.0f);

        Expr v_avg = v_sum(x, y) / max(uv_wt_sum(x, y), 0.0001f);
        Expr u_avg = u_sum(x, y) / max(uv_wt_sum(x, y), 0.0001f);

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

        // --- Schedule ---
        y_sum.compute_at(output, x);
        y_sum.update().reorder(r.x, r.y, x, y);
        wt_sum.compute_at(output, x);
        wt_sum.update();

        v_sum.compute_at(output, x);
        v_sum.update().reorder(ruv.x, ruv.y, x, y);
        u_sum.compute_at(output, x);
        u_sum.update().reorder(ruv.x, ruv.y, x, y);
        uv_wt_sum.compute_at(output, x);
        uv_wt_sum.update();

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Input planes: contiguous rows
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);

        // Output: interleaved RGB (channel stride = 1, x stride = 3)
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(NV21PipelineArea, nv21_pipeline_area)
