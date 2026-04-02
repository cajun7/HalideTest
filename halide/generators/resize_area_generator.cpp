// =============================================================================
// INTER_AREA Resize Generator (Box-Filter Area-Based Downsampling)
// =============================================================================
//
// The optimal method for DOWNSCALING images. Each output pixel is the weighted
// average of all source pixels whose footprint overlaps it.
//
// ## How INTER_AREA Works
//
// For an output pixel at position x with scale factor s (output/input ratio):
//   - The source footprint is [x/s, (x+1)/s] — a continuous interval
//   - Source pixels that partially overlap get proportional weight
//
// Example: 2x downscale (s=0.5), output pixel x=1:
//   Source footprint: [1/0.5, 2/0.5] = [2.0, 4.0]
//   Source pixel 2: fully inside -> weight = 1.0
//   Source pixel 3: fully inside -> weight = 1.0
//   Total weight = 2.0, result = (src[2] + src[3]) / 2.0
//
// Example: 3x downscale (s=0.333), output pixel x=0:
//   Source footprint: [0, 3.0]
//   Source pixels 0,1,2: all fully inside -> weight = 1.0 each
//   Total weight = 3.0, result = (src[0] + src[1] + src[2]) / 3.0
//
// For non-integer ratios, edge pixels get fractional weights (partial overlap).
//
// ## Separable 2-Pass Implementation
//
// Like Gaussian blur, the 2D box filter is separable: the 2D weight is the
// product of 1D horizontal and vertical weights. We compute:
//   1. Horizontal pass: weighted average along x for each source row
//   2. Vertical pass: weighted average along y over horizontal results
//
// This reduces O(kernel_w * kernel_h) to O(kernel_w + kernel_h) per pixel.
//
// ## max_kernel GeneratorParam
//
// Bounds the RDom (reduction domain) at compile time. Default 8 supports
// up to 8x downscale. For larger downscale ratios, increase this value.
// The guard condition (in_range) prevents accumulating past the actual kernel.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class ResizeArea : public Generator<ResizeArea> {
public:
    GeneratorParam<int> max_kernel{"max_kernel", 8};

    Input<Buffer<uint8_t, 3>> input{"input"};   // src width x height x 3
    Input<float> scale_x{"scale_x"};            // output_width / input_width (< 1 for downscale)
    Input<float> scale_y{"scale_y"};            // output_height / input_height
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mk = max_kernel;

        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // ================================================================
        // HORIZONTAL PASS
        // ================================================================
        // For output column x, the source footprint is [x/scale_x, (x+1)/scale_x].
        // inv_sx = 1/scale_x = input_width/output_width (the downscale ratio).
        Expr inv_sx = 1.0f / scale_x;
        Expr src_left_h = cast<float>(x) / scale_x;             // left edge in source
        Expr src_right_h = (cast<float>(x) + 1.0f) / scale_x;  // right edge in source
        Expr base_h = cast<int>(floor(src_left_h));              // first source pixel that might overlap

        // RDom: iterate over candidate source pixels.
        // At most mk pixels can overlap (bounded by max_kernel GeneratorParam).
        RDom rh(0, mk);
        Expr src_px_h = base_h + rh.x;  // current source pixel index

        // Compute overlap of source pixel [src_px, src_px+1] with footprint [src_left, src_right].
        // The overlap width is: min(src_px+1, src_right) - max(src_px, src_left)
        // This is 0 when there's no overlap, and 1.0 for fully-contained pixels.
        Expr overlap_left_h = max(cast<float>(src_px_h), src_left_h);
        Expr overlap_right_h = min(cast<float>(src_px_h) + 1.0f, src_right_h);
        Expr weight_h = max(overlap_right_h - overlap_left_h, 0.0f);

        // Guard: only accumulate if the RDom index is within actual kernel extent.
        // ceil(inv_sx) + 1 is the maximum number of source pixels that can overlap.
        Expr in_range_h = rh.x < cast<int>(ceil(inv_sx)) + 1;

        // Accumulate weighted sum and weight sum.
        // h_sum: sum of (weight * pixel_value) per output column
        // h_wsum: sum of weights per output column (for normalization)
        Func h_sum("h_sum"), h_wsum("h_wsum");
        h_sum(x, y, c) = 0.0f;
        h_wsum(x, y) = 0.0f;
        h_sum(x, y, c) += select(in_range_h, weight_h * as_float(src_px_h, y, c), 0.0f);
        h_wsum(x, y) += select(in_range_h, weight_h, 0.0f);

        // Normalize: divide weighted sum by total weight.
        // max(..., 0.0001f) prevents division by zero.
        Func h_result("h_result");
        h_result(x, y, c) = h_sum(x, y, c) / max(h_wsum(x, y), 0.0001f);

        // ================================================================
        // VERTICAL PASS (same algorithm, applied to h_result along y-axis)
        // ================================================================
        Expr inv_sy = 1.0f / scale_y;
        Expr src_top_v = cast<float>(y) / scale_y;
        Expr src_bot_v = (cast<float>(y) + 1.0f) / scale_y;
        Expr base_v = cast<int>(floor(src_top_v));

        RDom rv(0, mk);
        Expr src_py_v = base_v + rv.x;
        Expr overlap_top_v = max(cast<float>(src_py_v), src_top_v);
        Expr overlap_bot_v = min(cast<float>(src_py_v) + 1.0f, src_bot_v);
        Expr weight_v = max(overlap_bot_v - overlap_top_v, 0.0f);
        Expr in_range_v = rv.x < cast<int>(ceil(inv_sy)) + 1;

        Func v_sum("v_sum"), v_wsum("v_wsum");
        v_sum(x, y, c) = 0.0f;
        v_wsum(x, y) = 0.0f;
        v_sum(x, y, c) += select(in_range_v, weight_v * h_result(x, src_py_v, c), 0.0f);
        v_wsum(x, y) += select(in_range_v, weight_v, 0.0f);

        output(x, y, c) = cast<uint8_t>(clamp(
            v_sum(x, y, c) / max(v_wsum(x, y), 0.0001f),
            0.0f, 255.0f));

        // ================================================================
        // SCHEDULE
        // ================================================================
        // h_result: computed per output y-tile strip.
        h_result.compute_at(output, yi)
                .reorder(c, x, y)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        // h_sum/h_wsum: computed per pixel of h_result.
        // The update (reduction) step is scheduled with channels unrolled.
        h_sum.compute_at(h_result, x)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c);
        h_sum.update()
             .reorder(c, x, rh.x, y)
             .unroll(c);

        h_wsum.compute_at(h_result, x);
        h_wsum.update();

        // v_sum/v_wsum: computed per output pixel.
        v_sum.compute_at(output, x)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c);
        v_sum.update()
             .reorder(c, x, rv.x, y)
             .unroll(c);

        v_wsum.compute_at(output, x);
        v_wsum.update();

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Interleaved layout constraints
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeArea, resize_area)

// ---------------------------------------------------------------------------
// INTER_AREA Resize — Target-size variant
// ---------------------------------------------------------------------------
// Takes explicit target width/height instead of scale factors.
// Same separable 2-pass box filter algorithm with direct coordinate mapping.
class ResizeAreaTarget : public Generator<ResizeAreaTarget> {
public:
    GeneratorParam<int> max_kernel{"max_kernel", 8};

    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mk = max_kernel;

        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        Expr src_w = cast<float>(input.dim(0).extent());
        Expr src_h = cast<float>(input.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // --- Horizontal pass ---
        // inv_sx = src_w / tw = how many source pixels per output pixel (horizontally)
        // Source footprint per output column: [x * src_w/tw, (x+1) * src_w/tw]
        Expr inv_sx = src_w / tw;
        Expr src_left_h = cast<float>(x) * src_w / tw;
        Expr src_right_h = (cast<float>(x) + 1.0f) * src_w / tw;
        Expr base_h = cast<int>(floor(src_left_h));

        RDom rh(0, mk);
        Expr src_px_h = base_h + rh.x;
        Expr overlap_left_h = max(cast<float>(src_px_h), src_left_h);
        Expr overlap_right_h = min(cast<float>(src_px_h) + 1.0f, src_right_h);
        Expr weight_h = max(overlap_right_h - overlap_left_h, 0.0f);
        Expr in_range_h = rh.x < cast<int>(ceil(inv_sx)) + 1;

        Func h_sum("h_sum"), h_wsum("h_wsum");
        h_sum(x, y, c) = 0.0f;
        h_wsum(x, y) = 0.0f;
        h_sum(x, y, c) += select(in_range_h, weight_h * as_float(src_px_h, y, c), 0.0f);
        h_wsum(x, y) += select(in_range_h, weight_h, 0.0f);

        Func h_result("h_result");
        h_result(x, y, c) = h_sum(x, y, c) / max(h_wsum(x, y), 0.0001f);

        // --- Vertical pass ---
        Expr inv_sy = src_h / th;
        Expr src_top_v = cast<float>(y) * src_h / th;
        Expr src_bot_v = (cast<float>(y) + 1.0f) * src_h / th;
        Expr base_v = cast<int>(floor(src_top_v));

        RDom rv(0, mk);
        Expr src_py_v = base_v + rv.x;
        Expr overlap_top_v = max(cast<float>(src_py_v), src_top_v);
        Expr overlap_bot_v = min(cast<float>(src_py_v) + 1.0f, src_bot_v);
        Expr weight_v = max(overlap_bot_v - overlap_top_v, 0.0f);
        Expr in_range_v = rv.x < cast<int>(ceil(inv_sy)) + 1;

        Func v_sum("v_sum"), v_wsum("v_wsum");
        v_sum(x, y, c) = 0.0f;
        v_wsum(x, y) = 0.0f;
        v_sum(x, y, c) += select(in_range_v, weight_v * h_result(x, src_py_v, c), 0.0f);
        v_wsum(x, y) += select(in_range_v, weight_v, 0.0f);

        output(x, y, c) = cast<uint8_t>(clamp(
            v_sum(x, y, c) / max(v_wsum(x, y), 0.0001f),
            0.0f, 255.0f));

        // --- Schedule (same as ResizeArea) ---
        h_result.compute_at(output, yi)
                .reorder(c, x, y)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        h_sum.compute_at(h_result, x)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c);
        h_sum.update()
             .reorder(c, x, rh.x, y)
             .unroll(c);

        h_wsum.compute_at(h_result, x);
        h_wsum.update();

        v_sum.compute_at(output, x)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c);
        v_sum.update()
             .reorder(c, x, rv.x, y)
             .unroll(c);

        v_wsum.compute_at(output, x);
        v_wsum.update();

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeAreaTarget, resize_area_target)
