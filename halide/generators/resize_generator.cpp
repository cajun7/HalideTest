// =============================================================================
// Resize Generators (Bilinear and Bicubic / Catmull-Rom)
// =============================================================================
//
// ## Pixel-Center Alignment
//
// All resize generators use pixel-center alignment for coordinate mapping:
//   src = (out + 0.5) / scale - 0.5      (scale-factor variant)
//   src = (out + 0.5) * src_dim / target_dim - 0.5  (target-size variant)
//
// This formula maps pixel centers to pixel centers (not pixel edges).
// Example: scaling 4 pixels to 2 pixels:
//   out=0 -> src = (0.5)/0.5 - 0.5 = 0.5    (between input pixels 0 and 1)
//   out=1 -> src = (1.5)/0.5 - 0.5 = 2.5    (between input pixels 2 and 3)
// This is the same convention as OpenCV INTER_LINEAR.
//
// ## unsafe_promise_clamped
//
// An optimization hint that tells Halide "I promise this value is in [min, max]".
// Halide trusts this promise and eliminates redundant bounds checks.
//
// Why it's safe here: We use repeat_edge boundary condition, which already
// clamps coordinates to [0, extent-1]. So the floor() result is always in
// [-1, extent-1], which we promise. This eliminates a clamp() in the inner
// loop, improving SIMD vectorization.
//
// WARNING: If the promise is wrong, undefined behavior results (buffer overrun).
//
// ## Bilinear vs Bicubic
//
// Bilinear: 2x2 pixel neighborhood, linear weights. Fast, slight blurring.
// Bicubic (Catmull-Rom): 4x4 pixel neighborhood, cubic polynomial weights.
//   Higher quality (sharper), but ~2x slower. Catmull-Rom (alpha=-0.5) passes
//   through data points and has continuous first derivatives.
//
// Both bicubic variants use SEPARABLE 2-pass implementation:
//   1. Horizontal pass: 4-tap cubic along x for each source row
//   2. Vertical pass: 4-tap cubic along y over horizontal results
// This reduces operations from 16 multiply-accumulates to 8 per output pixel.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Bilinear Resize (scale-factor variant)
// ---------------------------------------------------------------------------
class ResizeBilinear : public Generator<ResizeBilinear> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};   // src width x height x 3
    Input<float> scale_x{"scale_x"};            // output_width / input_width
    Input<float> scale_y{"scale_y"};            // output_height / input_height
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        // repeat_edge: clamp out-of-bounds coordinates to the nearest edge pixel.
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // Map output pixel center to source coordinate (pixel-center alignment).
        Expr src_x = (cast<float>(x) + 0.5f) / scale_x - 0.5f;
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;

        // Integer floor and fractional part for bilinear weights.
        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);  // horizontal weight [0, 1)
        Expr fy = src_y - cast<float>(iy);  // vertical weight [0, 1)

        // Promise coordinates are in valid range for repeat_edge.
        // ix ranges from -1 (for src_x just below 0) to extent-1.
        // This eliminates redundant clamp code in the vectorized inner loop.
        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());
        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        // Bilinear interpolation: weighted average of 2x2 pixel neighborhood.
        //
        //   (ix, iy)-------(ix+1, iy)
        //       |               |
        //       |   (fx, fy)    |    <- fractional position within the 2x2 cell
        //       |               |
        //   (ix, iy+1)-----(ix+1, iy+1)
        //
        // Weight for each corner:
        //   top-left:     (1-fx) * (1-fy)
        //   top-right:    fx * (1-fy)
        //   bottom-left:  (1-fx) * fy
        //   bottom-right: fx * fy
        Expr val = as_float(ix_s, iy_s, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix_s + 1, iy_s, c) * fx * (1.0f - fy) +
                   as_float(ix_s, iy_s + 1, c) * (1.0f - fx) * fy +
                   as_float(ix_s + 1, iy_s + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule
        // compute_at(output, yi): Compute as_float values within each y-tile.
        // This keeps the float conversion results in cache for the interpolation.
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Interleaved layout constraints
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBilinear, resize_bilinear)

// ---------------------------------------------------------------------------
// Bilinear Resize — Target-size variant
// ---------------------------------------------------------------------------
// Takes explicit target width/height instead of scale factors.
// Computes source coordinates directly: src = (out + 0.5) * src_dim / target_dim - 0.5
// This avoids float precision loss from computing an intermediate scale factor.
class ResizeBilinearTarget : public Generator<ResizeBilinearTarget> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<int32_t> target_w{"target_w"};
    Input<int32_t> target_h{"target_h"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        Expr src_w = cast<float>(input.dim(0).extent());
        Expr src_h = cast<float>(input.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // Direct coordinate mapping — no intermediate scale float.
        // Mathematically equivalent to: src = (out + 0.5) / (target/src) - 0.5
        // but with fewer floating-point operations and better precision.
        Expr src_x = (cast<float>(x) + 0.5f) * src_w / tw - 0.5f;
        Expr src_y = (cast<float>(y) + 0.5f) * src_h / th - 0.5f;

        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());
        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        Expr val = as_float(ix_s, iy_s, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix_s + 1, iy_s, c) * fx * (1.0f - fy) +
                   as_float(ix_s, iy_s + 1, c) * (1.0f - fx) * fy +
                   as_float(ix_s + 1, iy_s + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBilinearTarget, resize_bilinear_target)

// ---------------------------------------------------------------------------
// Bicubic Resize (Catmull-Rom, alpha=-0.5) — Separable 2-pass
// ---------------------------------------------------------------------------
// Horizontal 4-tap pass followed by vertical 4-tap pass.
// Reduces 16 multiply-accumulates per pixel (4x4 kernel) to 8 (4+4).
class ResizeBicubic : public Generator<ResizeBicubic> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<float> scale_x{"scale_x"};
    Input<float> scale_y{"scale_y"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    // Catmull-Rom cubic weight function (alpha = -0.5):
    //
    //   |t| <= 1:  W(t) = (3/2)|t|^3 - (5/2)|t|^2 + 1
    //   |t| <= 2:  W(t) = -(1/2)|t|^3 + (5/2)|t|^2 - 4|t| + 2
    //   |t| > 2:   W(t) = 0
    //
    // Properties of Catmull-Rom:
    //   - Interpolating: W(0) = 1, W(±1) = 0 (passes through data points)
    //   - C1 continuous: first derivative is continuous
    //   - Partition of unity: sum of weights = 1 for any fractional position
    //   - alpha = -0.5: optimal for natural images (minimizes blocking artifacts)
    static Expr cubic_weight(Expr t) {
        Expr at = abs(t);
        Expr at2 = at * at;
        Expr at3 = at2 * at;
        // select(condition, true_value, false_value) — Halide's ternary operator.
        // For |t| > 2, the second branch returns values near 0, which is
        // mathematically correct (the Catmull-Rom kernel has finite support).
        return select(
            at <= 1.0f,
            1.5f * at3 - 2.5f * at2 + 1.0f,
            -0.5f * at3 + 2.5f * at2 - 4.0f * at + 2.0f
        );
    }

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // --- Horizontal pass: 4-tap cubic along x for each source row ---
        // Source x-coordinate for this output column
        Expr src_x = (cast<float>(x) + 0.5f) / scale_x - 0.5f;
        Expr ix = cast<int>(floor(src_x));
        Expr fx = src_x - cast<float>(ix);  // fractional position within the 4-tap window

        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());

        // 4-tap cubic: sample at offsets dx = {-1, 0, 1, 2} relative to floor(src_x).
        // The weight for each tap is cubic_weight(fx - dx).
        // This C++ loop unrolls at generator time into 4 multiply-adds.
        Func h_interp("h_interp");
        Expr h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            h_val += as_float(ix_s + dx, y, c) * cubic_weight(fx - cast<float>(dx));
        }
        h_interp(x, y, c) = h_val;

        // --- Vertical pass: 4-tap cubic along y over horizontal results ---
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;
        Expr iy = cast<int>(floor(src_y));
        Expr fy = src_y - cast<float>(iy);

        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        // 4-tap cubic along y: reads from the horizontal intermediate h_interp.
        Expr v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            v_val += h_interp(x, iy_s + dy, c) * cubic_weight(fy - cast<float>(dy));
        }

        output(x, y, c) = cast<uint8_t>(clamp(v_val, 0.0f, 255.0f));

        // --- Schedule ---
        //
        // h_interp.compute_at(output, yi): Compute horizontal intermediate
        // within each output y-tile. This means h_interp rows are computed
        // just before the vertical pass reads them, staying in L1/L2 cache.
        //
        // as_float.compute_at(h_interp, y): Convert uint8->float once per
        // h_interp row (reused by the 4 horizontal taps).
        h_interp.compute_at(output, yi)
                .reorder(c, x, y)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        as_float.compute_at(h_interp, y);

        // Smaller vector width (8) than bilinear (16) because cubic requires
        // more arithmetic per pixel (4 taps x 2 passes vs 2 taps x 1 pass).
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 16)
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

HALIDE_REGISTER_GENERATOR(ResizeBicubic, resize_bicubic)

// ---------------------------------------------------------------------------
// Bicubic Resize (Catmull-Rom) — Target-size variant
// ---------------------------------------------------------------------------
// Same separable 2-pass algorithm, but takes explicit target dimensions
// instead of scale factors. Direct coordinate mapping for better precision.
class ResizeBicubicTarget : public Generator<ResizeBicubicTarget> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
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
            1.5f * at3 - 2.5f * at2 + 1.0f,
            -0.5f * at3 + 2.5f * at2 - 4.0f * at + 2.0f
        );
    }

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        Expr src_w = cast<float>(input.dim(0).extent());
        Expr src_h = cast<float>(input.dim(1).extent());
        Expr tw = cast<float>(target_w);
        Expr th = cast<float>(target_h);

        // Horizontal pass — direct coordinate mapping
        Expr src_x = (cast<float>(x) + 0.5f) * src_w / tw - 0.5f;
        Expr ix = cast<int>(floor(src_x));
        Expr fx = src_x - cast<float>(ix);

        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());

        Func h_interp("h_interp");
        Expr h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            h_val += as_float(ix_s + dx, y, c) * cubic_weight(fx - cast<float>(dx));
        }
        h_interp(x, y, c) = h_val;

        // Vertical pass — direct coordinate mapping
        Expr src_y = (cast<float>(y) + 0.5f) * src_h / th - 0.5f;
        Expr iy = cast<int>(floor(src_y));
        Expr fy = src_y - cast<float>(iy);

        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        Expr v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            v_val += h_interp(x, iy_s + dy, c) * cubic_weight(fy - cast<float>(dy));
        }

        output(x, y, c) = cast<uint8_t>(clamp(v_val, 0.0f, 255.0f));

        h_interp.compute_at(output, yi)
                .reorder(c, x, y)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        as_float.compute_at(h_interp, y);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 16)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBicubicTarget, resize_bicubic_target)
