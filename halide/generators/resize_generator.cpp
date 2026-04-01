#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Bilinear Resize
// Uses repeat_edge boundary for safe edge access.
// scale_x and scale_y are output/input ratios (e.g., 2.0 = double size).
// ---------------------------------------------------------------------------
class ResizeBilinear : public Generator<ResizeBilinear> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};   // src width x height x 3
    Input<float> scale_x{"scale_x"};            // output_width / input_width
    Input<float> scale_y{"scale_y"};            // output_height / input_height
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // Map output pixel to source coordinate
        Expr src_x = (cast<float>(x) + 0.5f) / scale_x - 0.5f;
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;

        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // Promise coordinates are in valid range for repeat_edge.
        // This eliminates redundant boundary clamp code in the inner loop,
        // improving vectorization. Safe because repeat_edge already clamps.
        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());
        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        // Bilinear interpolation: 4-tap (2x2 neighborhood)
        Expr val = as_float(ix_s, iy_s, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix_s + 1, iy_s, c) * fx * (1.0f - fy) +
                   as_float(ix_s, iy_s + 1, c) * (1.0f - fx) * fy +
                   as_float(ix_s + 1, iy_s + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Interleaved layout: channel stride = 1, x stride = 3
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
// Takes explicit target width/height instead of scale factors.
// Computes source coordinates directly: src = (out + 0.5) * src_dim / target_dim - 0.5
// This avoids float precision loss from intermediate scale factor computation.
// ---------------------------------------------------------------------------
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

        // Direct coordinate mapping — no intermediate scale float
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
// Bicubic Resize (Catmull-Rom) — Separable 2-pass
// Horizontal 4-tap pass followed by vertical 4-tap pass.
// Reduces 16 multiply-accumulates to 8 per output pixel.
// ---------------------------------------------------------------------------
class ResizeBicubic : public Generator<ResizeBicubic> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<float> scale_x{"scale_x"};
    Input<float> scale_y{"scale_y"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    // Catmull-Rom cubic weight function (α = -0.5):
    //   W(t) = (3/2)|t|^3 - (5/2)|t|^2 + 1           for |t| <= 1
    //   W(t) = -(1/2)|t|^3 + (5/2)|t|^2 - 4|t| + 2   for 1 < |t| <= 2
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

        // Source coordinates for horizontal pass (fractional x per output column)
        Expr src_x = (cast<float>(x) + 0.5f) / scale_x - 0.5f;
        Expr ix = cast<int>(floor(src_x));
        Expr fx = src_x - cast<float>(ix);

        // Promise source x-coordinate in valid range for repeat_edge,
        // eliminating redundant boundary clamp code in the inner loop.
        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());

        // --- Horizontal pass: 4-tap cubic along x for each source row ---
        Func h_interp("h_interp");
        Expr h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            h_val += as_float(ix_s + dx, y, c) * cubic_weight(fx - cast<float>(dx));
        }
        h_interp(x, y, c) = h_val;

        // Source coordinates for vertical pass (fractional y per output row)
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;
        Expr iy = cast<int>(floor(src_y));
        Expr fy = src_y - cast<float>(iy);

        // Promise source y-coordinate in valid range for repeat_edge.
        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        // --- Vertical pass: 4-tap cubic along y over horizontal results ---
        Expr v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            v_val += h_interp(x, iy_s + dy, c) * cubic_weight(fy - cast<float>(dy));
        }

        output(x, y, c) = cast<uint8_t>(clamp(v_val, 0.0f, 255.0f));

        // --- Schedule ---
        // Horizontal intermediate computed per output tile strip
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

        // Interleaved layout: channel stride = 1, x stride = 3
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
// Takes explicit target width/height. Same separable 2-pass algorithm.
// ---------------------------------------------------------------------------
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
