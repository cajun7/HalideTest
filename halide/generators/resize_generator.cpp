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

        // Bilinear interpolation: 4-tap (2x2 neighborhood)
        Expr val = as_float(ix, iy, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix + 1, iy, c) * fx * (1.0f - fy) +
                   as_float(ix, iy + 1, c) * (1.0f - fx) * fy +
                   as_float(ix + 1, iy + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBilinear, resize_bilinear)

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

        // --- Horizontal pass: 4-tap cubic along x for each source row ---
        Func h_interp("h_interp");
        Expr h_val = cast<float>(0);
        for (int dx = -1; dx <= 2; dx++) {
            h_val += as_float(ix + dx, y, c) * cubic_weight(fx - cast<float>(dx));
        }
        h_interp(x, y, c) = h_val;

        // Source coordinates for vertical pass (fractional y per output row)
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;
        Expr iy = cast<int>(floor(src_y));
        Expr fy = src_y - cast<float>(iy);

        // --- Vertical pass: 4-tap cubic along y over horizontal results ---
        Expr v_val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            v_val += h_interp(x, iy + dy, c) * cubic_weight(fy - cast<float>(dy));
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

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBicubic, resize_bicubic)
