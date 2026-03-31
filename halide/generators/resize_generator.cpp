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
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBilinear, resize_bilinear)

// ---------------------------------------------------------------------------
// Bicubic Resize (Catmull-Rom)
// 16-tap (4x4 neighborhood) interpolation with cubic weighting.
// ---------------------------------------------------------------------------
class ResizeBicubic : public Generator<ResizeBicubic> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<float> scale_x{"scale_x"};
    Input<float> scale_y{"scale_y"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        Expr src_x = (cast<float>(x) + 0.5f) / scale_x - 0.5f;
        Expr src_y = (cast<float>(y) + 0.5f) / scale_y - 0.5f;

        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // Catmull-Rom cubic weight function: W(t) for t >= 0
        //   W(t) = (3/2)|t|^3 - (5/2)|t|^2 + 1           for |t| <= 1
        //   W(t) = -(1/2)|t|^3 + (5/2)|t|^2 - 4|t| + 2   for 1 < |t| <= 2
        auto cubic_weight = [](Expr t) -> Expr {
            Expr at = abs(t);
            Expr at2 = at * at;
            Expr at3 = at2 * at;
            return select(
                at <= 1.0f,
                1.5f * at3 - 2.5f * at2 + 1.0f,
                -0.5f * at3 + 2.5f * at2 - 4.0f * at + 2.0f
            );
        };

        // 4x4 interpolation
        Expr val = cast<float>(0);
        for (int dy = -1; dy <= 2; dy++) {
            Expr wy = cubic_weight(fy - cast<float>(dy));
            for (int dx = -1; dx <= 2; dx++) {
                Expr wx = cubic_weight(fx - cast<float>(dx));
                val += as_float(ix + dx, iy + dy, c) * wx * wy;
            }
        }

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeBicubic, resize_bicubic)
