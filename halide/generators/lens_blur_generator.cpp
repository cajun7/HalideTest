#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// Lens Blur (Bokeh) using a disc-shaped kernel.
// Averages all pixels within a circular region of given radius.
// Uses constant_exterior boundary (black outside image bounds).
class LensBlur : public Generator<LensBlur> {
public:
    // Compile-time max radius bounds the RDom size. Runtime radius must be <= max_radius.
    GeneratorParam<int> max_radius{"max_radius", 8};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (RGB interleaved)
    Input<int32_t> radius{"radius"};             // Runtime blur radius
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mr = max_radius;
        Func clamped = constant_exterior(input, cast<uint8_t>(0));

        // Disc kernel: gather all pixels within circular radius
        RDom r(-mr, 2 * mr + 1, -mr, 2 * mr + 1);
        Expr dist_sq = r.x * r.x + r.y * r.y;
        Expr in_disc = dist_sq <= radius * radius;

        // Accumulate sum and count per pixel
        Func sum_f("sum_f"), count_f("count_f");
        sum_f(x, y, c) = cast<float>(0);
        count_f(x, y) = 0;

        sum_f(x, y, c) += select(in_disc,
            cast<float>(clamped(x + r.x, y + r.y, c)), 0.0f);
        count_f(x, y) += select(in_disc, 1, 0);

        output(x, y, c) = cast<uint8_t>(clamp(
            sum_f(x, y, c) / cast<float>(max(count_f(x, y), 1)),
            0.0f, 255.0f));

        // Schedule
        count_f.compute_root()
               .vectorize(x, 8, TailStrategy::GuardWithIf)
               .parallel(y);
        count_f.update(0)
               .vectorize(x, 8, TailStrategy::GuardWithIf);

        sum_f.compute_at(output, yi)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c)
             .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Update step schedule
        sum_f.update()
             .reorder(c, x, r.x, r.y, y)
             .unroll(c)
             .vectorize(x, 8, TailStrategy::GuardWithIf);

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

HALIDE_REGISTER_GENERATOR(LensBlur, lens_blur)
