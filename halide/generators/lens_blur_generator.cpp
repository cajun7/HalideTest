// =============================================================================
// Lens Blur (Bokeh) Generator
// =============================================================================
//
// Simulates the out-of-focus blur of a camera lens using a disc-shaped kernel.
// For each output pixel, averages all input pixels within a circular region
// of the given radius. This produces a characteristic "bokeh" effect where
// point light sources become soft circles.
//
// ## Algorithm
//
// Unlike Gaussian blur (which uses a weighted kernel with exponential falloff),
// lens blur uses a UNIFORM disc kernel — all pixels inside the circle get
// equal weight. The kernel is defined as:
//   w(dx, dy) = 1 if (dx^2 + dy^2 <= radius^2), else 0
//
// This is NOT separable (a disc is not the outer product of two 1D functions),
// so we use a 2D reduction domain (RDom) instead of separable passes.
// Complexity is O(radius^2) per pixel.
//
// ## Boundary Handling: constant_exterior
//
// constant_exterior pads the image with a constant value (0 = black) outside
// its bounds. This means edge pixels may appear slightly darker because the
// average includes black padding pixels. For bokeh effects this is usually
// acceptable and avoids edge artifacts from repeat_edge.
//
// ## RDom (Reduction Domain)
//
// RDom defines a multi-dimensional iteration space for reductions (sums,
// products, min/max). Here we use a 2D RDom of size (2*max_radius+1)^2
// centered at the origin. The disc condition (dist_sq <= radius^2) masks
// out pixels outside the circle at runtime.
//
// max_radius (GeneratorParam): compile-time upper bound for the RDom size.
// The runtime radius can be anything from 0 to max_radius (default 8).
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class LensBlur : public Generator<LensBlur> {
public:
    GeneratorParam<int> max_radius{"max_radius", 8};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (RGB interleaved)
    Input<int32_t> radius{"radius"};             // Runtime blur radius (<= max_radius)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mr = max_radius;

        // constant_exterior: returns 0 (black) for any access outside the
        // image bounds. This is different from repeat_edge (which clamps to
        // the nearest edge pixel).
        Func clamped = constant_exterior(input, cast<uint8_t>(0));

        // 2D reduction domain centered at (0, 0), spanning [-mr, mr] in both axes.
        // RDom(start_x, extent_x, start_y, extent_y):
        //   r.x ranges over [-mr, mr] (inclusive)
        //   r.y ranges over [-mr, mr] (inclusive)
        RDom r(-mr, 2 * mr + 1, -mr, 2 * mr + 1);

        // Disc mask: only include pixels whose squared distance from center
        // is within the squared radius. This turns the square RDom into a
        // circular kernel at runtime.
        Expr dist_sq = r.x * r.x + r.y * r.y;
        Expr in_disc = dist_sq <= radius * radius;

        // Accumulate weighted sum and count for each output pixel.
        //
        // sum_f: accumulates pixel values inside the disc
        // count_f: counts how many pixels are inside the disc (for normalization)
        //
        // These Funcs have an INIT step (= 0) and an UPDATE step (+= ...).
        // The update step iterates over the RDom, making this a reduction.
        Func sum_f("sum_f"), count_f("count_f");
        sum_f(x, y, c) = cast<float>(0);
        count_f(x, y) = 0;

        // select(condition, true_value, false_value):
        //   If in_disc: accumulate the pixel value / count
        //   If outside disc: add 0 (no contribution)
        sum_f(x, y, c) += select(in_disc,
            cast<float>(clamped(x + r.x, y + r.y, c)), 0.0f);
        count_f(x, y) += select(in_disc, 1, 0);

        // Normalize: divide sum by count to get the average.
        // max(count, 1) prevents division by zero (shouldn't happen with radius >= 1,
        // but is defensive for radius = 0 edge case).
        output(x, y, c) = cast<uint8_t>(clamp(
            sum_f(x, y, c) / cast<float>(max(count_f(x, y), 1)),
            0.0f, 255.0f));

        // --- Schedule ---
        //
        // count_f is pixel-count only (no channel dimension), so compute it
        // once globally (compute_root) and reuse for all 3 channels.
        count_f.compute_root()
               .vectorize(x, 8, TailStrategy::GuardWithIf)
               .parallel(y);
        count_f.update(0)
               .vectorize(x, 8, TailStrategy::GuardWithIf);

        // sum_f: computed per output tile (yi).
        // The reduction iterates over the RDom for each pixel in the tile.
        sum_f.compute_at(output, yi)
             .reorder(c, x, y)
             .bound(c, 0, 3)
             .unroll(c)
             .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Schedule for the UPDATE step of sum_f:
        // Reorder so the RDom variables (r.x, r.y) are in the middle of the loop
        // nest, with channel unrolled and x vectorized in the inner loops.
        sum_f.update()
             .reorder(c, x, r.x, r.y, y)
             .unroll(c)
             .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Output: 16-row tiles, parallelized and vectorized.
        // Smaller tile (16) than Gaussian blur (32) because the O(r^2) kernel
        // means more computation per pixel.
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

HALIDE_REGISTER_GENERATOR(LensBlur, lens_blur)
