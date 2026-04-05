// =============================================================================
// RGB <-> BGR Optimized Channel Swap Generator
// =============================================================================
//
// Same algorithm as rgb_bgr_generator.cpp: output(x, y, c) = input(x, y, 2-c)
//
// Optimizations:
//   - Wider vectorization: 32 pixels per SIMD iteration (2 NEON Q registers)
//   - Multi-row tiling: 8-row inner tiles reduce loop overhead
//   - Prefetching: loads next tile's data ahead of use
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class RgbBgrOptimized : public Generator<RgbBgrOptimized> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"}, yo{"yo"};

    void generate() {
        output(x, y, c) = input(x, y, 2 - c);
    }

    void schedule() {
        // Wider vectorization (32 pixels = 96 bytes per iteration, 2 Q registers)
        // Multi-row tiling (8 rows per inner tile)
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, yo, yi, 8, TailStrategy::GuardWithIf)
              .vectorize(x, 32, TailStrategy::GuardWithIf)
              .parallel(yo);

        // Prefetch next tile's input data
        output.prefetch(input, yo, yo, 1);

        // Interleaved layout
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(RgbBgrOptimized, rgb_bgr_optimized)
