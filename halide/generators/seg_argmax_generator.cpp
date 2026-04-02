#include "Halide.h"

using namespace Halide;

class SegArgmax : public Generator<SegArgmax> {
public:
    GeneratorParam<int> num_classes{"num_classes", 8};

    Input<Buffer<float, 3>> input{"input"};       // (x, y, c) planar float32
    Output<Buffer<uint8_t, 2>> output{"output"};  // (x, y) class mask

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int nc = num_classes;

        // Fully unrolled argmax: pure functional, no RDom needed.
        // Since softmax(exp) is monotonic, argmax on raw logits == argmax on softmax.
        Expr max_val = input(x, y, 0);
        Expr max_idx = cast<uint8_t>(0);
        for (int c = 1; c < nc; c++) {
            Expr val = input(x, y, c);
            Expr is_greater = val > max_val;
            max_idx = select(is_greater, cast<uint8_t>(c), max_idx);
            max_val = select(is_greater, val, max_val);
        }
        output(x, y) = max_idx;

        // --- Schedule for ARM NEON ---

        // Bound channel dimension for optimization
        input.dim(2).set_bounds(0, nc);

        // Tile by 32 rows, parallelize across tiles, vectorize 8 pixels wide.
        // Halide splits 8-wide float ops into 2x4-wide NEON instructions.
        output.split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Prefetch input planes ahead to hide latency from planar memory layout
        output.prefetch(input, y, y, 2);
    }
};

HALIDE_REGISTER_GENERATOR(SegArgmax, seg_argmax)
