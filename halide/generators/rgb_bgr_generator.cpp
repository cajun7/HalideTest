#include "Halide.h"

using namespace Halide;

// RGB <-> BGR channel swap. Since swapping channels 0 and 2 is symmetric,
// a single generator serves both directions.
class RgbBgrConvert : public Generator<RgbBgrConvert> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (interleaved)
    Output<Buffer<uint8_t, 3>> output{"output"}; // width x height x 3 (interleaved)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // Swap channel 0 and channel 2 (R<->B), keep channel 1 (G)
        output(x, y, c) = input(x, y, 2 - c);
    }

    void schedule() {
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        // Interleaved layout: channel stride = 1, x stride = 3
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(RgbBgrConvert, rgb_bgr_convert)
