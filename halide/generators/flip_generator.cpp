#include "Halide.h"

using namespace Halide;

// ---------------------------------------------------------------------------
// Flip (Horizontal / Vertical)
// Pure index remapping, no interpolation needed.
// flip_code: 0 = horizontal (mirror left-right), 1 = vertical (mirror top-bottom)
// ---------------------------------------------------------------------------
class FlipFixed : public Generator<FlipFixed> {
public:
    GeneratorParam<int> flip_code{"flip_code", 0};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr w = input.dim(0).extent();
        Expr h = input.dim(1).extent();

        int code = flip_code;
        if (code == 0) {
            // Horizontal flip: mirror left-right
            output(x, y, c) = input(w - 1 - x, y, c);
        } else {
            // Vertical flip: mirror top-bottom
            output(x, y, c) = input(x, h - 1 - y, c);
        }

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

HALIDE_REGISTER_GENERATOR(FlipFixed, flip_fixed)
