// =============================================================================
// Flip Generator (Horizontal / Vertical)
// =============================================================================
//
// Pure index remapping — no interpolation or computation needed.
// Simply reads from a mirrored source coordinate.
//
// flip_code (GeneratorParam, compile-time):
//   0 = horizontal flip (mirror left-right)
//   1 = vertical flip (mirror top-bottom)
//
// Because flip_code is a compile-time GeneratorParam, the if/else in generate()
// is resolved at generator time. The AOT-compiled function contains only the
// selected flip direction — no runtime branching. Two separate .a/.h files
// are generated: flip_horizontal and flip_vertical.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class FlipFixed : public Generator<FlipFixed> {
public:
    // Compile-time parameter: 0 = horizontal, 1 = vertical.
    // Set during AOT compilation via: -p flip_code=0 or -p flip_code=1
    GeneratorParam<int> flip_code{"flip_code", 0};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // input.dim(N).extent() returns the size of dimension N at runtime.
        // dim(0) = width (x), dim(1) = height (y), dim(2) = channels (c).
        Expr w = input.dim(0).extent();
        Expr h = input.dim(1).extent();

        // This if/else runs at GENERATOR TIME (C++ compile-time), not at
        // runtime on the device. Only one branch is compiled into the output.
        int code = flip_code;
        if (code == 0) {
            // Horizontal flip: mirror around the vertical center line.
            // x=0 reads from x=W-1, x=1 reads from x=W-2, etc.
            output(x, y, c) = input(w - 1 - x, y, c);
        } else {
            // Vertical flip: mirror around the horizontal center line.
            // y=0 reads from y=H-1, y=1 reads from y=H-2, etc.
            output(x, y, c) = input(x, h - 1 - y, c);
        }

        // Standard interleaved RGB schedule
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        // Interleaved layout constraints
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(FlipFixed, flip_fixed)
