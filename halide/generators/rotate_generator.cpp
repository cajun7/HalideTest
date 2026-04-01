#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Fixed Rotation (90, 180, 270 degrees)
// Pure index remapping, no interpolation needed.
// The rotation_degrees GeneratorParam selects which transform to apply.
// ---------------------------------------------------------------------------
class RotateFixed : public Generator<RotateFixed> {
public:
    // 1 = 90 CW, 2 = 180, 3 = 270 CW (i.e., 90 CCW)
    GeneratorParam<int> rotation_code{"rotation_code", 1};

    Input<Buffer<uint8_t, 3>> input{"input"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr w = input.dim(0).extent();
        Expr h = input.dim(1).extent();

        int code = rotation_code;
        if (code == 1) {
            // 90 CW: output HxW, output(x,y) -> input(y, h-1-x)
            output(x, y, c) = input(y, h - 1 - x, c);
        } else if (code == 2) {
            // 180: output WxH, output(x,y) -> input(w-1-x, h-1-y)
            output(x, y, c) = input(w - 1 - x, h - 1 - y, c);
        } else {
            // 270 CW (90 CCW): output HxW, output(x,y) -> input(w-1-y, x)
            output(x, y, c) = input(w - 1 - y, x, c);
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

HALIDE_REGISTER_GENERATOR(RotateFixed, rotate_fixed)

// ---------------------------------------------------------------------------
// Arbitrary Angle Rotation
// Rotates around the image center with bilinear interpolation.
// Uses constant_exterior (black) for out-of-bounds pixels.
// ---------------------------------------------------------------------------
class RotateArbitrary : public Generator<RotateArbitrary> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<float> angle_rad{"angle_rad"};  // Rotation angle in radians (positive = CCW)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Func clamped = constant_exterior(input, cast<uint8_t>(0));
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // Center of rotation
        Expr cx = cast<float>(input.dim(0).extent()) / 2.0f;
        Expr cy = cast<float>(input.dim(1).extent()) / 2.0f;

        Expr cos_a = cos(angle_rad);
        Expr sin_a = sin(angle_rad);

        // Inverse rotation: map output pixel to source coordinate
        Expr dx = cast<float>(x) - cx;
        Expr dy = cast<float>(y) - cy;
        Expr src_x = cos_a * dx + sin_a * dy + cx;
        Expr src_y = -sin_a * dx + cos_a * dy + cy;

        // Bilinear interpolation at source coordinates
        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // Note: cannot use unsafe_promise_clamped here -- rotated coordinates
        // frequently go far outside the image bounds. constant_exterior handles this.
        Expr val = as_float(ix, iy, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix + 1, iy, c) * fx * (1.0f - fy) +
                   as_float(ix, iy + 1, c) * (1.0f - fx) * fy +
                   as_float(ix + 1, iy + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 8, TailStrategy::GuardWithIf)
              .parallel(y);

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(RotateArbitrary, rotate_arbitrary)
