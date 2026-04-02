// =============================================================================
// Rotation Generators (Fixed 90/180/270 and Arbitrary Angle)
// =============================================================================
//
// ## Fixed Rotation (RotateFixed)
//
// Pure index remapping for 90, 180, and 270 degree rotations.
// No interpolation needed — each output pixel maps exactly to one input pixel.
//
// Rotation formulas (clockwise):
//   90 CW:  output(x, y) = input(y, H-1-x)    — output is H x W
//   180:    output(x, y) = input(W-1-x, H-1-y) — output is W x H
//   270 CW: output(x, y) = input(W-1-y, x)     — output is H x W
//
// rotation_code is a compile-time GeneratorParam:
//   1 = 90 CW, 2 = 180, 3 = 270 CW
// Three separate .a/.h files are generated (one per angle).
//
// ## Arbitrary Rotation (RotateArbitrary)
//
// Rotates by any angle using INVERSE MAPPING with bilinear interpolation.
//
// Why inverse mapping? Forward mapping (for each input pixel, compute where
// it goes in the output) creates holes — multiple output pixels may map to
// the same input pixel, while others get nothing. Inverse mapping (for each
// output pixel, compute where it came from in the input) guarantees every
// output pixel gets a value.
//
// Uses constant_exterior (black) for out-of-bounds source pixels, since
// rotated coordinates frequently fall far outside the original image bounds.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Fixed Rotation (90, 180, 270 degrees CW)
// ---------------------------------------------------------------------------
class RotateFixed : public Generator<RotateFixed> {
public:
    // 1 = 90 CW, 2 = 180, 3 = 270 CW (= 90 CCW)
    GeneratorParam<int> rotation_code{"rotation_code", 1};

    Input<Buffer<uint8_t, 3>> input{"input"};
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        Expr w = input.dim(0).extent();  // input width
        Expr h = input.dim(1).extent();  // input height

        // This if/else resolves at generator time (compile-time).
        // Only the selected rotation formula is compiled into the output.
        int code = rotation_code;
        if (code == 1) {
            // 90 CW: columns become rows (transposed + horizontal flip).
            //   Output size: H x W (width and height swap)
            //   output(x, y) reads from input(y, H-1-x)
            //
            // Visualization (4x3 input -> 3x4 output):
            //   Input:          Output (90 CW):
            //   A B C D         I E A
            //   E F G H   ->   J F B
            //   I J K L         K G C
            //                   L H D
            output(x, y, c) = input(y, h - 1 - x, c);
        } else if (code == 2) {
            // 180: flip both axes. Output size: W x H (same as input).
            //   output(x, y) reads from input(W-1-x, H-1-y)
            output(x, y, c) = input(w - 1 - x, h - 1 - y, c);
        } else {
            // 270 CW (= 90 CCW): columns become rows (transposed + vertical flip).
            //   Output size: H x W (width and height swap)
            //   output(x, y) reads from input(W-1-y, x)
            output(x, y, c) = input(w - 1 - y, x, c);
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

HALIDE_REGISTER_GENERATOR(RotateFixed, rotate_fixed)

// ---------------------------------------------------------------------------
// Arbitrary Angle Rotation
// ---------------------------------------------------------------------------
class RotateArbitrary : public Generator<RotateArbitrary> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<float> angle_rad{"angle_rad"};  // Rotation angle in radians (positive = CCW)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // constant_exterior: returns black (0) for out-of-bounds access.
        // Used instead of repeat_edge because rotated coordinates often go
        // far outside the image — clamping to edges would create visible
        // stretching artifacts.
        Func clamped = constant_exterior(input, cast<uint8_t>(0));

        // Convert to float for bilinear interpolation arithmetic.
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // Center of rotation: image center (W/2, H/2).
        Expr cx = cast<float>(input.dim(0).extent()) / 2.0f;
        Expr cy = cast<float>(input.dim(1).extent()) / 2.0f;

        // Precompute cos/sin of the rotation angle.
        Expr cos_a = cos(angle_rad);
        Expr sin_a = sin(angle_rad);

        // INVERSE MAPPING: for each output pixel (x, y), compute the
        // source coordinate in the input image.
        //
        // The inverse rotation matrix is the transpose of the forward rotation:
        //   [cos_a   sin_a]   [dx]     [src_x - cx]
        //   [-sin_a  cos_a] * [dy]  =  [src_y - cy]
        //
        // where (dx, dy) = (x - cx, y - cy) is the output pixel relative to center.
        Expr dx = cast<float>(x) - cx;
        Expr dy = cast<float>(y) - cy;
        Expr src_x = cos_a * dx + sin_a * dy + cx;
        Expr src_y = -sin_a * dx + cos_a * dy + cy;

        // Bilinear interpolation at the (non-integer) source coordinate.
        //
        // Split into integer part (floor) and fractional part:
        //   ix, iy = top-left integer pixel
        //   fx, fy = fractional offset [0, 1) used as interpolation weights
        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // NOTE: We CANNOT use unsafe_promise_clamped here.
        // Unlike resize (where repeat_edge guarantees coordinates stay in range),
        // rotated coordinates frequently go far outside the image bounds.
        // constant_exterior handles this correctly by returning black.
        Expr val = as_float(ix, iy, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix + 1, iy, c) * fx * (1.0f - fy) +
                   as_float(ix, iy + 1, c) * (1.0f - fx) * fy +
                   as_float(ix + 1, iy + 1, c) * fx * fy;

        output(x, y, c) = cast<uint8_t>(clamp(val, 0.0f, 255.0f));

        // Schedule: smaller vector width (8) because bilinear interpolation
        // with non-linear coordinate transforms limits SIMD efficiency.
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 8, TailStrategy::GuardWithIf)
              .parallel(y);

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(RotateArbitrary, rotate_arbitrary)
