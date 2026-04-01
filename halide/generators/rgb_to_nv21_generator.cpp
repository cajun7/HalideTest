#include "Halide.h"

using namespace Halide;

// RGB to NV21 conversion using BT.601 coefficients (fixed-point integer arithmetic).
//
// Inverse of nv21_to_rgb_generator.cpp. Produces:
//   Y plane:  width x height (full resolution), uint8
//   UV plane: width x (height/2) raw bytes, interleaved V,U pairs
//             V at even offsets, U at odd offsets within each row
//
// BT.601 forward transform:
//   Y  = (( 66*R + 129*G +  25*B + 128) >> 8) +  16
//   Cb = ((-38*R -  74*G + 112*B + 128) >> 8) + 128
//   Cr = ((112*R -  94*G -  18*B + 128) >> 8) + 128
//
// UV is subsampled by averaging the chroma of each 2x2 pixel block.
class RgbToNv21 : public Generator<RgbToNv21> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};          // width x height x 3 (RGB interleaved)
    Output<Buffer<uint8_t, 2>> y_output{"y_output"};   // width x height
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"}; // width x (height/2) raw bytes

    Var x{"x"}, y{"y"};

    void generate() {
        // Helper: extract R, G, B as int16 at any (px, py)
        // Use int32 to avoid overflow in BT.601 arithmetic
        Func r_val("r_val"), g_val("g_val"), b_val("b_val");
        r_val(x, y) = cast<int32_t>(input(x, y, 0));
        g_val(x, y) = cast<int32_t>(input(x, y, 1));
        b_val(x, y) = cast<int32_t>(input(x, y, 2));

        // Y at full resolution
        Expr r = r_val(x, y);
        Expr g = g_val(x, y);
        Expr b = b_val(x, y);
        Expr y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        y_output(x, y) = cast<uint8_t>(clamp(y_val, 0, 255));

        // UV at half resolution: average Cb/Cr over each 2x2 block
        // x here is the byte offset (0, 1, 2, ... w-1), y is the UV row
        // Even x = V (Cr), odd x = U (Cb)
        // Block pixel coords: bx = (x/2)*2, by = y*2
        Expr bx = (x / 2) * 2;
        Expr by = 2 * y;

        Func cb_full("cb_full"), cr_full("cr_full");
        cb_full(x, y) = ((-38 * r_val(x, y) - 74 * g_val(x, y) + 112 * b_val(x, y) + 128) >> 8) + 128;
        cr_full(x, y) = ((112 * r_val(x, y) - 94 * g_val(x, y) - 18 * b_val(x, y) + 128) >> 8) + 128;

        Expr cr_avg = (cr_full(bx, by) + cr_full(bx + 1, by) +
                       cr_full(bx, by + 1) + cr_full(bx + 1, by + 1) + 2) / 4;
        Expr cb_avg = (cb_full(bx, by) + cb_full(bx + 1, by) +
                       cb_full(bx, by + 1) + cb_full(bx + 1, by + 1) + 2) / 4;

        // NV21: V at even byte, U at odd byte
        Expr is_v = (x % 2) == 0;
        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, cr_avg, cb_avg), 0, 255));
    }

    void schedule() {
        // Y output schedule
        y_output.vectorize(x, 16, TailStrategy::GuardWithIf)
                .parallel(y);

        // UV output schedule
        uv_output.vectorize(x, 8, TailStrategy::GuardWithIf)
                 .parallel(y);

        input.dim(2).set_bounds(0, 3);

        // UV output: contiguous bytes
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(RgbToNv21, rgb_to_nv21)
