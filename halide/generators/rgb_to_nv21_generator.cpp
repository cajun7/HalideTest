// =============================================================================
// RGB to NV21 Conversion Generator (BT.601 Limited-Range)
// =============================================================================
//
// Inverse of nv21_to_rgb_generator.cpp. Converts a 3-channel RGB interleaved
// image to NV21 (YUV 4:2:0 semi-planar) format.
//
// ## Output Format
//
// Produces two separate output buffers:
//   Y plane:  width x height (full resolution), one luma byte per pixel
//   UV plane: width x (height/2) raw bytes, interleaved V,U pairs
//             V at even byte offsets, U at odd byte offsets
//
// ## BT.601 Forward Transform (RGB -> YUV)
//
// The forward coefficients are derived by inverting the BT.601 matrix:
//   Y  = (( 66*R + 129*G +  25*B + 128) >> 8) +  16
//   Cb = ((-38*R -  74*G + 112*B + 128) >> 8) + 128    (U component)
//   Cr = ((112*R -  94*G -  18*B + 128) >> 8) + 128    (V component)
//
// Note: Cb = U (blue-difference), Cr = V (red-difference).
//
// ## Chroma Subsampling (4:2:0)
//
// UV is subsampled by averaging the Cb/Cr values of each 2x2 pixel block:
//   UV_avg = (UV(0,0) + UV(1,0) + UV(0,1) + UV(1,1) + 2) / 4
// The "+2" provides rounding (bias toward nearest integer instead of truncation).
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class RgbToNv21 : public Generator<RgbToNv21> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};          // width x height x 3 (RGB interleaved)
    Output<Buffer<uint8_t, 2>> y_output{"y_output"};   // width x height
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"}; // width x (height/2) raw bytes

    Var x{"x"}, y{"y"};

    void generate() {
        // Extract R, G, B channels as int32 at any pixel coordinate.
        // Using int32 avoids overflow in the BT.601 arithmetic
        // (e.g., 129*255 = 32,895 which overflows int16 range).
        Func r_val("r_val"), g_val("g_val"), b_val("b_val");
        r_val(x, y) = cast<int32_t>(input(x, y, 0));
        g_val(x, y) = cast<int32_t>(input(x, y, 1));
        b_val(x, y) = cast<int32_t>(input(x, y, 2));

        // --- Y output: full resolution ---
        // BT.601 luma: Y = ((66*R + 129*G + 25*B + 128) >> 8) + 16
        //   - Coefficients sum to 66+129+25 = 220 (not 256) because limited-range
        //     Y only spans [16, 235], not [0, 255]
        //   - +128: rounding bias for right-shift by 8
        //   - +16: limited-range offset (Y=16 is black)
        Expr r = r_val(x, y);
        Expr g = g_val(x, y);
        Expr b = b_val(x, y);
        Expr y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        y_output(x, y) = cast<uint8_t>(clamp(y_val, 0, 255));

        // --- UV output: half resolution with 2x2 block averaging ---
        //
        // The UV plane uses a different coordinate system than the Y plane:
        //   x = byte offset within the UV row (0 to width-1)
        //   y = UV row index (0 to height/2 - 1)
        //
        // For a given UV byte at (x, y):
        //   - Pixel pair index: x/2 (each pair is 2 bytes: V, U)
        //   - Block top-left in RGB coordinates: bx = (x/2)*2, by = y*2
        //
        // Even x = V (Cr) byte, odd x = U (Cb) byte
        Expr bx = (x / 2) * 2;  // top-left column of the 2x2 block in RGB space
        Expr by = 2 * y;         // top-left row of the 2x2 block in RGB space

        // Compute full-resolution Cb (U) and Cr (V) at every pixel.
        // These will be averaged over the 2x2 block below.
        Func cb_full("cb_full"), cr_full("cr_full");
        cb_full(x, y) = ((-38 * r_val(x, y) - 74 * g_val(x, y) + 112 * b_val(x, y) + 128) >> 8) + 128;
        cr_full(x, y) = ((112 * r_val(x, y) - 94 * g_val(x, y) - 18 * b_val(x, y) + 128) >> 8) + 128;

        // Average Cr (V) over the 2x2 block: 4 pixels at (bx, by), (bx+1, by),
        // (bx, by+1), (bx+1, by+1). The "+2" provides rounding.
        Expr cr_avg = (cr_full(bx, by) + cr_full(bx + 1, by) +
                       cr_full(bx, by + 1) + cr_full(bx + 1, by + 1) + 2) / 4;
        Expr cb_avg = (cb_full(bx, by) + cb_full(bx + 1, by) +
                       cb_full(bx, by + 1) + cb_full(bx + 1, by + 1) + 2) / 4;

        // NV21 byte order: V at even byte offsets, U at odd byte offsets.
        Expr is_v = (x % 2) == 0;
        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, cr_avg, cb_avg), 0, 255));
    }

    void schedule() {
        // Y output: vectorize 16 pixels at a time (16 uint8 = 128-bit NEON register)
        y_output.vectorize(x, 16, TailStrategy::GuardWithIf)
                .parallel(y);

        // UV output: vectorize 8 bytes at a time.
        // Smaller vector width than Y because each UV byte requires computing
        // a 2x2 average (more arithmetic per output element).
        uv_output.vectorize(x, 8, TailStrategy::GuardWithIf)
                 .parallel(y);

        // Inform Halide about input layout (3-channel interleaved: stride x=3, c=1)
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);

        // UV output: contiguous bytes (stride 1)
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(RgbToNv21, rgb_to_nv21)
