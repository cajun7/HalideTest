// =============================================================================
// NV21 to RGB Conversion Generator (BT.601 Limited-Range)
// =============================================================================
//
// Converts NV21 (YUV 4:2:0 semi-planar) to RGB using BT.601 limited-range
// coefficients with fixed-point integer arithmetic.
//
// ## NV21 Memory Layout
//
// NV21 is the standard Android camera YUV format (also called YCrCb).
// The image is stored in two separate planes:
//
//   Y plane (luma): Full resolution — one byte per pixel.
//     Row 0: Y(0,0) Y(1,0) Y(2,0) ... Y(W-1,0)
//     Row 1: Y(0,1) Y(1,1) Y(2,1) ... Y(W-1,1)
//     ...
//
//   UV plane (chroma): Half resolution in both dimensions.
//     Each row has width bytes, alternating V (Cr) and U (Cb):
//     Row 0: V(0,0) U(0,0) V(1,0) U(1,0) ... V(W/2-1,0) U(W/2-1,0)
//     Row 1: V(0,1) U(0,1) V(1,1) U(1,1) ...
//     (height/2 rows total)
//
//   The "NV21" name comes from the byte order: V first, then U (NV12 = U first).
//   Each UV pair corresponds to a 2x2 block of Y pixels (4:2:0 subsampling).
//
// ## BT.601 Limited-Range Conversion
//
// ITU-R BT.601 defines the standard for SDTV YUV<->RGB conversion.
// "Limited range" means:
//   - Y:  [16, 235]  (16 = black, 235 = white)
//   - UV: [16, 240]  (128 = neutral/gray)
//
// The conversion formulas (floating-point):
//   R = 1.164 * (Y - 16) + 1.596 * (V - 128)
//   G = 1.164 * (Y - 16) - 0.391 * (U - 128) - 0.813 * (V - 128)
//   B = 1.164 * (Y - 16) + 2.018 * (U - 128)
//
// Fixed-point (scaled by 256, shift by 8):
//   1.164 * 256 = 297.984 -> 298
//   1.596 * 256 = 408.576 -> 409
//   0.391 * 256 = 100.096 -> 100
//   0.813 * 256 = 208.128 -> 208
//   2.018 * 256 = 516.608 -> 516
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class Nv21ToRgb : public Generator<Nv21ToRgb> {
public:
    // We model the UV plane as a 2D buffer of raw bytes (width x height/2).
    // This is simpler than trying to use a 3D buffer with interleaved V,U.
    // Accessing V and U requires computing the correct byte offset manually.
    Input<Buffer<uint8_t, 2>> y_plane{"y_plane"};    // width x height
    Input<Buffer<uint8_t, 2>> uv_plane{"uv_plane"};  // width x (height/2) raw bytes

    Output<Buffer<uint8_t, 3>> output{"output"};      // width x height x 3 (RGB)

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // Sample Y at full resolution.
        // Cast to int32 to avoid overflow in the BT.601 arithmetic below
        // (e.g., 298 * 235 = 69,930 which fits int32 but overflows uint8/int16).
        Expr y_val = cast<int32_t>(y_plane(x, y));

        // Sample UV at half resolution from the raw byte buffer.
        //
        // For a full-resolution pixel at column x:
        //   - The UV pair index is x/2 (integer division)
        //   - The byte offset of the V value is (x/2)*2 (= x & ~1 for even alignment)
        //   - U is at (x/2)*2 + 1
        //
        // Example for x=0,1,2,3,4,5:
        //   x/2  = 0, 0, 1, 1, 2, 2
        //   uv_x = 0, 0, 2, 2, 4, 4   (byte offset of V)
        //   V from byte 0, 0, 2, 2, 4, 4
        //   U from byte 1, 1, 3, 3, 5, 5
        Expr uv_x = (x / 2) * 2;   // byte offset of the V,U pair
        Expr uv_y = y / 2;          // row in UV plane (half vertical resolution)
        Expr v_val = cast<int32_t>(uv_plane(uv_x, uv_y)) - 128;      // V (Cr), centered at 0
        Expr u_val = cast<int32_t>(uv_plane(uv_x + 1, uv_y)) - 128;  // U (Cb), centered at 0

        // BT.601 YUV to RGB conversion (fixed-point, shift by 8).
        //
        // Y' = (Y - 16) * 298 + 128
        //   - Subtract 16: limited-range offset (Y=16 is black)
        //   - Multiply by 298: ~1.164 * 256 scaling factor
        //   - Add 128: rounding bias for the right-shift by 8
        Expr y_scaled = (y_val - 16) * 298 + 128;

        // Apply chroma contribution and shift back to [0, 255] range.
        Expr r = (y_scaled + 409 * v_val) >> 8;           // Red depends on V (Cr)
        Expr g = (y_scaled - 100 * u_val - 208 * v_val) >> 8;  // Green depends on both U and V
        Expr b = (y_scaled + 516 * u_val) >> 8;           // Blue depends on U (Cb)

        // mux(c, {r, g, b}) selects r when c=0, g when c=1, b when c=2.
        // This is Halide's way of writing a channel-dependent expression
        // without explicit if/select chains. It compiles to efficient
        // conditional moves or lookup tables in SIMD code.
        //
        // clamp(..., 0, 255) ensures the result fits in uint8 range.
        // This is necessary because the BT.601 formula can produce values
        // slightly outside [0, 255] for extreme input combinations.
        output(x, y, c) = cast<uint8_t>(clamp(
            mux(c, {r, g, b}), 0, 255));
    }

    void schedule() {
        // Standard interleaved RGB output schedule.
        // See rgb_bgr_generator.cpp for detailed explanation of each directive.
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        // Y and UV planes use default planar layout (stride(0) = 1).
        // Each row is contiguous in memory.
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ToRgb, nv21_to_rgb)
