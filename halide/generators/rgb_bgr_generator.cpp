// =============================================================================
// RGB <-> BGR Channel Swap Generator
// =============================================================================
//
// Swaps channel 0 (R) and channel 2 (B), keeping channel 1 (G) unchanged.
// Since `2 - c` maps {0,1,2} -> {2,1,0}, the operation is its own inverse:
// applying it twice returns the original image. A single generator serves
// both RGB->BGR and BGR->RGB directions.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

// Every Halide generator inherits from Generator<CRTP_Self>.
// The CRTP (Curiously Recurring Template Pattern) gives the base class
// access to the derived class's Input/Output/GeneratorParam declarations
// via static introspection at compile time.
class RgbBgrConvert : public Generator<RgbBgrConvert> {
public:
    // Input/Output declarations define the function signature of the
    // generated AOT code. Buffer<T, N> specifies element type and
    // number of dimensions.
    //
    // The string names ("input", "output") become parameter names in the
    // generated C function signature.
    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (interleaved)
    Output<Buffer<uint8_t, 3>> output{"output"}; // width x height x 3 (interleaved)

    // Var represents a loop variable in Halide's abstract loop nest.
    // x = column index, y = row index, c = channel index (0=R, 1=G, 2=B).
    // These are symbolic — they don't hold values themselves. They define
    // the iteration space when used in Func definitions.
    Var x{"x"}, y{"y"}, c{"c"};

    // generate() defines the ALGORITHM (what to compute).
    // Halide separates algorithm from schedule — the algorithm is independent
    // of how it's parallelized, vectorized, or tiled.
    void generate() {
        // The core operation: reverse channel order via index arithmetic.
        // `2 - c` maps: c=0(R) -> 2(B), c=1(G) -> 1(G), c=2(B) -> 0(R)
        //
        // This creates a pure Func (no update steps / reductions), which
        // means Halide can evaluate any pixel independently — enabling
        // trivial parallelization and vectorization.
        output(x, y, c) = input(x, y, 2 - c);
    }

    // schedule() defines HOW to compute the algorithm efficiently.
    // This controls loop ordering, parallelism, vectorization, and memory layout.
    // Changing the schedule never changes the computed result.
    void schedule() {
        // --- Loop ordering and optimization ---
        //
        // .reorder(c, x, y): Make the innermost loop iterate over channels (c),
        //   then columns (x), then rows (y). This matches the interleaved memory
        //   layout where RGB values for one pixel are adjacent in memory.
        //
        // .bound(c, 0, 3): Promise the compiler that c is always in [0, 3).
        //   This enables the compiler to eliminate bounds checks and generate
        //   tighter code. Without this, Halide must handle arbitrary channel counts.
        //
        // .unroll(c): Fully unroll the channel loop (only 3 iterations).
        //   This eliminates the loop overhead and allows the compiler to generate
        //   separate load/store instructions for R, G, B — often resulting in
        //   a single NEON instruction that shuffles all 3 channels at once.
        //
        // .vectorize(x, 16, TailStrategy::GuardWithIf): Process 16 pixels
        //   simultaneously using SIMD (NEON on ARM). 16 uint8 values = 128 bits
        //   = one NEON register. GuardWithIf means the last (potentially partial)
        //   vector uses scalar code protected by an if-check, safely handling
        //   image widths not divisible by 16.
        //
        // .parallel(y): Each row runs on a separate CPU core. On a 4-8 core
        //   mobile SoC, a 1080-row image creates 1080 parallel tasks.
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::GuardWithIf)
              .parallel(y);

        // --- Memory layout constraints ---
        //
        // Tell Halide that the input/output use INTERLEAVED layout (RGBRGBRGB...),
        // not planar layout (RRR...GGG...BBB...).
        //
        // In interleaved layout:
        //   - dim(0) is the x (column) dimension, with stride = 3
        //     (skip 3 bytes to reach the next pixel's R value)
        //   - dim(2) is the channel dimension, with stride = 1
        //     (R, G, B are adjacent bytes)
        //
        // This matches how Android Bitmap and OpenCV cv::Mat store RGB images.
        // Without these constraints, Halide assumes the default planar layout
        // (stride(0)=1), which would produce incorrect results on interleaved data.
        input.dim(0).set_stride(3);   // x-stride = 3 (skip RGB triplet)
        input.dim(2).set_stride(1);   // channel stride = 1 (R,G,B adjacent)
        input.dim(2).set_bounds(0, 3); // exactly 3 channels
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

// Register this generator class with the name "rgb_bgr_convert".
// The build script invokes:
//   ./bin/rgb_bgr_generator -g rgb_bgr_convert -f rgb_bgr_convert target=arm-64-android ...
// -g selects which registered generator to use
// -f sets the C function name in the generated .h/.a files
HALIDE_REGISTER_GENERATOR(RgbBgrConvert, rgb_bgr_convert)
