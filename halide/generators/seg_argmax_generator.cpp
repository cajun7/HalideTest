// =============================================================================
// Segmentation Argmax Generator
// =============================================================================
//
// Purpose:
//   Post-process semantic segmentation model output. A segmentation model
//   (e.g., DeepLab, U-Net) outputs a tensor of shape (H x W x C) where C is
//   the number of classes. Each value is a "logit" (raw score) representing
//   how likely that pixel belongs to each class. This generator finds the
//   class with the highest logit for each pixel (argmax operation).
//
// Input:  float32 planar buffer, shape (width, height, num_classes)
//         - "Planar" means each class channel is stored as a separate
//           contiguous 2D plane in memory:
//           [plane_0: H*W floats][plane_1: H*W floats]...[plane_7: H*W floats]
//         - This is the default Halide memory layout for 3D buffers.
//
// Output: uint8 buffer, shape (width, height)
//         - Each pixel contains the class index (0 to num_classes-1) with
//           the highest logit value.
//
// Why no softmax?
//   Softmax = exp(x_i) / sum(exp(x_j)). Since exp() is a monotonically
//   increasing function, the relative ordering of values is preserved:
//     argmax(softmax(x)) == argmax(x)
//   Skipping softmax avoids expensive exp() and division per pixel, making
//   this generator significantly faster when only the class label is needed.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

// Every Halide generator inherits from Generator<CRTP_Self>.
// The CRTP (Curiously Recurring Template Pattern) gives the base class
// access to the derived class's Input/Output/GeneratorParam declarations.
class SegArgmax : public Generator<SegArgmax> {
public:
    // -------------------------------------------------------------------------
    // GeneratorParam: compile-time parameter
    // -------------------------------------------------------------------------
    // Unlike runtime parameters, GeneratorParams are baked into the generated
    // code at AOT (Ahead-Of-Time) compilation. The C++ for-loop below unrolls
    // at generator execution time (on the host), NOT at runtime on the device.
    // This means changing num_classes requires re-running the generator, but
    // the resulting code has zero overhead from the parameter.
    //
    // Usage in build script:
    //   ./bin/seg_argmax_generator -g seg_argmax -f seg_argmax num_classes=8
    GeneratorParam<int> num_classes{"num_classes", 8};

    // -------------------------------------------------------------------------
    // Input / Output declarations
    // -------------------------------------------------------------------------
    // Buffer<float, 3>: a 3-dimensional buffer of float32 values.
    //   - dim 0 = x (width),  dim 1 = y (height),  dim 2 = c (class channel)
    //   - Planar layout: stride(0)=1, stride(1)=width, stride(2)=width*height
    //     (Halide default -- no manual stride configuration needed)
    //
    // Buffer<uint8_t, 2>: a 2-dimensional buffer of uint8 values.
    //   - dim 0 = x (width),  dim 1 = y (height)
    //   - Each pixel stores the winning class index (0-7 for 8 classes)
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<uint8_t, 2>> output{"output"};

    // -------------------------------------------------------------------------
    // Halide Variables (loop indices)
    // -------------------------------------------------------------------------
    // Var represents a loop variable in Halide's abstract loop nest.
    // x iterates over columns, y over rows, yi is a "split" variable
    // created when we tile the y dimension for cache efficiency.
    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int nc = num_classes;

        // =====================================================================
        // Algorithm definition (the "what" -- independent of schedule)
        // =====================================================================
        //
        // Halide separates algorithm from schedule. Here we define WHAT to
        // compute. The schedule (below) defines HOW to compute it efficiently.
        //
        // Strategy: fully unrolled argmax using Halide Expr trees.
        //
        // This C++ for-loop runs at GENERATOR TIME (on host), not at runtime.
        // It builds a Halide expression tree that looks like a chain of
        // select() (ternary) operations:
        //
        //   max_idx = select(ch7 > max6,  7,
        //             select(ch6 > max5,  6,
        //             select(ch5 > max4,  5,
        //             ...
        //             select(ch1 > ch0,   1, 0)...)))
        //
        // This is equivalent to a nested ternary in C:
        //   result = (ch7>max6) ? 7 : (ch6>max5) ? 6 : ... : 0
        //
        // Why not use Halide RDom (reduction domain)?
        //   RDom creates an update definition (sequential reduction), which
        //   requires careful scheduling of the reduction dimension. For small
        //   fixed counts (8 classes), unrolling produces cleaner code:
        //   - No reduction scheduling complexity
        //   - Pure Func (no update steps) -> vectorizes trivially
        //   - Compiler can optimize the entire select chain as one expression
        //
        // Tie-breaking behavior:
        //   We use strict greater-than (>). If multiple channels have the same
        //   maximum value, the LOWEST index wins (class 0 beats class 1, etc.)
        //   because later classes must be strictly greater to replace the current
        //   winner. This deterministic behavior is important for test verification.

        Expr max_val = input(x, y, 0);       // Start with class 0's logit
        Expr max_idx = cast<uint8_t>(0);      // Current best class index

        for (int c = 1; c < nc; c++) {
            Expr val = input(x, y, c);        // Load logit for class c
            Expr is_greater = val > max_val;   // Compare against current max

            // select(condition, true_value, false_value) -- Halide's ternary op.
            // At runtime, this compiles to conditional moves (no branches),
            // which are SIMD-friendly and avoid branch misprediction penalties.
            max_idx = select(is_greater, cast<uint8_t>(c), max_idx);
            max_val = select(is_greater, val, max_val);
        }

        // Define the output: each pixel (x,y) gets the winning class index.
        output(x, y) = max_idx;

        // =====================================================================
        // Schedule definition (the "how" -- performance optimization)
        // =====================================================================
        //
        // The schedule controls loop ordering, parallelism, vectorization,
        // and memory access patterns. It does NOT change the computed result.

        // Tell Halide the exact channel count so it can:
        // 1. Eliminate bounds checks on the channel dimension
        // 2. Perform tighter bounds inference for intermediate computations
        // Without this, Halide must assume arbitrary channel counts.
        input.dim(2).set_bounds(0, nc);

        // --- Tiling: split y into outer (y) and inner (yi) loops ---
        // split(y, y_outer, y_inner, tile_size):
        //   Original: for y in [0, height)
        //   After:    for y in [0, height/64)       <- parallel across tiles
        //               for yi in [0, 64)           <- sequential within tile
        //
        // Why 64 rows per tile?
        //   For a 1920-wide image with 8 float32 planes:
        //   64 rows * 1920 pixels * 4 bytes * 8 planes = 3.75 MB
        //   This fits in L2 cache on most ARM64 SoCs (4-8 MB),
        //   and reduces parallel scheduling overhead vs smaller tiles.
        //
        // --- Parallelism: parallel(y) ---
        // Each tile of 64 rows runs on a separate CPU core.
        // On a typical mobile SoC with 4-8 cores, a 1080-row image creates
        // 17 tiles, providing good load balancing.
        //
        // --- Vectorization: vectorize(x, 16) ---
        // Process 16 pixels simultaneously using SIMD (NEON on ARM).
        // For float32 comparisons: ARM NEON has 128-bit registers (4 floats),
        // so Halide automatically splits the 16-wide vector into 4 NEON ops.
        // Wider vectorization amortizes loop overhead and enables better
        // instruction-level parallelism.
        //
        // TailStrategy::GuardWithIf:
        //   When image width isn't divisible by 16 (e.g., 641 pixels), the
        //   last iteration uses scalar code protected by an if-check.
        //   This safely handles odd resolutions without buffer overruns.
        //   Alternative TailStrategy::RoundUp would read past the buffer edge.
        output.split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // --- Prefetching ---
        // prefetch(buffer, at_var, prefetch_var, offset):
        //   Insert CPU prefetch instructions to load data into cache BEFORE
        //   it's needed. With planar layout, each of the 8 class planes is
        //   stored at a different memory address (separated by width*height*4 bytes).
        //   Without prefetching, accessing 8 planes per pixel row would cause
        //   cache misses. Prefetching 2 rows ahead hides the memory latency.
        //
        //   On ARM, this compiles to PLD/PRFM instructions.
        output.prefetch(input, y, y, 2);
    }
};

// Register this generator class with the name "seg_argmax".
// This name is used in the build script:
//   ./bin/seg_argmax_generator -g seg_argmax -f seg_argmax num_classes=8
//
// -g: generator name (must match HALIDE_REGISTER_GENERATOR second arg)
// -f: function name in the generated .h/.a (the C symbol you call at runtime)
HALIDE_REGISTER_GENERATOR(SegArgmax, seg_argmax)
