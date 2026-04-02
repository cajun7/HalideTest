// =============================================================================
// Gaussian Blur Generator (Single-Channel and RGB)
// =============================================================================
//
// Implements Gaussian blur using a SEPARABLE 2-pass approach:
//   1. Horizontal pass: blur each row independently
//   2. Vertical pass: blur each column of the horizontal result
//
// ## Why Separable?
//
// A direct 2D Gaussian convolution with kernel size K requires K^2 multiply-
// accumulate operations per pixel (e.g., 25 for a 5x5 kernel). Because the
// 2D Gaussian kernel is the outer product of two 1D Gaussians:
//   G(x,y) = G(x) * G(y)
// we can decompose it into two 1D passes requiring only 2*K operations per
// pixel (e.g., 10 for 5x5). This is a classic image processing optimization.
//
// ## Boundary Handling: repeat_edge
//
// repeat_edge clamps out-of-bounds coordinates to the nearest edge pixel:
//   access(-1, y) -> access(0, y)    (left edge)
//   access(W, y)  -> access(W-1, y)  (right edge)
//
// This avoids dark borders that would occur with zero-padding, and is more
// natural than mirroring for typical photographic images.
//
// ## Kernel Weights
//
// Computed at generator time (compile-time) using the standard Gaussian formula:
//   w(i) = exp(-i^2 / (2 * sigma^2))
// Then normalized so all weights sum to 1.0.
//
// sigma is derived from radius as sigma = radius/2.0, with a minimum of 0.5.
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Gaussian Blur on single-channel (grayscale / Y plane)
// ---------------------------------------------------------------------------
class GaussianBlurY : public Generator<GaussianBlurY> {
public:
    // GeneratorParam: compile-time parameter baked into the AOT code.
    // Changing radius requires re-running the generator, but the resulting
    // code has zero runtime overhead from the parameter.
    //
    // Kernel size = 2*radius+1. Default radius=2 -> 5x5 kernel.
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 2>> input{"input"};   // width x height (single channel)
    Output<Buffer<uint8_t, 2>> output{"output"};

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int r = radius;

        // repeat_edge: boundary condition that clamps coordinates to [0, extent-1].
        // Returns a Func that can be accessed at any coordinate safely.
        Func clamped = repeat_edge(input);

        // Pre-compute Gaussian kernel weights.
        // This C++ loop runs at GENERATOR TIME (on the host machine), not at
        // runtime on the target device. The resulting weights become compile-time
        // constants embedded in the generated code.
        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;  // Minimum sigma to ensure meaningful blur
        std::vector<float> kernel_weights(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_weights[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_weights[i + r];
        }
        for (auto& w : kernel_weights) w /= sum;  // Normalize so weights sum to 1.0

        // --- Horizontal pass ---
        // For each pixel (x,y), sum weighted neighbors along x-axis.
        // This C++ for-loop unrolls at generator time, producing a chain of
        // multiply-add operations (no loop at runtime).
        Func blur_x("blur_x");
        Expr val_x = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_x += cast<float>(clamped(x + i, y)) * kernel_weights[i + r];
        }
        blur_x(x, y) = val_x;

        // --- Vertical pass ---
        // For each pixel (x,y), sum weighted neighbors along y-axis
        // using the horizontal-blurred intermediate result.
        Func blur_y("blur_y");
        Expr val_y = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_y += blur_x(x, y + i) * kernel_weights[i + r];
        }
        blur_y(x, y) = val_y;

        // Clamp result to [0, 255] and convert back to uint8.
        output(x, y) = cast<uint8_t>(clamp(blur_y(x, y), 0.0f, 255.0f));

        // --- Schedule for ARM NEON ---
        //
        // compute_at(output, y): Compute blur_x values just before they're
        // needed for each output row. This means blur_x results are computed
        // per row strip and stay in L1 cache for the vertical pass.
        //
        // TailStrategy::RoundUp on blur_x: Compute extra elements past the
        // row end (they'll be discarded). This avoids the scalar tail loop
        // overhead on intermediates where the extra reads are safe (repeat_edge).
        blur_x.compute_at(output, y)
              .vectorize(x, 16, TailStrategy::RoundUp);

        // split(y, y, yi, 32): Tile the y dimension into 32-row strips.
        // GuardWithIf on the final output ensures correct handling when
        // height isn't divisible by 32.
        output.split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // prefetch(input, y, y, 2): Insert hardware prefetch instructions
        // to load input rows 2 iterations ahead of the current y position.
        // On ARM, this compiles to PLD/PRFM instructions that fill cache lines
        // before data is needed, hiding memory latency.
        output.prefetch(input, y, y, 2);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurY, gaussian_blur_y)

// ---------------------------------------------------------------------------
// Gaussian Blur on 3-channel RGB (interleaved)
// ---------------------------------------------------------------------------
// Same separable approach, applied to each channel independently via
// the c dimension. Interleaved layout: R0 G0 B0 R1 G1 B1 ...
class GaussianBlurRgb : public Generator<GaussianBlurRgb> {
public:
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (interleaved)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int r = radius;
        Func clamped = repeat_edge(input);

        // Same kernel computation as GaussianBlurY
        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        std::vector<float> kernel_weights(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_weights[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_weights[i + r];
        }
        for (auto& w : kernel_weights) w /= sum;

        // Horizontal pass (operates on each channel via the c variable)
        Func blur_x("blur_x");
        Expr val_x = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_x += cast<float>(clamped(x + i, y, c)) * kernel_weights[i + r];
        }
        blur_x(x, y, c) = val_x;

        // Vertical pass
        Func blur_y("blur_y");
        Expr val_y = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_y += blur_x(x, y + i, c) * kernel_weights[i + r];
        }
        blur_y(x, y, c) = val_y;

        output(x, y, c) = cast<uint8_t>(clamp(blur_y(x, y, c), 0.0f, 255.0f));

        // --- Schedule for ARM NEON ---
        //
        // For interleaved RGB, we process all 3 channels per pixel in the
        // innermost loop (.reorder(c, x, y), .unroll(c)). This matches
        // the memory layout and allows the compiler to load one RGB triplet
        // and process all channels together.
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c);

        blur_x.compute_at(output, y)
              .reorder(c, x, y)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::RoundUp);

        output.split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        output.prefetch(input, y, y, 2);

        // Interleaved layout constraints (see rgb_bgr_generator.cpp for details)
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurRgb, gaussian_blur_rgb)
