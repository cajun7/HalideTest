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
// ## Optimizations (over baseline float approach)
//
// 1. Q10 fixed-point arithmetic: kernel weights scaled to integers summing
//    to 1024. Intermediate stored as int16, enabling 8-wide NEON vectorization
//    (vs 4-wide for float32). Accumulation in int32, normalized via >>10.
//
// 2. Sliding window: store_at(yi)/compute_at(y) reuses horizontal-pass rows
//    across adjacent output rows. For radius=2, each row is used by 5 output
//    rows — eliminates ~80% redundant H-pass computation.
//
// 3. Larger tiles (64 rows vs 32) for better parallel efficiency.
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
// Then converted to Q10 integers (sum = 1024) for fixed-point computation.
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
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 2>> input{"input"};   // width x height (single channel)
    Output<Buffer<uint8_t, 2>> output{"output"};

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int r = radius;

        Func clamped = repeat_edge(input);

        // Compute Q10 kernel weights at generator time.
        // This C++ code runs on the host machine, not on the target device.
        // The resulting integer weights become compile-time constants.
        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        std::vector<float> kernel_f(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_f[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_f[i + r];
        }
        // Convert to Q10 integer weights (sum = 1024)
        std::vector<int> kernel_q10(2 * r + 1);
        int q10_sum = 0;
        for (int i = 0; i < 2 * r + 1; i++) {
            kernel_q10[i] = (int)(kernel_f[i] / sum * 1024.0f + 0.5f);
            q10_sum += kernel_q10[i];
        }
        // Adjust center weight to ensure exact sum of 1024
        kernel_q10[r] += (1024 - q10_sum);

        // --- Horizontal pass ---
        // Accumulate in int32, normalize to int16 via >>10.
        // Max accumulation: (2r+1) × 1024 × 255 = 1,305,600 → fits int32.
        // After >>10: max ~255, stored as int16 (max 32767) ✓
        Func blur_x("blur_x");
        Expr h_acc = cast<int32_t>(0);
        for (int i = -r; i <= r; i++) {
            h_acc += cast<int32_t>(kernel_q10[i + r]) *
                     cast<int32_t>(clamped(x + i, y));
        }
        blur_x(x, y) = cast<int16_t>((h_acc + 512) >> 10);

        // --- Vertical pass ---
        // Accumulate in int32, normalize to uint8 via >>10 + clamp.
        Expr v_acc = cast<int32_t>(0);
        for (int i = -r; i <= r; i++) {
            v_acc += cast<int32_t>(kernel_q10[i + r]) *
                     cast<int32_t>(blur_x(x, y + i));
        }
        output(x, y) = cast<uint8_t>(clamp((v_acc + 512) >> 10, 0, 255));

        // --- Schedule for ARM NEON ---
        //
        // Sliding window: store_at(yi) allocates a ring buffer for the tile,
        // compute_at(y) computes rows lazily. Halide reuses previously computed
        // rows — for radius=2, each blur_x row is computed once and used 5 times.
        // Sliding window: store buffer at outer tile level (y), compute rows
        // lazily per inner row (yi). Halide reuses previously computed rows —
        // for radius=2, each blur_x row is computed once and used 5 times.
        output.split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        blur_x.store_at(output, y)
              .compute_at(output, yi)
              .vectorize(x, 16, TailStrategy::RoundUp);

        output.prefetch(input, y, y, 2);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurY, gaussian_blur_y)

// ---------------------------------------------------------------------------
// Gaussian Blur on 3-channel RGB (interleaved)
// ---------------------------------------------------------------------------
// Same Q10 fixed-point separable approach, applied to each channel
// independently via the c dimension. Interleaved layout: R0 G0 B0 R1 G1 B1 ...
class GaussianBlurRgb : public Generator<GaussianBlurRgb> {
public:
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (interleaved)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int r = radius;
        Func clamped = repeat_edge(input);

        // Same Q10 kernel computation as GaussianBlurY
        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        std::vector<float> kernel_f(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_f[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_f[i + r];
        }
        std::vector<int> kernel_q10(2 * r + 1);
        int q10_sum = 0;
        for (int i = 0; i < 2 * r + 1; i++) {
            kernel_q10[i] = (int)(kernel_f[i] / sum * 1024.0f + 0.5f);
            q10_sum += kernel_q10[i];
        }
        kernel_q10[r] += (1024 - q10_sum);

        // Horizontal pass with channel dimension
        Func blur_x("blur_x");
        Expr h_acc = cast<int32_t>(0);
        for (int i = -r; i <= r; i++) {
            h_acc += cast<int32_t>(kernel_q10[i + r]) *
                     cast<int32_t>(clamped(x + i, y, c));
        }
        blur_x(x, y, c) = cast<int16_t>((h_acc + 512) >> 10);

        // Vertical pass
        Expr v_acc = cast<int32_t>(0);
        for (int i = -r; i <= r; i++) {
            v_acc += cast<int32_t>(kernel_q10[i + r]) *
                     cast<int32_t>(blur_x(x, y + i, c));
        }
        output(x, y, c) = cast<uint8_t>(clamp((v_acc + 512) >> 10, 0, 255));

        // --- Schedule for ARM NEON ---
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c);

        output.split(y, y, yi, 64)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        blur_x.store_at(output, y)
              .compute_at(output, yi)
              .reorder(c, x, y)
              .unroll(c)
              .vectorize(x, 16, TailStrategy::RoundUp);

        output.prefetch(input, y, y, 2);

        // Interleaved layout constraints
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurRgb, gaussian_blur_rgb)
