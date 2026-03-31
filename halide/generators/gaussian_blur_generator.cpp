#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Gaussian Blur on single-channel (Y plane of NV21)
// Separable implementation: horizontal pass then vertical pass.
// Uses repeat_edge boundary condition for safe edge handling.
// ---------------------------------------------------------------------------
class GaussianBlurY : public Generator<GaussianBlurY> {
public:
    // Kernel radius: kernel size = 2*radius+1. Default radius=2 -> 5x5 kernel.
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 2>> input{"input"};   // width x height (single channel)
    Output<Buffer<uint8_t, 2>> output{"output"};

    Var x{"x"}, y{"y"}, yi{"yi"};

    void generate() {
        int r = radius;
        Func clamped = repeat_edge(input);

        // Pre-compute Gaussian kernel weights (compile-time constant)
        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        std::vector<float> kernel_weights(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_weights[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_weights[i + r];
        }
        for (auto& w : kernel_weights) w /= sum;

        // Horizontal pass
        Func blur_x("blur_x");
        Expr val_x = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_x += cast<float>(clamped(x + i, y)) * kernel_weights[i + r];
        }
        blur_x(x, y) = val_x;

        // Vertical pass
        Func blur_y("blur_y");
        Expr val_y = cast<float>(0);
        for (int i = -r; i <= r; i++) {
            val_y += blur_x(x, y + i) * kernel_weights[i + r];
        }
        blur_y(x, y) = val_y;

        output(x, y) = cast<uint8_t>(clamp(blur_y(x, y), 0.0f, 255.0f));

        // Schedule for ARM NEON
        blur_x.compute_at(output, y)
              .vectorize(x, 16, TailStrategy::RoundUp);

        output.split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurY, gaussian_blur_y)

// ---------------------------------------------------------------------------
// Gaussian Blur on 3-channel RGB
// Same separable approach, applied to each channel independently.
// ---------------------------------------------------------------------------
class GaussianBlurRgb : public Generator<GaussianBlurRgb> {
public:
    GeneratorParam<int> radius{"radius", 2};

    Input<Buffer<uint8_t, 3>> input{"input"};   // width x height x 3 (interleaved)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int r = radius;
        Func clamped = repeat_edge(input);

        float sigma = r / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        std::vector<float> kernel_weights(2 * r + 1);
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            kernel_weights[i + r] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += kernel_weights[i + r];
        }
        for (auto& w : kernel_weights) w /= sum;

        // Horizontal pass
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

        // Schedule for ARM NEON
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

        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(GaussianBlurRgb, gaussian_blur_rgb)
