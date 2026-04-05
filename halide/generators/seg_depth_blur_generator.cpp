// =============================================================================
// Depth-Map Guided Multi-Kernel Blur Generator
// =============================================================================
//
// Purpose:
//   Simulate realistic camera depth-of-field by applying different disc blur
//   radii to different depth zones. Unlike seg_portrait_blur (binary sharp vs
//   blurred), this generator supports multiple blur levels driven by a
//   continuous depth map.
//
// Algorithm (layered pre-blur with depth-based selection):
//   1. Bilinear-upsample depth_map to output resolution → normalized [0, 1]
//   2. Pre-blur the image at each discrete radius from kernel_config
//   3. For each output pixel, look up depth → find matching kernel → select
//      from pre-blurred layer
//
// Why layered pre-blur?
//   Per-pixel variable-radius blur is O(max_r² × W × H) with no vectorization
//   opportunity. By pre-blurring at discrete levels and selecting per pixel,
//   each blur layer is a standard disc convolution that Halide can schedule
//   efficiently (tile, parallelize, vectorize).
//
// Inputs:
//   input         — RGB interleaved (W × H × 3)
//   depth_map     — Continuous depth (uint8, 0-255), any resolution
//   kernel_config — float buffer (max_layers × 3):
//                   [min_depth_norm, max_depth_norm, blur_radius]
//                   Depth values normalized to [0, 1]. Kernels must be sorted
//                   by min_depth ascending and non-overlapping.
//   num_kernels   — Number of active entries in kernel_config (runtime, <= max_layers)
//
// Output:
//   output        — RGB interleaved (same size as input)
//
// Boundary handling:
//   - Input image: repeat_edge (natural appearance, no dark borders)
//   - Depth map: repeat_edge (smooth depth extension at edges)
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class SegDepthBlur : public Generator<SegDepthBlur> {
public:
    // Compile-time upper bound for disc blur radius.
    GeneratorParam<int> max_radius{"max_radius", 12};
    // Compile-time upper bound for number of blur layers.
    // Each layer pre-blurs the entire image at its configured radius.
    GeneratorParam<int> max_layers{"max_layers", 5};

    Input<Buffer<uint8_t, 3>> input{"input"};           // RGB interleaved
    Input<Buffer<uint8_t, 2>> depth_map{"depth_map"};    // Continuous depth, any resolution
    Input<Buffer<float, 2>>   kernel_config{"kernel_config"}; // (max_layers, 3)
    Input<int32_t>            num_kernels{"num_kernels"};     // Active kernel count
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mr = max_radius;
        int ml = max_layers;

        // =================================================================
        // Step 1: Bilinear upsample depth map to output resolution
        // =================================================================
        Func depth_float("depth_float");
        Func depth_clamped = repeat_edge(depth_map);
        depth_float(x, y) = cast<float>(depth_clamped(x, y)) / 255.0f;

        Func depth_up("depth_up");
        Expr dm_w = cast<float>(depth_map.dim(0).extent());
        Expr dm_h = cast<float>(depth_map.dim(1).extent());
        Expr out_w = cast<float>(input.dim(0).extent());
        Expr out_h = cast<float>(input.dim(1).extent());

        // Pixel-center alignment (same as seg_portrait_blur)
        Expr map_x = (cast<float>(x) + 0.5f) * dm_w / out_w - 0.5f;
        Expr map_y = (cast<float>(y) + 0.5f) * dm_h / out_h - 0.5f;
        Expr dxi = cast<int>(floor(map_x));
        Expr dyi = cast<int>(floor(map_y));
        Expr dfx = map_x - cast<float>(dxi);
        Expr dfy = map_y - cast<float>(dyi);

        depth_up(x, y) = depth_float(dxi, dyi) * (1.0f - dfx) * (1.0f - dfy)
                        + depth_float(dxi + 1, dyi) * dfx * (1.0f - dfy)
                        + depth_float(dxi, dyi + 1) * (1.0f - dfx) * dfy
                        + depth_float(dxi + 1, dyi + 1) * dfx * dfy;

        // =================================================================
        // Step 2: Pre-blur image at each discrete radius level
        // =================================================================
        Func clamped = repeat_edge(input);

        // 2D reduction domain for disc kernel (shared by all layers)
        RDom r(-mr, 2 * mr + 1, -mr, 2 * mr + 1);

        // Create per-layer blur Funcs.
        // Each layer blurs the image at kernel_config(k, 2) radius.
        // Inactive layers (k >= num_kernels) produce the original image.
        std::vector<Func> layer_sum(ml);
        std::vector<Func> layer_count(ml);
        std::vector<Func> layer_blurred(ml);

        for (int k = 0; k < ml; k++) {
            layer_sum[k] = Func("layer_sum_" + std::to_string(k));
            layer_count[k] = Func("layer_count_" + std::to_string(k));
            layer_blurred[k] = Func("layer_blurred_" + std::to_string(k));

            // Init
            layer_sum[k](x, y, c) = cast<float>(0);
            layer_count[k](x, y) = 0;

            // Disc blur at this layer's radius
            // kernel_config(k, 2) is the blur_radius for layer k
            Expr layer_radius = kernel_config(k, 2);
            Expr layer_r_sq = cast<int>(layer_radius * layer_radius);
            Expr dist_sq = r.x * r.x + r.y * r.y;
            Expr k_active = k < num_kernels;
            Expr in_disc = k_active && (dist_sq <= layer_r_sq);

            layer_sum[k](x, y, c) += select(in_disc,
                cast<float>(clamped(x + r.x, y + r.y, c)), 0.0f);
            layer_count[k](x, y) += select(in_disc, 1, 0);

            // Normalized blur result; fallback to original for inactive layers
            layer_blurred[k](x, y, c) = select(k_active,
                layer_sum[k](x, y, c) / cast<float>(max(layer_count[k](x, y), 1)),
                cast<float>(clamped(x, y, c)));
        }

        // =================================================================
        // Step 3: Per-pixel depth-based layer selection
        // =================================================================
        // For each pixel, find which kernel's depth range contains its depth
        // and select the corresponding blurred layer.
        // Kernels are sorted by min_depth; last match wins (iterate forward).
        Expr d = depth_up(x, y);

        // Start with the original sharp image as default
        Expr result = cast<float>(clamped(x, y, c));

        // Iterate through layers: if depth falls in [min_depth, max_depth],
        // select that layer's blur result
        for (int k = 0; k < ml; k++) {
            Expr k_active = k < num_kernels;
            Expr min_d = kernel_config(k, 0);
            Expr max_d = kernel_config(k, 1);
            Expr in_range = k_active && (d >= min_d) && (d <= max_d);
            result = select(in_range, layer_blurred[k](x, y, c), result);
        }

        output(x, y, c) = cast<uint8_t>(clamp(result + 0.5f, 0.0f, 255.0f));

        // =================================================================
        // Schedule
        // =================================================================

        // Per-layer blur scheduling
        for (int k = 0; k < ml; k++) {
            // Count: channel-independent, compute once globally
            layer_count[k].compute_root()
                          .vectorize(x, 8, TailStrategy::GuardWithIf)
                          .parallel(y);
            layer_count[k].update(0)
                          .vectorize(x, 8, TailStrategy::GuardWithIf);

            // Sum: compute per output tile
            layer_sum[k].compute_at(output, yi)
                        .reorder(c, x, y)
                        .bound(c, 0, 3)
                        .unroll(c)
                        .vectorize(x, 8, TailStrategy::GuardWithIf);
            layer_sum[k].update()
                        .reorder(c, x, r.x, r.y, y)
                        .unroll(c)
                        .vectorize(x, 8, TailStrategy::GuardWithIf);

            // Layer result: compute per tile (reused for each channel)
            layer_blurred[k].compute_at(output, yi)
                            .reorder(c, x, y)
                            .bound(c, 0, 3)
                            .unroll(c)
                            .vectorize(x, 8, TailStrategy::GuardWithIf);
        }

        // Depth upsampling: compute per tile
        depth_up.compute_at(output, yi)
                .vectorize(x, 8, TailStrategy::GuardWithIf);
        depth_float.compute_at(depth_up, x);

        // Output: 16-row tiles (smaller due to O(r²) blur per layer)
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 16)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Prefetch input data for next tile
        output.prefetch(input, yi, yi, 2);

        // Interleaved layout constraints (RGB input and output)
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(SegDepthBlur, seg_depth_blur)
