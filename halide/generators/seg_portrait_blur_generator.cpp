// =============================================================================
// Segmentation-Guided Portrait Blur (Bokeh) Generator
// =============================================================================
//
// Purpose:
//   Real-time portrait mode effect — keep foreground (person) sharp, blur
//   background with a disc-shaped bokeh kernel. This is the #1 most popular
//   segmentation-driven camera effect.
//
// Fused pipeline (single-pass):
//   1. Bilinear upsample seg_mask to full resolution → soft alpha [0, 1]
//   2. Apply sigmoid-like feathering for controllable edge softness
//   3. Compute disc-kernel blur (same as lens_blur) for background
//   4. Alpha-blend: output = alpha * sharp + (1 - alpha) * blurred
//
// Why fuse?
//   Separate approach: argmax → resize mask → blur image → alpha blend
//   = 4 steps, 3 intermediate full-res buffers.
//   Fused: reads input + mask, writes output. Zero intermediates.
//   ~2x faster from eliminated memory traffic.
//
// Boundary handling:
//   - Input image: repeat_edge (natural for portrait, no dark borders)
//   - Seg mask: repeat_edge (smooth mask extension at edges)
//
// Resolution independence:
//   - seg_mask can be any resolution (typically 256×256 or 512×512)
//   - Bilinear upsampling maps mask coords to output coords automatically
//   - Handles odd resolutions via GuardWithIf tail strategy
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class SegPortraitBlur : public Generator<SegPortraitBlur> {
public:
    // Compile-time upper bound for disc blur radius.
    // Runtime radius must be <= max_radius.
    // 12 is suitable for portrait mode (larger than lens_blur default of 8).
    GeneratorParam<int> max_radius{"max_radius", 12};

    // Original sharp RGB image (interleaved)
    Input<Buffer<uint8_t, 3>> input{"input"};
    // Segmentation mask from model (argmax class indices, any resolution)
    Input<Buffer<uint8_t, 2>> seg_mask{"seg_mask"};
    // Which class index is foreground (e.g., 15 = "person" in DeepLab)
    Input<int32_t> fg_class{"fg_class"};
    // Runtime blur radius (0 = no blur, must be <= max_radius)
    Input<int32_t> blur_radius{"blur_radius"};
    // Edge feathering sharpness (1.0 = linear blend, 3-5 = natural portrait, >10 = hard edge)
    Input<float> edge_softness{"edge_softness"};
    // Output RGB image (interleaved, same size as input)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        int mr = max_radius;

        // =================================================================
        // Step 1: Soft alpha mask via bilinear upsampling
        // =================================================================
        // Convert seg_mask to a binary float mask (1.0 = foreground, 0.0 = background)
        // at the mask's native resolution.
        Func mask_float("mask_float");
        Func mask_clamped = repeat_edge(seg_mask);
        mask_float(x, y) = select(mask_clamped(x, y) == cast<uint8_t>(fg_class),
                                   1.0f, 0.0f);

        // Bilinear upsample mask to output resolution.
        // This produces smooth [0, 1] values at class boundaries — key for
        // natural-looking edges without jagged mask artifacts.
        Func alpha_mask("alpha_mask");

        Expr seg_w = cast<float>(seg_mask.dim(0).extent());
        Expr seg_h = cast<float>(seg_mask.dim(1).extent());
        Expr out_w = cast<float>(input.dim(0).extent());
        Expr out_h = cast<float>(input.dim(1).extent());

        // Map output pixel center to mask pixel center (pixel-center alignment)
        Expr mask_x = (cast<float>(x) + 0.5f) * seg_w / out_w - 0.5f;
        Expr mask_y = (cast<float>(y) + 0.5f) * seg_h / out_h - 0.5f;

        Expr mx = cast<int>(floor(mask_x));
        Expr my = cast<int>(floor(mask_y));
        Expr mfx = mask_x - cast<float>(mx);
        Expr mfy = mask_y - cast<float>(my);

        // Bilinear interpolation of binary mask → smooth alpha
        Expr raw_alpha = mask_float(mx, my) * (1.0f - mfx) * (1.0f - mfy) +
                         mask_float(mx + 1, my) * mfx * (1.0f - mfy) +
                         mask_float(mx, my + 1) * (1.0f - mfx) * mfy +
                         mask_float(mx + 1, my + 1) * mfx * mfy;

        // Sigmoid-like feathering: sharpens the alpha transition at boundaries.
        // clamp((raw_alpha - 0.5) * edge_softness + 0.5, 0, 1)
        //   edge_softness = 1.0 → linear blend (same as raw bilinear)
        //   edge_softness = 3-5 → natural portrait feathering
        //   edge_softness > 10  → near-hard binary mask
        alpha_mask(x, y) = clamp((raw_alpha - 0.5f) * edge_softness + 0.5f,
                                  0.0f, 1.0f);

        // =================================================================
        // Step 2: Disc-kernel blur for background
        // =================================================================
        Func clamped = repeat_edge(input);

        // 2D reduction domain for disc kernel (same pattern as lens_blur)
        RDom r(-mr, 2 * mr + 1, -mr, 2 * mr + 1);

        Expr dist_sq = r.x * r.x + r.y * r.y;
        Expr in_disc = dist_sq <= blur_radius * blur_radius;

        Func blur_sum("blur_sum"), blur_count("blur_count");
        blur_sum(x, y, c) = cast<float>(0);
        blur_count(x, y) = 0;

        blur_sum(x, y, c) += select(in_disc,
            cast<float>(clamped(x + r.x, y + r.y, c)), 0.0f);
        blur_count(x, y) += select(in_disc, 1, 0);

        // Normalized blur result
        Func blurred("blurred");
        blurred(x, y, c) = blur_sum(x, y, c) / cast<float>(max(blur_count(x, y), 1));

        // =================================================================
        // Step 3: Alpha blend — sharp foreground, blurred background
        // =================================================================
        Expr alpha = alpha_mask(x, y);
        Expr sharp_val = cast<float>(clamped(x, y, c));
        Expr blur_val = blurred(x, y, c);

        // output = alpha * sharp + (1 - alpha) * blurred
        // +0.5f for proper rounding before uint8 truncation
        output(x, y, c) = cast<uint8_t>(clamp(
            alpha * sharp_val + (1.0f - alpha) * blur_val + 0.5f,
            0.0f, 255.0f));

        // =================================================================
        // Schedule
        // =================================================================

        // blur_count: channel-independent, compute once globally
        blur_count.compute_root()
                  .vectorize(x, 8, TailStrategy::GuardWithIf)
                  .parallel(y);
        blur_count.update(0)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);

        // alpha_mask: compute per output tile (reused across 3 channels)
        alpha_mask.compute_at(output, yi)
                  .vectorize(x, 8, TailStrategy::GuardWithIf);

        // mask_float: compute lazily within alpha_mask
        mask_float.compute_at(alpha_mask, x);

        // blur_sum: compute per output tile
        blur_sum.compute_at(output, yi)
                .reorder(c, x, y)
                .bound(c, 0, 3)
                .unroll(c)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        blur_sum.update()
                .reorder(c, x, r.x, r.y, y)
                .unroll(c)
                .vectorize(x, 8, TailStrategy::GuardWithIf);

        // blurred: compute inline (folded into output expression)

        // Output: 16-row tiles (smaller due to O(r²) blur cost)
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 16)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Prefetch input data for next tile to hide memory latency
        output.prefetch(input, yi, yi, 2);

        // Interleaved layout constraints (RGB input and output)
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(SegPortraitBlur, seg_portrait_blur)
