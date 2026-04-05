// =============================================================================
// Segmentation-Guided Background Replacement Generator
// =============================================================================
//
// Purpose:
//   Virtual background replacement — composites foreground (person) from camera
//   frame onto an arbitrary background image. Zoom/Meet-style effect without
//   a green screen.
//
// Fused pipeline (single-pass):
//   1. Bilinear upsample seg_mask to full resolution → soft alpha [0, 1]
//   2. Sigmoid feathering for controllable edge softness
//   3. Bilinear resize background image to match output resolution
//   4. Alpha composite: output = alpha * foreground + (1 - alpha) * background
//
// Why fuse?
//   Separate approach: argmax → resize mask → resize bg → alpha blend
//   = 4 steps, 3 intermediate full-res buffers (~18 bytes/pixel).
//   Fused: reads 3 inputs, writes 1 output. ~9 bytes/pixel total.
//   ~3-4x faster from eliminated memory traffic.
//
// Performance:
//   No disc blur → much cheaper than portrait blur. Dominated by memory
//   bandwidth (read fg + bg + mask, write output). Very fast (~10-15 ops/pixel).
//
// Resolution independence:
//   - bg_image can be ANY resolution (bilinear-resized to match fg)
//   - seg_mask can be any resolution (bilinear-upsampled to match fg)
//   - Output size matches fg_image size
//   - Handles odd resolutions via GuardWithIf tail strategy
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class SegBgReplace : public Generator<SegBgReplace> {
public:
    // Camera frame (foreground) — interleaved RGB
    Input<Buffer<uint8_t, 3>> fg_image{"fg_image"};
    // Replacement background — interleaved RGB, any resolution
    Input<Buffer<uint8_t, 3>> bg_image{"bg_image"};
    // Segmentation mask from model (argmax class indices, any resolution)
    Input<Buffer<uint8_t, 2>> seg_mask{"seg_mask"};
    // Which class index is foreground (e.g., 15 = "person" in DeepLab)
    Input<int32_t> fg_class{"fg_class"};
    // Edge feathering sharpness (1.0 = linear, 3-5 = natural, >10 = hard)
    Input<float> edge_softness{"edge_softness"};
    // Output RGB image (interleaved, same size as fg_image)
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        // =================================================================
        // Step 1: Soft alpha mask via bilinear upsampling
        // =================================================================
        // Same pattern as seg_portrait_blur: binary mask → bilinear → feathering
        Func mask_float("mask_float");
        Func mask_clamped = repeat_edge(seg_mask);
        mask_float(x, y) = select(mask_clamped(x, y) == cast<uint8_t>(fg_class),
                                   1.0f, 0.0f);

        Func alpha_mask("alpha_mask");

        Expr seg_w = cast<float>(seg_mask.dim(0).extent());
        Expr seg_h = cast<float>(seg_mask.dim(1).extent());
        Expr out_w = cast<float>(fg_image.dim(0).extent());
        Expr out_h = cast<float>(fg_image.dim(1).extent());

        // Pixel-center aligned mapping from output to mask
        Expr mask_x = (cast<float>(x) + 0.5f) * seg_w / out_w - 0.5f;
        Expr mask_y = (cast<float>(y) + 0.5f) * seg_h / out_h - 0.5f;

        Expr mx = cast<int>(floor(mask_x));
        Expr my = cast<int>(floor(mask_y));
        Expr mfx = mask_x - cast<float>(mx);
        Expr mfy = mask_y - cast<float>(my);

        Expr raw_alpha = mask_float(mx, my) * (1.0f - mfx) * (1.0f - mfy) +
                         mask_float(mx + 1, my) * mfx * (1.0f - mfy) +
                         mask_float(mx, my + 1) * (1.0f - mfx) * mfy +
                         mask_float(mx + 1, my + 1) * mfx * mfy;

        // Sigmoid feathering
        alpha_mask(x, y) = clamp((raw_alpha - 0.5f) * edge_softness + 0.5f,
                                  0.0f, 1.0f);

        // =================================================================
        // Step 2: Bilinear resize background to match output resolution
        // =================================================================
        Func bg_clamped = repeat_edge(bg_image);
        Func bg_float("bg_float");
        bg_float(x, y, c) = cast<float>(bg_clamped(x, y, c));

        Expr bg_w = cast<float>(bg_image.dim(0).extent());
        Expr bg_h = cast<float>(bg_image.dim(1).extent());

        // Pixel-center aligned mapping from output to background
        Expr bg_src_x = (cast<float>(x) + 0.5f) * bg_w / out_w - 0.5f;
        Expr bg_src_y = (cast<float>(y) + 0.5f) * bg_h / out_h - 0.5f;

        Expr bx = cast<int>(floor(bg_src_x));
        Expr by = cast<int>(floor(bg_src_y));
        Expr bfx = bg_src_x - cast<float>(bx);
        Expr bfy = bg_src_y - cast<float>(by);

        Func bg_sampled("bg_sampled");
        bg_sampled(x, y, c) = bg_float(bx, by, c) * (1.0f - bfx) * (1.0f - bfy) +
                              bg_float(bx + 1, by, c) * bfx * (1.0f - bfy) +
                              bg_float(bx, by + 1, c) * (1.0f - bfx) * bfy +
                              bg_float(bx + 1, by + 1, c) * bfx * bfy;

        // =================================================================
        // Step 3: Alpha composite foreground + background
        // =================================================================
        Expr alpha = alpha_mask(x, y);
        Expr fg_val = cast<float>(fg_image(x, y, c));
        Expr bg_val = bg_sampled(x, y, c);

        // output = alpha * fg + (1 - alpha) * bg
        // +0.5f for proper rounding before uint8 truncation
        output(x, y, c) = cast<uint8_t>(clamp(
            alpha * fg_val + (1.0f - alpha) * bg_val + 0.5f,
            0.0f, 255.0f));

        // =================================================================
        // Schedule
        // =================================================================

        // alpha_mask: compute per tile, reused across 3 channels
        alpha_mask.compute_at(output, yi)
                  .vectorize(x, 16, TailStrategy::GuardWithIf);

        // mask_float: compute lazily within alpha_mask
        mask_float.compute_at(alpha_mask, x);

        // bg_float: compute lazily within bg_sampled
        bg_float.compute_at(output, yi);

        // bg_sampled: compute per tile
        bg_sampled.compute_at(output, yi)
                  .reorder(c, x, y)
                  .bound(c, 0, 3)
                  .unroll(c)
                  .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Output: 32-row tiles (no heavy blur → larger tiles for better parallelism)
        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 16, TailStrategy::GuardWithIf);

        // Prefetch for next tile
        output.prefetch(fg_image, yi, yi, 2);
        output.prefetch(bg_image, y, y, 1);

        // Interleaved layout constraints for all RGB buffers
        fg_image.dim(0).set_stride(3);
        fg_image.dim(2).set_stride(1);
        fg_image.dim(2).set_bounds(0, 3);
        bg_image.dim(0).set_stride(3);
        bg_image.dim(2).set_stride(1);
        bg_image.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(SegBgReplace, seg_bg_replace)
