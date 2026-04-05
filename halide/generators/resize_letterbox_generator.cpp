// =============================================================================
// Letterbox Resize Generator (Aspect-Ratio-Preserving with Black Padding)
// =============================================================================
//
// Fits an entire image into a target rectangle without cropping or distortion.
// The image is uniformly scaled and centered, with black (0,0,0) padding
// filling any remaining space.
//
// ## Algorithm
//
// 1. Compute uniform scale = min(target_w/src_w, target_h/src_h)
//    This ensures the entire image fits within the target dimensions.
//
// 2. Compute scaled image dimensions and centering offsets:
//    scaled_w = round(src_w * scale)
//    scaled_h = round(src_h * scale)
//    offset_x = (target_w - scaled_w) / 2
//    offset_y = (target_h - scaled_h) / 2
//
// 3. For each output pixel (x, y):
//    - If inside the scaled image region: bilinear sample from source
//    - If outside (in padding area): output black (0)
//
// ## Letterbox vs Pillarbox
//
// When the source is wider than the target aspect ratio:
//   -> Padding appears on top and bottom (letterbox, like widescreen movies)
//
// When the source is taller than the target aspect ratio:
//   -> Padding appears on left and right (pillarbox)
//
// Common use case: ML preprocessing (e.g., 1920x1080 -> 640x640 square input)
//
// =============================================================================

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class ResizeLetterbox : public Generator<ResizeLetterbox> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};     // src width x height x 3
    Input<int32_t> target_w{"target_w"};           // target output width
    Input<int32_t> target_h{"target_h"};           // target output height
    Output<Buffer<uint8_t, 3>> output{"output"};

    Var x{"x"}, y{"y"}, c{"c"}, yi{"yi"};

    void generate() {
        Func clamped = repeat_edge(input);
        Func as_float("as_float");
        as_float(x, y, c) = cast<float>(clamped(x, y, c));

        // Source dimensions from buffer metadata (resolved at runtime).
        Expr src_w = cast<float>(input.dim(0).extent());
        Expr src_h = cast<float>(input.dim(1).extent());

        // Uniform scale: choose the smaller of the two scale factors
        // to ensure the entire image fits (no cropping).
        Expr sx = cast<float>(target_w) / src_w;
        Expr sy = cast<float>(target_h) / src_h;
        Expr uniform_scale = min(sx, sy);

        // Scaled image dimensions (rounded to nearest integer)
        Expr scaled_w = cast<int>(round(src_w * uniform_scale));
        Expr scaled_h = cast<int>(round(src_h * uniform_scale));

        // Centering offsets: position the scaled image in the center of the target.
        // Integer division truncates — for odd remainders, the image shifts
        // 0.5 pixels left/up, which is acceptable for display/ML use.
        Expr offset_x = (target_w - scaled_w) / 2;
        Expr offset_y = (target_h - scaled_h) / 2;

        // Check if this output pixel falls inside the scaled image region.
        Expr in_region = (x >= offset_x) && (x < offset_x + scaled_w) &&
                         (y >= offset_y) && (y < offset_y + scaled_h);

        // Map output pixel to source coordinate (bilinear, pixel-center aligned).
        // rel_x/rel_y: position relative to the top-left of the scaled image.
        Expr rel_x = cast<float>(x - offset_x);
        Expr rel_y = cast<float>(y - offset_y);
        Expr src_x = (rel_x + 0.5f) / uniform_scale - 0.5f;
        Expr src_y = (rel_y + 0.5f) / uniform_scale - 0.5f;

        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // No unsafe_promise_clamped here: padding pixels (outside in_region)
        // are speculatively evaluated by SIMD vectorization, producing source
        // coordinates far outside [-1, extent). The repeat_edge boundary
        // condition on `clamped` safely handles all out-of-range coordinates
        // via branchless min/max clamping.

        // Bilinear interpolation (same as resize_bilinear)
        Expr val = as_float(ix, iy, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix + 1, iy, c) * fx * (1.0f - fy) +
                   as_float(ix, iy + 1, c) * (1.0f - fx) * fy +
                   as_float(ix + 1, iy + 1, c) * fx * fy;

        // select: image pixel if inside region, black (0) if in padding.
        // Note: Halide may still evaluate the bilinear expression for padding
        // pixels (speculative execution for SIMD), but the result is discarded.
        output(x, y, c) = select(in_region,
            cast<uint8_t>(clamp(val, 0.0f, 255.0f)),
            cast<uint8_t>(0));

        // --- Schedule ---
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Interleaved layout constraints
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeLetterbox, resize_letterbox)
