#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

// ---------------------------------------------------------------------------
// Letterbox Resize (Aspect-ratio-preserving bilinear resize)
//
// Computes a uniform scale factor so the entire source image fits within
// the target dimensions without cropping or stretching. The image is
// centered, with black (0,0,0) padding filling any remaining space
// (letterbox for wide images, pillarbox for tall images).
//
// Guarantees:
//   - Image composition is preserved (no crop, no distortion)
//   - Aspect ratio is exactly maintained
//   - Bilinear interpolation with pixel-center alignment
// ---------------------------------------------------------------------------
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

        // Source dimensions from buffer metadata
        Expr src_w = cast<float>(input.dim(0).extent());
        Expr src_h = cast<float>(input.dim(1).extent());

        // Uniform scale: fit entire source into target without cropping
        Expr sx = cast<float>(target_w) / src_w;
        Expr sy = cast<float>(target_h) / src_h;
        Expr uniform_scale = min(sx, sy);

        // Scaled image dimensions (rounded to int)
        Expr scaled_w = cast<int>(round(src_w * uniform_scale));
        Expr scaled_h = cast<int>(round(src_h * uniform_scale));

        // Centering offsets
        Expr offset_x = (target_w - scaled_w) / 2;
        Expr offset_y = (target_h - scaled_h) / 2;

        // Check if this output pixel falls inside the scaled image region
        Expr in_region = (x >= offset_x) && (x < offset_x + scaled_w) &&
                         (y >= offset_y) && (y < offset_y + scaled_h);

        // Map output pixel to source coordinate (bilinear, pixel-center aligned)
        Expr rel_x = cast<float>(x - offset_x);
        Expr rel_y = cast<float>(y - offset_y);
        Expr src_x = (rel_x + 0.5f) / uniform_scale - 0.5f;
        Expr src_y = (rel_y + 0.5f) / uniform_scale - 0.5f;

        Expr ix = cast<int>(floor(src_x));
        Expr iy = cast<int>(floor(src_y));
        Expr fx = src_x - cast<float>(ix);
        Expr fy = src_y - cast<float>(iy);

        // Promise coordinates are in valid range for repeat_edge.
        Expr ix_s = unsafe_promise_clamped(ix, -1, input.dim(0).extent());
        Expr iy_s = unsafe_promise_clamped(iy, -1, input.dim(1).extent());

        // Bilinear interpolation: 4-tap (2x2 neighborhood)
        Expr val = as_float(ix_s, iy_s, c) * (1.0f - fx) * (1.0f - fy) +
                   as_float(ix_s + 1, iy_s, c) * fx * (1.0f - fy) +
                   as_float(ix_s, iy_s + 1, c) * (1.0f - fx) * fy +
                   as_float(ix_s + 1, iy_s + 1, c) * fx * fy;

        // Output: interpolated pixel inside region, black outside
        output(x, y, c) = select(in_region,
            cast<uint8_t>(clamp(val, 0.0f, 255.0f)),
            cast<uint8_t>(0));

        // --- Schedule ---
        as_float.compute_at(output, yi);

        output.reorder(c, x, y)
              .bound(c, 0, 3)
              .unroll(c)
              .split(y, y, yi, 32)
              .parallel(y)
              .vectorize(x, 8, TailStrategy::GuardWithIf);

        // Interleaved layout: channel stride = 1, x stride = 3
        input.dim(0).set_stride(3);
        input.dim(2).set_stride(1);
        input.dim(2).set_bounds(0, 3);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(ResizeLetterbox, resize_letterbox)
