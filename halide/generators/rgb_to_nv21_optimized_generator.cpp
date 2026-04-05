// =============================================================================
// RGB to NV21 Optimized Generator (BT.601 Limited-Range)
// =============================================================================
//
// Same BT.601 forward transform as rgb_to_nv21_generator.cpp, but with
// scheduling optimizations:
//
//   1. compute_at for cr_full/cb_full — the baseline computes full-resolution
//      Cr and Cb at every 2x2 block demand site, causing redundant recomputation.
//      This version uses compute_at(uv_output, uv_yi) so each Cr/Cb value is
//      computed once per tile and reused.
//
//   2. Y-axis tiling — split(y, yo, yi, 32) for Y output and
//      split(y, yo, uv_yi, 16) for UV output, matching the 2:1 vertical ratio.
//
//   3. Wider vectorization for Y plane (16 pixels per SIMD iteration).
//
// =============================================================================

#include "Halide.h"

using namespace Halide;

class RgbToNv21Optimized : public Generator<RgbToNv21Optimized> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};          // width x height x 3 (RGB interleaved)
    Output<Buffer<uint8_t, 2>> y_output{"y_output"};   // width x height
    Output<Buffer<uint8_t, 2>> uv_output{"uv_output"}; // width x (height/2) raw bytes

    // Declared as member Funcs so compute_at() can reference them in schedule()
    Func r_val{"r_val"}, g_val{"g_val"}, b_val{"b_val"};
    Func cb_full{"cb_full"}, cr_full{"cr_full"};

    Var x{"x"}, y{"y"}, yo{"yo"}, yi{"yi"}, uv_yi{"uv_yi"};

    void generate() {
        // Extract R, G, B as int32
        r_val(x, y) = cast<int32_t>(input(x, y, 0));
        g_val(x, y) = cast<int32_t>(input(x, y, 1));
        b_val(x, y) = cast<int32_t>(input(x, y, 2));

        // --- Y output: full resolution ---
        Expr r = r_val(x, y);
        Expr g = g_val(x, y);
        Expr b = b_val(x, y);
        Expr y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        y_output(x, y) = cast<uint8_t>(clamp(y_val, 0, 255));

        // --- UV output: half resolution with 2x2 block averaging ---
        Expr bx = (x / 2) * 2;
        Expr by = 2 * y;

        // Full-resolution Cb (U) and Cr (V)
        cb_full(x, y) = ((-38 * r_val(x, y) - 74 * g_val(x, y) + 112 * b_val(x, y) + 128) >> 8) + 128;
        cr_full(x, y) = ((112 * r_val(x, y) - 94 * g_val(x, y) - 18 * b_val(x, y) + 128) >> 8) + 128;

        // 2x2 block average
        Expr cr_avg = (cr_full(bx, by) + cr_full(bx + 1, by) +
                       cr_full(bx, by + 1) + cr_full(bx + 1, by + 1) + 2) / 4;
        Expr cb_avg = (cb_full(bx, by) + cb_full(bx + 1, by) +
                       cb_full(bx, by + 1) + cb_full(bx + 1, by + 1) + 2) / 4;

        // NV21: V at even byte offsets, U at odd
        Expr is_v = (x % 2) == 0;
        uv_output(x, y) = cast<uint8_t>(clamp(
            select(is_v, cr_avg, cb_avg), 0, 255));
    }

    void schedule() {
        // OPTIMIZATION: Y with tiling and wider vectorization
        y_output.split(y, yo, yi, 32, TailStrategy::GuardWithIf)
                .vectorize(x, 16, TailStrategy::GuardWithIf)
                .parallel(yo);

        // OPTIMIZATION: UV with tiling
        uv_output.split(y, yo, uv_yi, 16, TailStrategy::GuardWithIf)
                 .vectorize(x, 8, TailStrategy::GuardWithIf)
                 .parallel(yo);

        // OPTIMIZATION: compute_at for cr_full/cb_full
        // This keeps the full-resolution Cr/Cb values in cache within each UV tile,
        // avoiding redundant recomputation of the RGB->Cr/Cb transform.
        cr_full.compute_at(uv_output, uv_yi);
        cb_full.compute_at(uv_output, uv_yi);

        // Input layout
        input.dim(2).set_bounds(0, 3);
        uv_output.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(RgbToNv21Optimized, rgb_to_nv21_optimized)
