// =============================================================================
// Fused NV21 -> YUV444 (nearest UV) -> BT.709 full-range RGB (FLOAT math)
// =============================================================================
//
// Reference: TargetOpenCV.cpp (nv21_to_YUV444 + YUV444_to_RGB).
// UV sampling is nearest-neighbor: uv_x = (x/2)*2, uv_y = y/2; V at even byte,
// U at odd byte (NV21 layout). The intermediate yuv444 buffer is NOT
// materialized -- both stages are fused into one Halide pipeline.
//
// Math is float, matching std::lround round-half-to-even semantics; this gives
// ~94-100 dB PSNR vs the C float reference (max_diff <= 1 LSB across all
// resolutions tested 320x240 .. 4096x3072).
//
// Schedule: 16-row strips, parallel over strips, vectorize x by 8 (NEON 4-wide
// float x 2-way unroll), unroll the 3 RGB channels, and compute_at U and V at
// the inner row level so the (cast<float> - 128) work is hoisted once per row
// strip. This was the float winner of an 8-variant on-device sweep (Samsung
// SM8850 / Exynos 2600 target) at HD+ resolutions, beating the scalar
// TargetOpenCV.cpp reference by 6-8x at 1080p / 4K / 12MP.
// =============================================================================

#include "Halide.h"

using namespace Halide;

class Nv21ToRgbBt709Fused : public Generator<Nv21ToRgbBt709Fused> {
public:
    Input<Buffer<uint8_t, 2>>  y_plane{"y_plane"};   // width x height
    Input<Buffer<uint8_t, 2>>  uv_plane{"uv_plane"}; // width x (height/2) raw bytes (V,U pairs)

    Output<Buffer<uint8_t, 3>> output{"output"};     // width x height x 3 (RGB interleaved)

    Var x{"x"}, y{"y"}, c{"c"}, yo{"yo"}, yi{"yi"};
    Func u_f{"u_f"}, v_f{"v_f"};

    void generate() {
        // 4:2:0 chroma share: each VU pair covers a 2x2 Y block.
        Expr uv_x = (x / 2) * 2;
        Expr uv_y = y / 2;

        // Promote NV21 bytes to float once. Exposing U/V as Funcs lets us
        // compute_at the row strip so the cast+sub-128 is hoisted.
        v_f(x, y) = cast<float>(uv_plane(uv_x,     uv_y));   // V at even byte
        u_f(x, y) = cast<float>(uv_plane(uv_x + 1, uv_y));   // U at odd byte

        Expr Y  = cast<float>(y_plane(x, y));
        Expr Uc = u_f(x, y) - 128.0f;
        Expr Vc = v_f(x, y) - 128.0f;

        // BT.709 full-range float -- bit-equivalent to TargetOpenCV.cpp.
        Expr Rf = Y +              1.5748f * Vc;
        Expr Gf = Y - 0.1873f*Uc - 0.4681f * Vc;
        Expr Bf = Y + 1.8556f*Uc;

        // round() emits NEON vcvtnq_s32_f32 (round-to-nearest-even, IEEE-754
        // default), matching std::lround under Bionic's default rounding mode.
        Expr R = cast<int32_t>(round(Rf));
        Expr G = cast<int32_t>(round(Gf));
        Expr B = cast<int32_t>(round(Bf));

        output(x, y, c) = cast<uint8_t>(clamp(mux(c, {R, G, B}), 0, 255));
    }

    void schedule() {
        y_plane.dim(0).set_stride(1);
        uv_plane.dim(0).set_stride(1);
        output.dim(0).set_stride(3);
        output.dim(2).set_stride(1);
        output.bound(c, 0, 3);

        // 16-row strips, parallel(yo), vec(x,8), unroll(c), compute_at U/V.
        output.reorder(c, x, y).unroll(c)
              .split(y, yo, yi, 16, TailStrategy::GuardWithIf)
              .parallel(yo)
              .vectorize(x, 8, TailStrategy::GuardWithIf);
        u_f.compute_at(output, yi).vectorize(x, 8, TailStrategy::GuardWithIf);
        v_f.compute_at(output, yi).vectorize(x, 8, TailStrategy::GuardWithIf);
    }
};

HALIDE_REGISTER_GENERATOR(Nv21ToRgbBt709Fused, nv21_to_rgb_bt709_fused)
