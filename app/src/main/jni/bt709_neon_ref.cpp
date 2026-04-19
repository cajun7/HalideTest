#include "bt709_neon_ref.h"

#include <algorithm>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define BT709_HAS_NEON 1
#else
#define BT709_HAS_NEON 0
#endif

namespace bt709 {

namespace {

inline uint8_t clamp_u8(int v) {
    return (uint8_t)std::clamp(v, 0, 255);
}

// Convert a single NV21 pixel. Chroma is shared with the neighboring column
// (NV21 is 4:2:0); caller is responsible for having looked up the right VU pair.
inline void convert_pixel(int Y, int V_m128, int U_m128, uint8_t* rgb_out) {
    int y_scaled = Y * 256 + 128;
    rgb_out[0] = clamp_u8((y_scaled + 403 * V_m128) >> 8);
    rgb_out[1] = clamp_u8((y_scaled -  48 * U_m128 - 120 * V_m128) >> 8);
    rgb_out[2] = clamp_u8((y_scaled + 475 * U_m128) >> 8);
}

}  // namespace

void nv21_to_rgb_bt709_full_range_scalar(
    const uint8_t* y,   int y_stride,
    const uint8_t* uv,  int uv_stride,
    uint8_t*       rgb, int rgb_stride,
    int width, int height) {
    for (int row = 0; row < height; ++row) {
        const uint8_t* y_row = y + row * y_stride;
        const uint8_t* uv_row = uv + (row / 2) * uv_stride;
        uint8_t* rgb_row = rgb + row * rgb_stride;
        for (int x = 0; x < width; ++x) {
            int uv_base = (x / 2) * 2;
            int V = (int)uv_row[uv_base]     - 128;
            int U = (int)uv_row[uv_base + 1] - 128;
            convert_pixel((int)y_row[x], V, U, rgb_row + x * 3);
        }
    }
}

#if BT709_HAS_NEON

namespace {

// Duplicate each of 8 chroma bytes across 2 adjacent lanes:
//   [c0 c1 c2 c3 c4 c5 c6 c7]  ->  [c0 c0 c1 c1 c2 c2 c3 c3 c4 c4 c5 c5 c6 c6 c7 c7]
inline uint8x16_t dup_each_chroma(uint8x8_t c8) {
    uint8x16_t c_ext = vcombine_u8(c8, c8);
    return vzip1q_u8(c_ext, c_ext);
}

// y_scaled = (int16)Y * 256 + 128, widened to int32x4.
inline int32x4_t y_scaled_i32(int16x4_t y4) {
    return vaddq_s32(vshll_n_s16(y4, 8), vdupq_n_s32(128));
}

// Convert one 16-Y-pixel strip. V/U must already be duplicated
// (each chroma covers 2 adjacent Y columns) and 128-centered to int16.
inline void convert_16(
    uint8x16_t Y,
    int16x8_t V_lo, int16x8_t V_hi,
    int16x8_t U_lo, int16x8_t U_hi,
    uint8_t* rgb_ptr) {
    int16x8_t Y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(Y)));
    int16x8_t Y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(Y)));

    int32x4_t ys_00 = y_scaled_i32(vget_low_s16(Y_lo));
    int32x4_t ys_01 = y_scaled_i32(vget_high_s16(Y_lo));
    int32x4_t ys_10 = y_scaled_i32(vget_low_s16(Y_hi));
    int32x4_t ys_11 = y_scaled_i32(vget_high_s16(Y_hi));

    // R = y_scaled + 403*V
    int32x4_t r00 = vmlal_n_s16(ys_00, vget_low_s16(V_lo),  403);
    int32x4_t r01 = vmlal_n_s16(ys_01, vget_high_s16(V_lo), 403);
    int32x4_t r10 = vmlal_n_s16(ys_10, vget_low_s16(V_hi),  403);
    int32x4_t r11 = vmlal_n_s16(ys_11, vget_high_s16(V_hi), 403);

    // G = y_scaled - 48*U - 120*V
    int32x4_t g00 = vmlsl_n_s16(ys_00, vget_low_s16(U_lo),   48);
    g00          = vmlsl_n_s16(g00,   vget_low_s16(V_lo),  120);
    int32x4_t g01 = vmlsl_n_s16(ys_01, vget_high_s16(U_lo),  48);
    g01          = vmlsl_n_s16(g01,   vget_high_s16(V_lo), 120);
    int32x4_t g10 = vmlsl_n_s16(ys_10, vget_low_s16(U_hi),   48);
    g10          = vmlsl_n_s16(g10,   vget_low_s16(V_hi),  120);
    int32x4_t g11 = vmlsl_n_s16(ys_11, vget_high_s16(U_hi),  48);
    g11          = vmlsl_n_s16(g11,   vget_high_s16(V_hi), 120);

    // B = y_scaled + 475*U
    int32x4_t b00 = vmlal_n_s16(ys_00, vget_low_s16(U_lo),  475);
    int32x4_t b01 = vmlal_n_s16(ys_01, vget_high_s16(U_lo), 475);
    int32x4_t b10 = vmlal_n_s16(ys_10, vget_low_s16(U_hi),  475);
    int32x4_t b11 = vmlal_n_s16(ys_11, vget_high_s16(U_hi), 475);

    // Saturating unsigned narrow with rounding:
    //   vqrshrun_n_s32(acc, 8) = clamp((acc + (1<<7)) >> 8, 0, 65535) as uint16.
    // The +128 rounding bias is already in `acc` (via y_scaled), so we use
    // vqshrun (no-round) instead -- Halide's expression is plain `>> 8`, not
    // rounding shift. vqshrun gives `acc >> 8` saturated to [0, 65535].
    uint16x4_t r16_00 = vqshrun_n_s32(r00, 8);
    uint16x4_t r16_01 = vqshrun_n_s32(r01, 8);
    uint16x4_t r16_10 = vqshrun_n_s32(r10, 8);
    uint16x4_t r16_11 = vqshrun_n_s32(r11, 8);
    uint16x4_t g16_00 = vqshrun_n_s32(g00, 8);
    uint16x4_t g16_01 = vqshrun_n_s32(g01, 8);
    uint16x4_t g16_10 = vqshrun_n_s32(g10, 8);
    uint16x4_t g16_11 = vqshrun_n_s32(g11, 8);
    uint16x4_t b16_00 = vqshrun_n_s32(b00, 8);
    uint16x4_t b16_01 = vqshrun_n_s32(b01, 8);
    uint16x4_t b16_10 = vqshrun_n_s32(b10, 8);
    uint16x4_t b16_11 = vqshrun_n_s32(b11, 8);

    // Final saturating narrow uint16 -> uint8 clamps to [0, 255].
    uint8x16x3_t rgb_out;
    rgb_out.val[0] = vcombine_u8(
        vqmovn_u16(vcombine_u16(r16_00, r16_01)),
        vqmovn_u16(vcombine_u16(r16_10, r16_11)));
    rgb_out.val[1] = vcombine_u8(
        vqmovn_u16(vcombine_u16(g16_00, g16_01)),
        vqmovn_u16(vcombine_u16(g16_10, g16_11)));
    rgb_out.val[2] = vcombine_u8(
        vqmovn_u16(vcombine_u16(b16_00, b16_01)),
        vqmovn_u16(vcombine_u16(b16_10, b16_11)));

    vst3q_u8(rgb_ptr, rgb_out);
}

}  // namespace

void nv21_to_rgb_bt709_full_range_neon(
    const uint8_t* y,   int y_stride,
    const uint8_t* uv,  int uv_stride,
    uint8_t*       rgb, int rgb_stride,
    int width, int height) {
    const int16x8_t k128 = vdupq_n_s16(128);

    for (int row = 0; row < height; row += 2) {
        const uint8_t* y_row0 = y + row * y_stride;
        const uint8_t* y_row1 = (row + 1 < height) ? y + (row + 1) * y_stride : nullptr;
        const uint8_t* uv_row = uv + (row / 2) * uv_stride;
        uint8_t* rgb_row0 = rgb + row * rgb_stride;
        uint8_t* rgb_row1 = (row + 1 < height) ? rgb + (row + 1) * rgb_stride : nullptr;

        int x = 0;
        for (; x + 16 <= width; x += 16) {
            // Load & de-interleave 16 UV bytes -> 8 V samples, 8 U samples.
            // vld2_u8 splits even bytes into val[0], odd into val[1].
            // NV21 stores V at even offsets, U at odd, so:
            uint8x8x2_t vu = vld2_u8(uv_row + x);
            uint8x16_t V16 = dup_each_chroma(vu.val[0]);
            uint8x16_t U16 = dup_each_chroma(vu.val[1]);

            int16x8_t V_lo = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(V16))),  k128);
            int16x8_t V_hi = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(V16))), k128);
            int16x8_t U_lo = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(U16))),  k128);
            int16x8_t U_hi = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(U16))), k128);

            uint8x16_t Y0 = vld1q_u8(y_row0 + x);
            convert_16(Y0, V_lo, V_hi, U_lo, U_hi, rgb_row0 + x * 3);

            if (y_row1) {
                uint8x16_t Y1 = vld1q_u8(y_row1 + x);
                convert_16(Y1, V_lo, V_hi, U_lo, U_hi, rgb_row1 + x * 3);
            }
        }

        // Scalar tail (width % 16 != 0)
        for (; x < width; ++x) {
            int uv_base = (x / 2) * 2;
            int V = (int)uv_row[uv_base]     - 128;
            int U = (int)uv_row[uv_base + 1] - 128;
            convert_pixel((int)y_row0[x], V, U, rgb_row0 + x * 3);
            if (y_row1) {
                convert_pixel((int)y_row1[x], V, U, rgb_row1 + x * 3);
            }
        }
    }
}

#else  // !BT709_HAS_NEON

void nv21_to_rgb_bt709_full_range_neon(
    const uint8_t* y,   int y_stride,
    const uint8_t* uv,  int uv_stride,
    uint8_t*       rgb, int rgb_stride,
    int width, int height) {
    nv21_to_rgb_bt709_full_range_scalar(y, y_stride, uv, uv_stride,
                                        rgb, rgb_stride, width, height);
}

#endif  // BT709_HAS_NEON

}  // namespace bt709
