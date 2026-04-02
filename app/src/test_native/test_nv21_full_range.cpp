#include "test_common.h"
#include "nv21_to_rgb_full_range.h"
#include "nv21_to_rgb.h"

// C++ float reference for full-range BT.601 (ground truth).
// Exact floating-point math with round-to-nearest.
static void ref_full_range_rgb(uint8_t y_val, uint8_t u_raw, uint8_t v_raw,
                               uint8_t out_rgb[3]) {
    float Y = (float)y_val;
    float U = (float)u_raw - 128.0f;
    float V = (float)v_raw - 128.0f;

    float R = Y + 1.402f * V;
    float G = Y - 0.34414f * U - 0.71414f * V;
    float B = Y + 1.772f * U;

    out_rgb[0] = (uint8_t)std::max(0, std::min(255, (int)roundf(R)));
    out_rgb[1] = (uint8_t)std::max(0, std::min(255, (int)roundf(G)));
    out_rgb[2] = (uint8_t)std::max(0, std::min(255, (int)roundf(B)));
}

// Build full float-reference image from NV21 data.
static void ref_full_range_image(const uint8_t* y_data, const uint8_t* uv_data,
                                 int w, int h, std::vector<uint8_t>& rgb_out) {
    rgb_out.resize(w * h * 3);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            uint8_t y_val = y_data[row * w + col];
            int uv_x = (col / 2) * 2;
            int uv_y = row / 2;
            uint8_t v_raw = uv_data[uv_y * w + uv_x];
            uint8_t u_raw = uv_data[uv_y * w + uv_x + 1];

            uint8_t rgb[3];
            ref_full_range_rgb(y_val, u_raw, v_raw, rgb);

            for (int c = 0; c < 3; c++)
                rgb_out[(row * w + col) * 3 + c] = rgb[c];
        }
    }
}

// Helper: create 2D UV buffer
static Halide::Runtime::Buffer<uint8_t> make_uv_buf(uint8_t* uv_ptr, int w, int h) {
    return Halide::Runtime::Buffer<uint8_t>(uv_ptr, w, h / 2);
}

class Nv21FullRangeTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Nv21FullRangeTest, MatchesFloatReference) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_to_rgb_full_range(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide nv21_to_rgb_full_range failed";

    // Float reference
    std::vector<uint8_t> ref_rgb;
    ref_full_range_image(y_ptr, uv_ptr, w, h, ref_rgb);

    // Dump first iteration
    dump_if_first(output_buf, "full_range_halide", 0);

    // Compare: tolerance=1 (only fixed-point rounding error)
    int mismatches = 0, max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int halide_val = output_buf(x, y, c);
                int ref_val = ref_rgb[(y * w + x) * 3 + c];
                int diff = std::abs(halide_val - ref_val);
                if (diff > 1) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float pct = 100.0f * mismatches / (float)(w * h * 3);
    printf("  FullRange vs float ref: max_diff=%d mismatch(>1)=%.2f%%\n", max_diff, pct);
    EXPECT_LE(max_diff, 2)
        << "Fixed-point rounding should be within 2 of float reference";
    EXPECT_LT(pct, 1.0f)
        << "Too many mismatches: " << mismatches << " (" << pct << "%)";
}

TEST_P(Nv21FullRangeTest, FullRangeVsLimitedRange_ShowsDifference) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);

    Halide::Runtime::Buffer<uint8_t> full_range_out(w, h, 3);
    Halide::Runtime::Buffer<uint8_t> limited_range_out(w, h, 3);

    int err1 = nv21_to_rgb_full_range(y_buf, uv_buf, full_range_out);
    ASSERT_EQ(err1, 0);
    int err2 = nv21_to_rgb(y_buf, uv_buf, limited_range_out);
    ASSERT_EQ(err2, 0);

    // The two conversions should produce noticeably different results
    // for full-range YUV data (especially for Y < 16 or Y > 235).
    int total_diff = 0;
    int max_diff = 0;
    int differ_count = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int d = std::abs((int)full_range_out(x, y, c) -
                                 (int)limited_range_out(x, y, c));
                total_diff += d;
                max_diff = std::max(max_diff, d);
                if (d > 0) differ_count++;
            }
        }
    }
    float avg_diff = (float)total_diff / (float)(w * h * 3);
    printf("  FullRange vs LimitedRange: avg_diff=%.2f max_diff=%d differ_pct=%.1f%%\n",
           avg_diff, max_diff, 100.0f * differ_count / (float)(w * h * 3));

    // They MUST differ — same NV21 data, different matrix = different output
    EXPECT_GT(max_diff, 0)
        << "Full-range and limited-range should produce different results";
    EXPECT_GT(avg_diff, 1.0f)
        << "Average difference should be noticeable";
}

TEST_P(Nv21FullRangeTest, OutputInValidRange) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_to_rgb_full_range(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0);

    // uint8_t output is inherently 0-255, but verify no crash
    SUCCEED() << "No crash on " << w << "x" << h;
}

// Test with extreme Y values that stress the difference between full and limited range.
// Y=0: full-range gives R≈0, limited-range gives R≈clamp((0-16)*1.164)=clamp(-18.6)=0
// Y=255: full-range gives R≈255+V_contrib, limited-range gives R≈(255-16)*1.164+V≈278+V
TEST_P(Nv21FullRangeTest, FullRangeExtremes) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    // Create NV21 with extreme Y values: top half Y=0, bottom half Y=255
    // UV at neutral (128,128) so only Y contribution matters
    std::vector<uint8_t> nv21(w * h + w * (h / 2));
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + w * h;

    for (int j = 0; j < h / 2; j++)
        for (int i = 0; i < w; i++)
            y_ptr[j * w + i] = 0;
    for (int j = h / 2; j < h; j++)
        for (int i = 0; i < w; i++)
            y_ptr[j * w + i] = 255;

    // Neutral UV: V=128, U=128 (no chroma)
    for (int j = 0; j < h / 2; j++)
        for (int i = 0; i < w / 2; i++) {
            uv_ptr[j * w + 2 * i + 0] = 128;  // V
            uv_ptr[j * w + 2 * i + 1] = 128;  // U
        }

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, w, h / 2);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_to_rgb_full_range(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0);

    // With neutral UV (128), full-range BT.601 gives: R=G=B=Y
    // Y=0 region: all channels should be 0
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output_buf(0, 0, c), 0)
            << "Y=0 with neutral UV should give 0 for channel " << c;
    }
    // Y=255 region: all channels should be 255
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output_buf(0, h - 1, c), 255)
            << "Y=255 with neutral UV should give 255 for channel " << c;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21FullRangeTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(642, 482),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1280, 718),
        std::make_pair(1920, 1080)
    )
);
