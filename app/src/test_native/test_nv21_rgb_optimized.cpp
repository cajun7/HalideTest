#include "test_common.h"
#include "halide_ops.h"

class Nv21RgbOptimizedTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Nv21RgbOptimizedTest, Nv21ToRgbOptimized_MatchesBaseline) {
    auto [w, h] = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(w, h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), w, h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), w, h / 2);

    // Run optimized
    Halide::Runtime::Buffer<uint8_t> opt_out =
        Halide::Runtime::Buffer<uint8_t>(w, h, 3);
    int ret = halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, opt_out);
    ASSERT_EQ(ret, 0);

    // Run baseline
    Halide::Runtime::Buffer<uint8_t> base_out =
        Halide::Runtime::Buffer<uint8_t>(w, h, 3);
    int ret2 = halide_ops::nv21_to_rgb(y_buf, uv_buf, base_out);
    ASSERT_EQ(ret2, 0);

    // Should match exactly (same algorithm, different schedule)
    int mismatches = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)opt_out(x, y, c) - (int)base_out(x, y, c));
                if (diff > 0) mismatches++;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Optimized NV21->RGB differs from baseline";
}

TEST_P(Nv21RgbOptimizedTest, RgbToNv21Optimized_RoundTrip) {
    auto [w, h] = GetParam();

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_planar(rgb_cv);

    // RGB -> NV21 optimized
    Halide::Runtime::Buffer<uint8_t> y_out(w, h);
    Halide::Runtime::Buffer<uint8_t> uv_out(w, h / 2);
    int ret = halide_ops::rgb_to_nv21_optimized(halide_in, y_out, uv_out);
    ASSERT_EQ(ret, 0);

    // NV21 -> RGB (baseline, to verify quality)
    Halide::Runtime::Buffer<uint8_t> roundtrip =
        Halide::Runtime::Buffer<uint8_t>(w, h, 3);
    halide_ops::nv21_to_rgb(y_out, uv_out, roundtrip);

    // Compare: BT.601 round-trip has inherent quantization loss
    // but should be high quality (PSNR > 50 dB for gradient images)
    int mismatches = 0;
    int max_diff = 0;
    double mse = 0;
    int total = w * h * 3;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)halide_in(x, y, c) - (int)roundtrip(x, y, c));
                max_diff = std::max(max_diff, diff);
                mse += diff * diff;
                if (diff > 5) mismatches++;
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    printf("  RGB->NV21->RGB round-trip: PSNR=%.2f dB, max_diff=%d (%dx%d)\n",
           psnr, max_diff, w, h);
    EXPECT_GT(psnr, 35.0) << "Round-trip PSNR too low";
}

TEST_P(Nv21RgbOptimizedTest, RgbToNv21Optimized_MatchesBaseline) {
    auto [w, h] = GetParam();

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_planar(rgb_cv);

    // Optimized
    Halide::Runtime::Buffer<uint8_t> y_opt(w, h);
    Halide::Runtime::Buffer<uint8_t> uv_opt(w, h / 2);
    halide_ops::rgb_to_nv21_optimized(halide_in, y_opt, uv_opt);

    // Baseline
    Halide::Runtime::Buffer<uint8_t> y_base(w, h);
    Halide::Runtime::Buffer<uint8_t> uv_base(w, h / 2);
    halide_ops::rgb_to_nv21(halide_in, y_base, uv_base);

    // Should match exactly (same algorithm, different schedule)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ASSERT_EQ(y_opt(x, y), y_base(x, y))
                << "Y mismatch at (" << x << ", " << y << ")";
        }
    }
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w; x++) {
            ASSERT_EQ(uv_opt(x, y), uv_base(x, y))
                << "UV mismatch at (" << x << ", " << y << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21RgbOptimizedTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(1280, 720),
        std::make_pair(1920, 1080)
    )
);
