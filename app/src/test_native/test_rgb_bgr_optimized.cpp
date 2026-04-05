#include "test_common.h"
#include "halide_ops.h"

class RgbBgrOptimizedTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(RgbBgrOptimizedTest, MatchesBaseline) {
    auto [w, h] = GetParam();

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    // Run optimized
    Halide::Runtime::Buffer<uint8_t> opt_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
    int ret = halide_ops::rgb_bgr_optimized(halide_in, opt_out);
    ASSERT_EQ(ret, 0);

    // Run baseline
    Halide::Runtime::Buffer<uint8_t> base_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
    int ret2 = halide_ops::rgb_bgr(halide_in, base_out);
    ASSERT_EQ(ret2, 0);

    // Must be pixel-perfect (pure index remapping, no arithmetic)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(opt_out(x, y, c), base_out(x, y, c))
                    << "Mismatch at (" << x << ", " << y << ", " << c << ")";
            }
        }
    }
}

TEST_P(RgbBgrOptimizedTest, RoundTrip_IsIdentity) {
    auto [w, h] = GetParam();

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> swapped =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
    halide_ops::rgb_bgr_optimized(halide_in, swapped);

    Halide::Runtime::Buffer<uint8_t> roundtrip =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
    halide_ops::rgb_bgr_optimized(swapped, roundtrip);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(halide_in(x, y, c), roundtrip(x, y, c));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RgbBgrOptimizedTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1920, 1080)
    )
);
