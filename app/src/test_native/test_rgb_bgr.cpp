#include "test_common.h"
#include "rgb_bgr_optimized.h"

class RgbBgrTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(RgbBgrTest, RgbToBgr_MatchesOpenCV) {
    auto [width, height] = GetParam();

    // Create test image in BGR (OpenCV default)
    cv::Mat bgr = make_test_image_bgr(width, height);

    // Convert BGR -> RGB for Halide input
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCV reference: RGB -> BGR is just cvtColor
    cv::Mat opencv_result;
    cv::cvtColor(rgb, opencv_result, cv::COLOR_RGB2BGR);

    // Halide: RGB -> BGR
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = rgb_bgr_optimized(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide rgb_bgr_optimized failed with error " << err;

    // Compare: Halide output is BGR (interleaved), OpenCV result is BGR
    // Both are in the same channel order now, so compare directly without channel swap
    ASSERT_EQ(output_buf.width(), opencv_result.cols);
    ASSERT_EQ(output_buf.height(), opencv_result.rows);

    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                uint8_t h_val = output_buf(x, y, c);
                uint8_t cv_val = opencv_result.at<cv::Vec3b>(y, x)[c];
                if (h_val != cv_val) {
                    mismatches++;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Channel swap should be exact (no rounding)";
}

TEST_P(RgbBgrTest, BgrToRgb_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);

    // OpenCV reference: BGR -> RGB
    cv::Mat opencv_result;
    cv::cvtColor(bgr, opencv_result, cv::COLOR_BGR2RGB);

    // Halide: BGR -> RGB (same function, swap is symmetric)
    auto input_buf = mat_to_halide_interleaved(bgr);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = rgb_bgr_optimized(input_buf, output_buf);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                uint8_t h_val = output_buf(x, y, c);
                uint8_t cv_val = opencv_result.at<cv::Vec3b>(y, x)[c];
                if (h_val != cv_val) {
                    mismatches++;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Channel swap should be exact";
}

TEST_P(RgbBgrTest, RoundTrip_IsIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_bgr(width, height);  // any data works
    auto input_buf = mat_to_halide_interleaved(rgb);

    Halide::Runtime::Buffer<uint8_t> intermediate =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);
    Halide::Runtime::Buffer<uint8_t> roundtrip =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    rgb_bgr_optimized(input_buf, intermediate);
    rgb_bgr_optimized(intermediate, roundtrip);

    // Round-trip should be pixel-perfect identity
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(input_buf(x, y, c), roundtrip(x, y, c))
                    << "Round-trip mismatch at (" << x << "," << y << "," << c << ")";
            }
        }
    }
}

// Structural verification: output(x,y,0)==input(x,y,2), output(x,y,1)==input(x,y,1),
// output(x,y,2)==input(x,y,0) for every pixel. Verifies exact byte-swap semantics.
TEST_P(RgbBgrTest, ChannelSwap_StructuralVerify) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_rgb(width, height);
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = rgb_bgr_optimized(input_buf, output_buf);
    ASSERT_EQ(err, 0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            ASSERT_EQ(output_buf(x, y, 0), input_buf(x, y, 2))
                << "R->B swap failed at (" << x << "," << y << ")";
            ASSERT_EQ(output_buf(x, y, 1), input_buf(x, y, 1))
                << "G passthrough failed at (" << x << "," << y << ")";
            ASSERT_EQ(output_buf(x, y, 2), input_buf(x, y, 0))
                << "B->R swap failed at (" << x << "," << y << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RgbBgrTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
