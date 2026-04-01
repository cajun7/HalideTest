#include "test_common.h"
#include "flip_horizontal.h"
#include "flip_vertical.h"

class FlipTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Test horizontal flip against OpenCV
TEST_P(FlipTest, Horizontal_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCV reference: flipCode=1 = horizontal (flip around y-axis)
    cv::Mat opencv_result;
    cv::flip(bgr, opencv_result, 1);

    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = flip_horizontal(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide flip_horizontal failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/0, /*opencv_is_bgr=*/true);
}

// Test vertical flip against OpenCV
TEST_P(FlipTest, Vertical_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCV reference: flipCode=0 = vertical (flip around x-axis)
    cv::Mat opencv_result;
    cv::flip(bgr, opencv_result, 0);

    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = flip_vertical(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide flip_vertical failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/0, /*opencv_is_bgr=*/true);
}

// Test that double horizontal flip = identity (exact)
TEST_P(FlipTest, DoubleHorizontal_IsIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_rgb(width, height);
    auto buf0 = mat_to_halide_interleaved(rgb);

    auto buf1 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);
    auto buf2 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    ASSERT_EQ(flip_horizontal(buf0, buf1), 0);
    ASSERT_EQ(flip_horizontal(buf1, buf2), 0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(buf0(x, y, c), buf2(x, y, c))
                    << "Double flip-H not identity at (" << x << "," << y << "," << c << ")";
            }
        }
    }
}

// Test that double vertical flip = identity (exact)
TEST_P(FlipTest, DoubleVertical_IsIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_rgb(width, height);
    auto buf0 = mat_to_halide_interleaved(rgb);

    auto buf1 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);
    auto buf2 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    ASSERT_EQ(flip_vertical(buf0, buf1), 0);
    ASSERT_EQ(flip_vertical(buf1, buf2), 0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(buf0(x, y, c), buf2(x, y, c))
                    << "Double flip-V not identity at (" << x << "," << y << "," << c << ")";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    FlipTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
