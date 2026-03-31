#include "test_common.h"
#include "gaussian_blur_y.h"
#include "gaussian_blur_rgb.h"

class GaussianBlurTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Test single-channel (Y plane) Gaussian blur against OpenCV
TEST_P(GaussianBlurTest, Y_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat gray = make_test_image_gray(width, height);

    // OpenCV reference: 5x5 Gaussian blur (matches radius=2 in our generator)
    cv::Mat opencv_result;
    cv::GaussianBlur(gray, opencv_result, cv::Size(5, 5), 0);

    // Halide
    Halide::Runtime::Buffer<uint8_t> input_buf(gray.data, width, height);
    Halide::Runtime::Buffer<uint8_t> output_buf(width, height);

    int err = gaussian_blur_y(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide gaussian_blur_y failed with error " << err;

    compare_buffers_gray(output_buf, opencv_result, /*tolerance=*/3);
}

// Test 3-channel RGB Gaussian blur against OpenCV
TEST_P(GaussianBlurTest, RGB_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);

    // OpenCV operates on BGR
    cv::Mat opencv_result;
    cv::GaussianBlur(bgr, opencv_result, cv::Size(5, 5), 0);

    // Convert BGR -> RGB for Halide input
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = gaussian_blur_rgb(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide gaussian_blur_rgb failed with error " << err;

    // Compare: Halide output (RGB) vs OpenCV result (BGR) -> set opencv_is_bgr=true
    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

// Verify no crash on odd resolutions (boundary condition safety)
TEST_P(GaussianBlurTest, Y_NoCrashOddSize) {
    auto [width, height] = GetParam();

    cv::Mat gray = make_test_image_gray(width, height);
    Halide::Runtime::Buffer<uint8_t> input_buf(gray.data, width, height);
    Halide::Runtime::Buffer<uint8_t> output_buf(width, height);

    int err = gaussian_blur_y(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Crashed on " << width << "x" << height;
}

TEST_P(GaussianBlurTest, RGB_NoCrashOddSize) {
    auto [width, height] = GetParam();

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = gaussian_blur_rgb(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Crashed on " << width << "x" << height;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    GaussianBlurTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
