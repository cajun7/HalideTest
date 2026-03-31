#include "test_common.h"
#include "resize_bilinear.h"
#include "resize_bicubic.h"

class ResizeTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Test bilinear downscale (half size)
TEST_P(ResizeTest, Bilinear_HalfSize_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int out_w = width / 2;
    int out_h = height / 2;
    if (out_w < 1 || out_h < 1) return;

    float sx = (float)out_w / width;
    float sy = (float)out_h / height;

    // OpenCV reference
    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

    // Halide
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_w, out_h, 3);

    int err = resize_bilinear(input_buf, sx, sy, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_bilinear failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

// Test bilinear upscale (double size)
TEST_P(ResizeTest, Bilinear_DoubleSize_MatchesOpenCV) {
    auto [width, height] = GetParam();

    // Use smaller source to keep test fast
    int src_w = std::min(width, 320);
    int src_h = std::min(height, 240);

    cv::Mat bgr = make_test_image_bgr(src_w, src_h);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int out_w = src_w * 2;
    int out_h = src_h * 2;
    float sx = (float)out_w / src_w;
    float sy = (float)out_h / src_h;

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_w, out_h, 3);

    int err = resize_bilinear(input_buf, sx, sy, output_buf);
    ASSERT_EQ(err, 0);

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

// Test bicubic downscale
TEST_P(ResizeTest, Bicubic_HalfSize_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int out_w = width / 2;
    int out_h = height / 2;
    if (out_w < 1 || out_h < 1) return;

    float sx = (float)out_w / width;
    float sy = (float)out_h / height;

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);

    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_w, out_h, 3);

    int err = resize_bicubic(input_buf, sx, sy, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_bicubic failed";

    // Bicubic kernels may differ slightly between Halide (Catmull-Rom) and OpenCV
    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/5, /*opencv_is_bgr=*/true);
}

// Verify no crash on odd output sizes
TEST_P(ResizeTest, NoCrash_OddOutputSize) {
    auto [width, height] = GetParam();

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(128, 64, 200));
    auto input_buf = mat_to_halide_interleaved(rgb);

    // Odd output size
    int out_w = width * 3 / 4;
    int out_h = height * 3 / 4;
    if (out_w < 1 || out_h < 1) return;

    float sx = (float)out_w / width;
    float sy = (float)out_h / height;

    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(out_w, out_h, 3);

    int err = resize_bilinear(input_buf, sx, sy, output_buf);
    ASSERT_EQ(err, 0) << "Crashed on resize to " << out_w << "x" << out_h;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    ResizeTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
