#include "test_common.h"
#include "resize_bilinear_optimized.h"
#include "resize_bicubic_optimized.h"
#include "resize_area_optimized.h"

class ResizeTargetTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// ---------------------------------------------------------------------------
// Bilinear target-size resize tests
// ---------------------------------------------------------------------------

TEST_P(ResizeTargetTest, Bilinear_HalfSize_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = width / 2, th = height / 2;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bilinear_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_bilinear_optimized failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

TEST_P(ResizeTargetTest, Bilinear_DoubleSize_MatchesOpenCV) {
    auto [width, height] = GetParam();
    if (width > 960) return;  // skip large upscale to save test time
    int tw = width * 2, th = height * 2;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bilinear_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

TEST_P(ResizeTargetTest, Bilinear_OddTarget_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = 641, th = 481;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bilinear_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

// Verify output dimensions exactly match target_w/target_h
TEST_P(ResizeTargetTest, Bilinear_ExactOutputDimensions) {
    auto [width, height] = GetParam();
    int tw = 333, th = 222;

    cv::Mat rgb = make_test_image_rgb(width, height);
    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bilinear_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    ASSERT_EQ(output_buf.width(), tw);
    ASSERT_EQ(output_buf.height(), th);
}

// ---------------------------------------------------------------------------
// Bicubic target-size resize tests
// ---------------------------------------------------------------------------

TEST_P(ResizeTargetTest, Bicubic_HalfSize_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = width / 2, th = height / 2;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_CUBIC);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bicubic_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_bicubic_optimized failed";

    // Higher tolerance for bicubic due to different kernel (Catmull-Rom vs OpenCV)
    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/5, /*opencv_is_bgr=*/true);
}

TEST_P(ResizeTargetTest, Bicubic_OddTarget_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = 641, th = 481;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_CUBIC);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_bicubic_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/5, /*opencv_is_bgr=*/true);
}

// ---------------------------------------------------------------------------
// INTER_AREA target-size resize tests
// ---------------------------------------------------------------------------

TEST_P(ResizeTargetTest, Area_HalfSize_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = width / 2, th = height / 2;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_area_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_area_optimized failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/3, /*opencv_is_bgr=*/true);
}

TEST_P(ResizeTargetTest, Area_QuarterSize_MatchesOpenCV) {
    auto [width, height] = GetParam();
    int tw = width / 4, th = height / 4;
    if (tw < 1 || th < 1) return;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat opencv_result;
    cv::resize(bgr, opencv_result, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_area_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/4, /*opencv_is_bgr=*/true);
}

TEST_P(ResizeTargetTest, Area_AsymmetricTarget_NoCrash) {
    auto [width, height] = GetParam();
    // Asymmetric target: different aspect ratio
    int tw = 1920, th = 720;

    cv::Mat rgb = make_test_image_rgb(width, height);
    auto input_buf = mat_to_halide_interleaved(rgb);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = resize_area_optimized(input_buf, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Halide resize_area_optimized crashed on asymmetric target "
                       << tw << "x" << th;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    ResizeTargetTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
