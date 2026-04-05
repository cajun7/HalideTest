#include "test_common.h"
#include "halide_ops.h"

class RgbResizeOptimizedTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(RgbResizeOptimizedTest, BilinearOptimized_MatchesOpenCV) {
    auto [src_w, src_h] = GetParam();
    int tw = src_w / 2, th = src_h / 2;

    cv::Mat rgb_cv = make_test_image_rgb(src_w, src_h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    // Halide optimized
    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    int ret = halide_ops::resize_bilinear_optimized(halide_in, tw, th, halide_out);
    ASSERT_EQ(ret, 0);

    // OpenCV reference
    cv::Mat cv_out;
    cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    compare_buffers_rgb(halide_out, cv_out, 2, /*opencv_is_bgr=*/false);
}

TEST_P(RgbResizeOptimizedTest, AreaOptimized_MatchesOpenCV) {
    auto [src_w, src_h] = GetParam();
    int tw = src_w / 2, th = src_h / 2;

    cv::Mat rgb_cv = make_test_image_rgb(src_w, src_h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    int ret = halide_ops::resize_area_optimized(halide_in, tw, th, halide_out);
    ASSERT_EQ(ret, 0);

    cv::Mat cv_out;
    cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

    compare_buffers_rgb(halide_out, cv_out, 3, /*opencv_is_bgr=*/false);
}

TEST_P(RgbResizeOptimizedTest, BicubicOptimized_MatchesOpenCV) {
    auto [src_w, src_h] = GetParam();
    int tw = src_w / 2, th = src_h / 2;

    cv::Mat rgb_cv = make_test_image_rgb(src_w, src_h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    int ret = halide_ops::resize_bicubic_optimized(halide_in, tw, th, halide_out);
    ASSERT_EQ(ret, 0);

    // OpenCV uses a=-0.75 for INTER_CUBIC, our optimized generator matches this
    cv::Mat cv_out;
    cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_CUBIC);

    compare_buffers_rgb(halide_out, cv_out, 3, /*opencv_is_bgr=*/false);
}

TEST_P(RgbResizeOptimizedTest, BilinearOptimized_PSNR) {
    auto [src_w, src_h] = GetParam();
    int tw = src_w / 2, th = src_h / 2;

    cv::Mat rgb_cv = make_test_image_rgb(src_w, src_h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    halide_ops::resize_bilinear_optimized(halide_in, tw, th, halide_out);

    cv::Mat cv_out;
    cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    // Compare directly in RGB space (both halide_out and cv_out are RGB)
    double mse = 0;
    int total = tw * th * 3;
    for (int y = 0; y < th; y++) {
        for (int x = 0; x < tw; x++) {
            for (int c = 0; c < 3; c++) {
                double diff = (double)halide_out(x, y, c) - (double)cv_out.at<cv::Vec3b>(y, x)[c];
                mse += diff * diff;
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    printf("  Bilinear PSNR: %.2f dB (%dx%d -> %dx%d)\n", psnr, src_w, src_h, tw, th);
    EXPECT_GT(psnr, 50.0) << "PSNR too low for bilinear resize";
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RgbResizeOptimizedTest,
    ::testing::Values(
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1920, 1080)
    )
);
