#include "test_common.h"
#include "lens_blur.h"

class LensBlurTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Compare Halide lens blur against OpenCV filter2D with a disc kernel
TEST_P(LensBlurTest, MatchesDiscFilter) {
    auto [width, height] = GetParam();
    int blur_radius = 4;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Build disc kernel for OpenCV reference
    int ksize = 2 * blur_radius + 1;
    cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);
    int count = 0;
    for (int dy = -blur_radius; dy <= blur_radius; dy++) {
        for (int dx = -blur_radius; dx <= blur_radius; dx++) {
            if (dx * dx + dy * dy <= blur_radius * blur_radius) {
                kernel.at<float>(dy + blur_radius, dx + blur_radius) = 1.0f;
                count++;
            }
        }
    }
    kernel /= (float)count;

    // OpenCV reference: filter2D with disc kernel on BGR, constant border (black)
    cv::Mat opencv_result;
    cv::filter2D(bgr, opencv_result, -1, kernel, cv::Point(-1, -1), 0,
                 cv::BORDER_CONSTANT);

    // Halide
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    int err = lens_blur(input_buf, blur_radius, output_buf);
    ASSERT_EQ(err, 0) << "Halide lens_blur failed with error " << err;

    // Compare with higher tolerance because boundary handling may differ slightly
    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/5, /*opencv_is_bgr=*/true);
}

// Verify no crash with various radii
TEST_P(LensBlurTest, NoCrashVariousRadii) {
    auto [width, height] = GetParam();

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
    auto input_buf = mat_to_halide_interleaved(rgb);

    for (int r = 1; r <= 8; r++) {
        Halide::Runtime::Buffer<uint8_t> output_buf =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

        int err = lens_blur(input_buf, r, output_buf);
        ASSERT_EQ(err, 0) << "Crashed at radius=" << r
                          << " on " << width << "x" << height;
    }
}

// Radius=0 should approximate identity (no blur)
TEST_P(LensBlurTest, RadiusZeroApproxIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_bgr(width, height);
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    // radius=0: disc contains only the center pixel -> identity
    int err = lens_blur(input_buf, 0, output_buf);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                if (input_buf(x, y, c) != output_buf(x, y, c)) {
                    mismatches++;
                }
            }
        }
    }
    float pct = 100.0f * mismatches / (width * height * 3);
    EXPECT_LT(pct, 0.1f) << "Radius 0 should be near-identity";
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    LensBlurTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719)
    )
);
