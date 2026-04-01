#include "test_common.h"
#include "rotate_fixed.h"
#include "rotate_arbitrary.h"

class RotateFixedTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Test 90-degree CW rotation against OpenCV
TEST_P(RotateFixedTest, Rotate90_MatchesOpenCV) {
    auto [width, height] = GetParam();

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCV reference: 90 CW
    cv::Mat opencv_result;
    cv::rotate(bgr, opencv_result, cv::ROTATE_90_CLOCKWISE);

    // Halide: output dimensions are swapped (height x width)
    auto input_buf = mat_to_halide_interleaved(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(height, width, 3);

    int err = rotate_fixed(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide rotate_fixed (90) failed";

    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/0, /*opencv_is_bgr=*/true);
}

// Test that 4x 90-degree rotations return to original
TEST_P(RotateFixedTest, FourRotations_IsIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_bgr(width, height);
    auto buf0 = mat_to_halide_interleaved(rgb);

    // We only have rotation_code=1 (90 CW) compiled as default.
    // Apply it 4 times: w x h -> h x w -> w x h -> h x w -> w x h
    auto buf1 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(height, width, 3);
    auto buf2 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);
    auto buf3 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(height, width, 3);
    auto buf4 = Halide::Runtime::Buffer<uint8_t>::make_interleaved(width, height, 3);

    ASSERT_EQ(rotate_fixed(buf0, buf1), 0);
    ASSERT_EQ(rotate_fixed(buf1, buf2), 0);
    ASSERT_EQ(rotate_fixed(buf2, buf3), 0);
    ASSERT_EQ(rotate_fixed(buf3, buf4), 0);

    // buf4 should match buf0 exactly
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                ASSERT_EQ(buf0(x, y, c), buf4(x, y, c))
                    << "4x rotation not identity at (" << x << "," << y << "," << c << ")";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RotateFixedTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719)
    )
);

// ---------------------------------------------------------------------------
// Arbitrary rotation tests
// ---------------------------------------------------------------------------

class RotateArbitraryTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Test 0-degree rotation (identity)
TEST_P(RotateArbitraryTest, ZeroAngle_IsNearIdentity) {
    auto [width, height] = GetParam();

    cv::Mat rgb = make_test_image_bgr(width, height);
    auto input_buf = mat_to_halide_planar(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf(width, height, 3);

    int err = rotate_arbitrary(input_buf, 0.0f, output_buf);
    ASSERT_EQ(err, 0);

    // 0-degree rotation with bilinear interpolation should be near-identity
    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)input_buf(x, y, c) - (int)output_buf(x, y, c));
                if (diff > 1) mismatches++;
            }
        }
    }
    float pct = 100.0f * mismatches / (width * height * 3);
    EXPECT_LT(pct, 1.0f) << "Zero-angle rotation should be near-identity";
}

// Test against OpenCV warpAffine for 45-degree rotation
TEST_P(RotateArbitraryTest, Rotate45_MatchesOpenCV) {
    auto [width, height] = GetParam();
    float angle_deg = 45.0f;
    float angle_rad = angle_deg * (float)M_PI / 180.0f;

    cv::Mat bgr = make_test_image_bgr(width, height);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCV reference: rotation around center
    cv::Point2f center(width / 2.0f, height / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_deg, 1.0);
    cv::Mat opencv_result;
    cv::warpAffine(bgr, opencv_result, rot_mat, cv::Size(width, height),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Halide (planar buffers — constant_exterior has issues with interleaved strides in v21)
    auto input_buf = mat_to_halide_planar(rgb);
    Halide::Runtime::Buffer<uint8_t> output_buf(width, height, 3);

    // Negate angle: Halide's "positive=CCW" in math coords is CW in image coords,
    // while OpenCV's getRotationMatrix2D uses positive=CCW in image coords.
    int err = rotate_arbitrary(input_buf, -angle_rad, output_buf);
    ASSERT_EQ(err, 0);

    // Higher tolerance due to interpolation differences at edges
    compare_buffers_rgb(output_buf, opencv_result, /*tolerance=*/5, /*opencv_is_bgr=*/true);
}

// No crash on various angles
TEST_P(RotateArbitraryTest, NoCrash_VariousAngles) {
    auto [width, height] = GetParam();

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
    auto input_buf = mat_to_halide_planar(rgb);

    float angles[] = {0.0f, 0.1f, 0.5f, 1.0f, 1.5708f, 3.14159f, 4.71239f, 6.28318f};
    for (float angle : angles) {
        Halide::Runtime::Buffer<uint8_t> output_buf(width, height, 3);

        int err = rotate_arbitrary(input_buf, angle, output_buf);
        ASSERT_EQ(err, 0) << "Crashed at angle=" << angle
                          << " on " << width << "x" << height;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RotateArbitraryTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719)
    )
);
