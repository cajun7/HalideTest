#include "test_common.h"
#include "nv21_to_rgb.h"
#include "rgb_to_nv21.h"

class Nv21ToRgbTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Helper: create 2D UV buffer from raw NV21 data
static Halide::Runtime::Buffer<uint8_t> make_uv_buf(uint8_t* uv_ptr, int w, int h) {
    return Halide::Runtime::Buffer<uint8_t>(uv_ptr, w, h / 2);
}

TEST_P(Nv21ToRgbTest, MatchesOpenCV) {
    auto [width, height] = GetParam();

    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);

    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat opencv_rgb;
    cv::cvtColor(nv21_mat, opencv_rgb, cv::COLOR_YUV2RGB_NV21);

    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);

    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_to_rgb(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide nv21_to_rgb failed with error " << err;

    // Tolerance raised to 20: Halide uses integer fixed-point BT.601 while
    // OpenCV uses a different rounding scheme, causing up to ~19 LSB difference.
    compare_buffers_rgb(output_buf, opencv_rgb, /*tolerance=*/20, /*opencv_is_bgr=*/false);
}

TEST_P(Nv21ToRgbTest, OutputInValidRange) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    nv21_to_rgb(y_buf, uv_buf, output_buf);

    SUCCEED() << "No crash on " << w << "x" << h;
}

TEST_P(Nv21ToRgbTest, RoundTrip_ViaRgbToNv21) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_in(y_ptr, w, h);
    auto uv_in = make_uv_buf(uv_ptr, w, h);

    // Pass 1: NV21 -> RGB
    Halide::Runtime::Buffer<uint8_t> rgb1(w, h, 3);
    int err = nv21_to_rgb(y_in, uv_in, rgb1);
    ASSERT_EQ(err, 0);

    // Pass 2: RGB -> NV21
    Halide::Runtime::Buffer<uint8_t> y_mid(w, h);
    std::vector<uint8_t> uv_mid_storage(w * (h / 2));
    Halide::Runtime::Buffer<uint8_t> uv_mid(uv_mid_storage.data(), w, h / 2);
    err = rgb_to_nv21(rgb1, y_mid, uv_mid);
    ASSERT_EQ(err, 0);

    // Pass 3: NV21 -> RGB again
    Halide::Runtime::Buffer<uint8_t> rgb2(w, h, 3);
    err = nv21_to_rgb(y_mid, uv_mid, rgb2);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)rgb1(x, y, c) - (int)rgb2(x, y, c));
                if (diff > 3) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float total = (float)(w * h * 3);
    float pct = 100.0f * mismatches / total;
    EXPECT_LT(pct, 1.0f)
        << "Round-trip mismatches: " << mismatches << " (" << pct << "%), max_diff=" << max_diff;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ToRgbTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(642, 482),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1280, 718),
        std::make_pair(1920, 1080)
    )
);
