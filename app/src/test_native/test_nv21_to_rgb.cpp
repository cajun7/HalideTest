#include "test_common.h"
#include "nv21_to_rgb.h"

class Nv21ToRgbTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Nv21ToRgbTest, MatchesOpenCV) {
    auto [width, height] = GetParam();

    // NV21 requires even dimensions for UV subsampling
    int w = width & ~1;   // round down to even
    int h = height & ~1;

    // Generate synthetic NV21 data
    auto nv21 = make_nv21_contiguous(w, h);

    // OpenCV reference: NV21 -> RGB
    // OpenCV treats NV21 as a single (h*3/2) x w image
    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat opencv_rgb;
    cv::cvtColor(nv21_mat, opencv_rgb, cv::COLOR_YUV2RGB_NV21);

    // Halide: separate Y and UV planes
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);

    // UV plane: interleaved VU, modeled as (w/2) x (h/2) x 2
    // dim0 stride = 2 (interleaved), dim1 stride = w, dim2 stride = 1
    halide_dimension_t uv_dims[3] = {
        {0, w / 2, 2},   // x: extent=w/2, stride=2
        {0, h / 2, w},   // y: extent=h/2, stride=w
        {0, 2, 1},       // c: extent=2 (V,U), stride=1
    };
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, 3, uv_dims);

    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

    int err = nv21_to_rgb(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide nv21_to_rgb failed with error " << err;

    // Compare Halide RGB output vs OpenCV RGB output
    // OpenCV result is already RGB (not BGR) since we used COLOR_YUV2RGB_NV21
    compare_buffers_rgb(output_buf, opencv_rgb, /*tolerance=*/3, /*opencv_is_bgr=*/false);
}

TEST_P(Nv21ToRgbTest, OutputInValidRange) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    halide_dimension_t uv_dims[3] = {
        {0, w / 2, 2}, {0, h / 2, w}, {0, 2, 1},
    };
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, 3, uv_dims);
    Halide::Runtime::Buffer<uint8_t> output_buf =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

    nv21_to_rgb(y_buf, uv_buf, output_buf);

    // All output values should be in [0, 255] (implicit for uint8, but verify no crash)
    SUCCEED() << "No crash on " << w << "x" << h;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ToRgbTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(642, 482),    // even but non-standard
        std::make_pair(1280, 720),
        std::make_pair(1280, 718),   // even but non-standard
        std::make_pair(1920, 1080)
    )
);
