#include "test_common.h"
#include "halide_ops.h"

// Test fused NV21->resize->RGB pipelines against OpenCV chained steps.

struct FusedResizeParams {
    int src_w, src_h, dst_w, dst_h;
};

class Nv21ResizeRgbOptimizedTest : public ::testing::TestWithParam<FusedResizeParams> {};

TEST_P(Nv21ResizeRgbOptimizedTest, BilinearFused_PSNR) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    // Halide fused: NV21 -> bilinear resize -> RGB
    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(p.dst_w, p.dst_h, 3);
    int ret = halide_ops::nv21_resize_rgb_bilinear_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, halide_out);
    ASSERT_EQ(ret, 0);

    // OpenCV reference: NV21 -> RGB -> resize
    std::vector<uint8_t> nv21_contiguous(y_data.size() + uv_data.size());
    std::copy(y_data.begin(), y_data.end(), nv21_contiguous.begin());
    std::copy(uv_data.begin(), uv_data.end(), nv21_contiguous.begin() + y_data.size());
    cv::Mat nv21_mat(p.src_h + p.src_h / 2, p.src_w, CV_8UC1, nv21_contiguous.data());
    cv::Mat rgb_cv;
    cv::cvtColor(nv21_mat, rgb_cv, cv::COLOR_YUV2RGB_NV21);
    cv::Mat resized_cv;
    cv::resize(rgb_cv, resized_cv, cv::Size(p.dst_w, p.dst_h), 0, 0, cv::INTER_LINEAR);

    // PSNR
    double mse = 0;
    int total = p.dst_w * p.dst_h * 3;
    for (int y = 0; y < p.dst_h; y++) {
        for (int x = 0; x < p.dst_w; x++) {
            for (int c = 0; c < 3; c++) {
                double diff = (double)halide_out(x, y, c) - (double)resized_cv.at<cv::Vec3b>(y, x)[c];
                mse += diff * diff;
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    printf("  Fused Bilinear PSNR: %.2f dB (%dx%d -> %dx%d)\n",
           psnr, p.src_w, p.src_h, p.dst_w, p.dst_h);
    // Note: PSNR may be lower than 50 dB because the fused pipeline resizes
    // in NV21 domain (Y at full res, UV at half res) while OpenCV resizes
    // the full RGB image. The chroma handling differs, especially at high
    // downscale ratios (e.g., 1920→640 = 3x). We expect > 28 dB.
    EXPECT_GT(psnr, 28.0) << "Fused bilinear PSNR too low";
}

TEST_P(Nv21ResizeRgbOptimizedTest, AreaFused_NoCrash) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(p.dst_w, p.dst_h, 3);
    int ret = halide_ops::nv21_resize_rgb_area_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, halide_out);
    ASSERT_EQ(ret, 0);
}

TEST_P(Nv21ResizeRgbOptimizedTest, BicubicFused_NoCrash) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    Halide::Runtime::Buffer<uint8_t> halide_out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(p.dst_w, p.dst_h, 3);
    int ret = halide_ops::nv21_resize_rgb_bicubic_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, halide_out);
    ASSERT_EQ(ret, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ResizeRgbOptimizedTest,
    ::testing::Values(
        FusedResizeParams{640, 480, 320, 240},
        FusedResizeParams{1280, 720, 640, 360},
        FusedResizeParams{1920, 1080, 640, 480},
        FusedResizeParams{1920, 1080, 960, 540}
    )
);
