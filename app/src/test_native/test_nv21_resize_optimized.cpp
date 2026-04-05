#include "test_common.h"
#include "halide_ops.h"

// Test NV21-domain resize by converting results to RGB and comparing with
// OpenCV's NV21->RGB->resize pipeline.

static double compute_psnr_buffers(
    const Halide::Runtime::Buffer<uint8_t>& a,
    const Halide::Runtime::Buffer<uint8_t>& b) {
    EXPECT_EQ(a.width(), b.width());
    EXPECT_EQ(a.height(), b.height());
    EXPECT_EQ(a.channels(), b.channels());
    double mse = 0;
    int total = a.width() * a.height() * a.channels();
    for (int y = 0; y < a.height(); y++) {
        for (int x = 0; x < a.width(); x++) {
            for (int c = 0; c < a.channels(); c++) {
                double diff = (double)a(x, y, c) - (double)b(x, y, c);
                mse += diff * diff;
            }
        }
    }
    mse /= total;
    if (mse == 0) return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

struct NV21ResizeParams {
    int src_w, src_h, dst_w, dst_h;
};

class Nv21ResizeOptimizedTest : public ::testing::TestWithParam<NV21ResizeParams> {};

TEST_P(Nv21ResizeOptimizedTest, BilinearResize_PSNR) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    // Halide: NV21-domain resize
    Halide::Runtime::Buffer<uint8_t> y_out(p.dst_w, p.dst_h);
    Halide::Runtime::Buffer<uint8_t> uv_out(p.dst_w, p.dst_h / 2);
    int ret = halide_ops::nv21_resize_bilinear_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, y_out, uv_out);
    ASSERT_EQ(ret, 0);

    // Convert Halide NV21 result to RGB (nv21_to_rgb uses planar output)
    Halide::Runtime::Buffer<uint8_t> halide_rgb(p.dst_w, p.dst_h, 3);
    halide_ops::nv21_to_rgb(y_out, uv_out, halide_rgb);

    // OpenCV reference: NV21 -> RGB -> resize
    std::vector<uint8_t> nv21_contiguous(y_data.size() + uv_data.size());
    std::copy(y_data.begin(), y_data.end(), nv21_contiguous.begin());
    std::copy(uv_data.begin(), uv_data.end(), nv21_contiguous.begin() + y_data.size());
    cv::Mat nv21_mat(p.src_h + p.src_h / 2, p.src_w, CV_8UC1, nv21_contiguous.data());
    cv::Mat rgb_cv;
    cv::cvtColor(nv21_mat, rgb_cv, cv::COLOR_YUV2RGB_NV21);
    cv::Mat resized_cv;
    cv::resize(rgb_cv, resized_cv, cv::Size(p.dst_w, p.dst_h), 0, 0, cv::INTER_LINEAR);

    // Compare
    // NV21-domain resize vs OpenCV RGB-domain resize has inherent chroma differences.
    // Large downscale ratios (e.g., 1920→640) amplify the UV interpolation gap.
    compare_buffers_rgb(halide_rgb, resized_cv, 50, /*opencv_is_bgr=*/false);

    // PSNR check
    double mse = 0;
    int total = p.dst_w * p.dst_h * 3;
    for (int y = 0; y < p.dst_h; y++) {
        for (int x = 0; x < p.dst_w; x++) {
            for (int c = 0; c < 3; c++) {
                double diff = (double)halide_rgb(x, y, c) - (double)resized_cv.at<cv::Vec3b>(y, x)[c];
                mse += diff * diff;
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    printf("  NV21 Bilinear Resize PSNR: %.2f dB (%dx%d -> %dx%d)\n",
           psnr, p.src_w, p.src_h, p.dst_w, p.dst_h);
    // NV21-domain resize differs from RGB-domain resize because chroma is at
    // half resolution. At high downscale ratios the UV interpolation introduces
    // more error vs OpenCV's RGB-domain resize. 28 dB is reasonable.
    EXPECT_GT(psnr, 28.0) << "PSNR too low for NV21 bilinear resize";
}

TEST_P(Nv21ResizeOptimizedTest, AreaResize_NoCrash) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    Halide::Runtime::Buffer<uint8_t> y_out(p.dst_w, p.dst_h);
    Halide::Runtime::Buffer<uint8_t> uv_out(p.dst_w, p.dst_h / 2);
    int ret = halide_ops::nv21_resize_area_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, y_out, uv_out);
    ASSERT_EQ(ret, 0);
}

TEST_P(Nv21ResizeOptimizedTest, BicubicResize_NoCrash) {
    auto p = GetParam();

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(p.src_w, p.src_h, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), p.src_w, p.src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), p.src_w, p.src_h / 2);

    Halide::Runtime::Buffer<uint8_t> y_out(p.dst_w, p.dst_h);
    Halide::Runtime::Buffer<uint8_t> uv_out(p.dst_w, p.dst_h / 2);
    int ret = halide_ops::nv21_resize_bicubic_optimized(
        y_buf, uv_buf, p.dst_w, p.dst_h, y_out, uv_out);
    ASSERT_EQ(ret, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ResizeOptimizedTest,
    ::testing::Values(
        NV21ResizeParams{640, 480, 320, 240},
        NV21ResizeParams{1280, 720, 640, 360},
        NV21ResizeParams{1920, 1080, 640, 480},
        NV21ResizeParams{640, 480, 320, 240}
    )
);
