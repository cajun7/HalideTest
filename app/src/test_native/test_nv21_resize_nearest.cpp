// =============================================================================
// Equivalence tests for nv21_resize_nearest_optimized.
//
// Two oracles:
//   1. Scalar reference implementing the same floor()-based mapping as the
//      generator. Halide must match byte-for-byte (tolerance 0).
//   2. OpenCV's cv::resize(_, _, INTER_NEAREST) after Halide NV21 -> RGB.
//      Allowed small mismatch rate because OpenCV resizes RGB directly while
//      we resize in NV21 domain (different chroma rounding).
// =============================================================================

#include "test_common.h"
#include "halide_ops.h"
#include "nv21_resize_nearest_optimized.h"

#include <cmath>
#include <cstring>

namespace {

// Scalar oracle: matches OpenCV's INTER_NEAREST convention exactly.
// sx = floor(dst_x * src_w / dst_w), sy = floor(dst_y * src_h / dst_h)
void scalar_nv21_resize_nearest(
    const uint8_t* y_src, int src_w, int src_h,
    const uint8_t* uv_src,
    int dst_w, int dst_h,
    uint8_t* y_dst, uint8_t* uv_dst) {
    for (int dy = 0; dy < dst_h; ++dy) {
        int sy = (int)std::floor((double)dy * src_h / dst_h);
        if (sy >= src_h) sy = src_h - 1;
        for (int dx = 0; dx < dst_w; ++dx) {
            int sx = (int)std::floor((double)dx * src_w / dst_w);
            if (sx >= src_w) sx = src_w - 1;
            y_dst[dy * dst_w + dx] = y_src[sy * src_w + sx];
        }
    }
    const int uv_src_w = src_w / 2, uv_src_h = src_h / 2;
    const int uv_dst_w = dst_w / 2, uv_dst_h = dst_h / 2;
    for (int dy = 0; dy < uv_dst_h; ++dy) {
        int sy = (int)std::floor((double)dy * uv_src_h / uv_dst_h);
        if (sy >= uv_src_h) sy = uv_src_h - 1;
        for (int dx_px = 0; dx_px < uv_dst_w; ++dx_px) {
            int sx_px = (int)std::floor((double)dx_px * uv_src_w / uv_dst_w);
            if (sx_px >= uv_src_w) sx_px = uv_src_w - 1;
            // V at even byte, U at odd byte
            uv_dst[dy * dst_w + 2 * dx_px    ] = uv_src[sy * src_w + 2 * sx_px];
            uv_dst[dy * dst_w + 2 * dx_px + 1] = uv_src[sy * src_w + 2 * sx_px + 1];
        }
    }
}

struct NV21ResizeNearestParams {
    int src_w, src_h, dst_w, dst_h;
};

}  // namespace

class Nv21ResizeNearestTest
    : public ::testing::TestWithParam<NV21ResizeNearestParams> {};

TEST_P(Nv21ResizeNearestTest, MatchesScalarOracle) {
    auto p = GetParam();
    // Force even dims — NV21 chroma subsampling requires it, and OpenCV
    // INTER_NEAREST on RGB would differ otherwise.
    int sw = p.src_w & ~1, sh = p.src_h & ~1;
    int dw = p.dst_w & ~1, dh = p.dst_h & ~1;

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);

    Halide::Runtime::Buffer<uint8_t> y_out(dw, dh);
    Halide::Runtime::Buffer<uint8_t> uv_out(dw, dh / 2);
    int err = nv21_resize_nearest_optimized(y_buf, uv_buf, dw, dh, y_out, uv_out);
    ASSERT_EQ(err, 0);

    std::vector<uint8_t> y_ref(dw * dh);
    std::vector<uint8_t> uv_ref(dw * (dh / 2));
    scalar_nv21_resize_nearest(y_data.data(), sw, sh, uv_data.data(),
                               dw, dh, y_ref.data(), uv_ref.data());

    // Y plane bit-exact
    int y_mismatches = 0, first_bad = -1;
    for (int i = 0; i < dw * dh; ++i) {
        if (y_out.data()[i] != y_ref[i]) {
            if (first_bad < 0) first_bad = i;
            ++y_mismatches;
        }
    }
    if (y_mismatches > 0) {
        int px = first_bad % dw, py = first_bad / dw;
        printf("  Y first mismatch (%d,%d): halide=%d scalar=%d (%d total)\n",
               px, py, (int)y_out.data()[first_bad], (int)y_ref[first_bad],
               y_mismatches);
    }
    EXPECT_EQ(y_mismatches, 0) << "Y plane must be bit-exact vs scalar oracle";

    // UV plane bit-exact
    int uv_mismatches = 0; first_bad = -1;
    for (int i = 0; i < dw * (dh / 2); ++i) {
        if (uv_out.data()[i] != uv_ref[i]) {
            if (first_bad < 0) first_bad = i;
            ++uv_mismatches;
        }
    }
    if (uv_mismatches > 0) {
        printf("  UV first mismatch idx=%d: halide=%d scalar=%d (%d total)\n",
               first_bad, (int)uv_out.data()[first_bad], (int)uv_ref[first_bad],
               uv_mismatches);
    }
    EXPECT_EQ(uv_mismatches, 0) << "UV plane must be bit-exact vs scalar oracle";
}

TEST_P(Nv21ResizeNearestTest, MatchesOpenCVRgbResize_PSNR) {
    auto p = GetParam();
    int sw = p.src_w & ~1, sh = p.src_h & ~1;
    int dw = p.dst_w & ~1, dh = p.dst_h & ~1;

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);

    Halide::Runtime::Buffer<uint8_t> y_out(dw, dh);
    Halide::Runtime::Buffer<uint8_t> uv_out(dw, dh / 2);
    int err = nv21_resize_nearest_optimized(y_buf, uv_buf, dw, dh, y_out, uv_out);
    ASSERT_EQ(err, 0);

    // Halide side: convert the resized NV21 to RGB
    Halide::Runtime::Buffer<uint8_t> halide_rgb(dw, dh, 3);
    err = halide_ops::nv21_to_rgb(y_out, uv_out, halide_rgb);
    ASSERT_EQ(err, 0);

    // OpenCV side: NV21 -> RGB -> resize INTER_NEAREST
    std::vector<uint8_t> nv21_contig(y_data.size() + uv_data.size());
    std::copy(y_data.begin(), y_data.end(), nv21_contig.begin());
    std::copy(uv_data.begin(), uv_data.end(), nv21_contig.begin() + y_data.size());
    cv::Mat nv21_mat(sh + sh / 2, sw, CV_8UC1, nv21_contig.data());
    cv::Mat cv_rgb;
    cv::cvtColor(nv21_mat, cv_rgb, cv::COLOR_YUV2RGB_NV21);
    cv::Mat cv_resized;
    cv::resize(cv_rgb, cv_resized, cv::Size(dw, dh), 0, 0, cv::INTER_NEAREST);

    // PSNR — nearest is order-sensitive (resize-then-convert vs convert-then-resize
    // produce slightly different output at chroma boundaries), but for synthetic
    // data the overall PSNR should stay high.
    double mse = 0;
    int total = dw * dh * 3;
    for (int yy = 0; yy < dh; yy++) {
        for (int xx = 0; xx < dw; xx++) {
            for (int c = 0; c < 3; c++) {
                double diff = (double)halide_rgb(xx, yy, c) -
                              (double)cv_resized.at<cv::Vec3b>(yy, xx)[c];
                mse += diff * diff;
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    printf("  NV21 Nearest Resize PSNR vs OpenCV: %.2f dB (%dx%d -> %dx%d)\n",
           psnr, sw, sh, dw, dh);
    EXPECT_GT(psnr, 24.0) << "PSNR too low for nearest (order-of-ops divergence)";
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ResizeNearestTest,
    ::testing::Values(
        NV21ResizeNearestParams{640,  480,  320,  240},   // downscale 2x
        NV21ResizeNearestParams{1920, 1080, 640,  480},   // downscale ~3x
        NV21ResizeNearestParams{1280, 720,  1280, 720},   // same-size
        NV21ResizeNearestParams{640,  480,  1280, 720},   // upscale
        NV21ResizeNearestParams{1920, 1080, 1280, 720},   // downscale 1.5x
        NV21ResizeNearestParams{640,  480,  641,  481},   // odd dst
        NV21ResizeNearestParams{642,  482,  320,  240}    // odd src
    )
);
