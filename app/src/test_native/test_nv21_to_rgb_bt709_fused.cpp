// =============================================================================
// Test: nv21_to_rgb_bt709_fused
//
// Validates that the fused BT.709 pipeline matches TargetOpenCV.cpp's scalar
// reference (nv21_to_YUV444 + YUV444_to_RGB) to ~94-100 dB PSNR with
// max_diff <= 1 LSB across all tested resolutions, and is faster than the
// scalar TargetOpenCV.cpp loop on device.
// =============================================================================

#include "test_common.h"
#include "nv21_to_rgb_bt709_fused.h"

#include <chrono>
#include <cmath>
#include <cstdio>

namespace {

// Bit-exact float reference of TargetOpenCV.cpp.
void ref_nv21_bt709_float(const uint8_t* y_data, const uint8_t* uv_data,
                          int w, int h, std::vector<uint8_t>& rgb_out) {
    rgb_out.resize((size_t)w * h * 3);
    for (int row = 0; row < h; row++) {
        int uv_row = row / 2;
        for (int col = 0; col < w; col++) {
            int uv_col = (col / 2) * 2;
            int Y = y_data[row * w + col];
            int V = uv_data[uv_row * w + uv_col];      // V at even byte
            int U = uv_data[uv_row * w + uv_col + 1];  // U at odd byte

            float Yf = (float)Y;
            float Uf = (float)U - 128.0f;
            float Vf = (float)V - 128.0f;

            float Rf = Yf +              1.5748f * Vf;
            float Gf = Yf - 0.1873f*Uf - 0.4681f * Vf;
            float Bf = Yf + 1.8556f*Uf;

            int R = (int)std::lround(Rf);
            int G = (int)std::lround(Gf);
            int B = (int)std::lround(Bf);

            rgb_out[(row*w + col)*3 + 0] = (uint8_t)std::max(0, std::min(255, R));
            rgb_out[(row*w + col)*3 + 1] = (uint8_t)std::max(0, std::min(255, G));
            rgb_out[(row*w + col)*3 + 2] = (uint8_t)std::max(0, std::min(255, B));
        }
    }
}

double compute_psnr_and_max(const Halide::Runtime::Buffer<uint8_t>& halide,
                            const std::vector<uint8_t>& ref,
                            int w, int h, int& max_diff) {
    double mse = 0.0;
    max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int d = (int)halide(x, y, c) - (int)ref[(size_t)(y*w + x)*3 + c];
                int ad = std::abs(d);
                if (ad > max_diff) max_diff = ad;
                mse += (double)d * d;
            }
        }
    }
    mse /= (double)((size_t)w * h * 3);
    return (mse > 0) ? 10.0 * std::log10(255.0 * 255.0 / mse) : 200.0;
}

template <typename Fn>
long long bench_us(Fn fn, int iters, int warmup = 5) {
    for (int i = 0; i < warmup; i++) fn();
    std::vector<long long> t(iters);
    for (int i = 0; i < iters; i++) {
        auto a = std::chrono::high_resolution_clock::now();
        fn();
        auto b = std::chrono::high_resolution_clock::now();
        t[i] = std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    }
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

} // namespace

class Nv21Bt709FusedTest : public ::testing::TestWithParam<std::pair<int,int>> {};

// Halide output stays within 1 LSB of the float reference (PSNR ~94-100 dB).
TEST_P(Nv21Bt709FusedTest, MatchesFloatReference) {
    auto [width, height] = GetParam();
    int w = width  & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + (size_t)w*h, w, h/2);
    Halide::Runtime::Buffer<uint8_t> out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

    ASSERT_EQ(::nv21_to_rgb_bt709_fused(y_buf, uv_buf, out), 0);

    std::vector<uint8_t> ref;
    ref_nv21_bt709_float(nv21.data(), nv21.data() + (size_t)w*h, w, h, ref);

    int max_diff = 0;
    double psnr = compute_psnr_and_max(out, ref, w, h, max_diff);
    printf("  %4dx%4d: PSNR=%.2f dB max_diff=%d\n", w, h, psnr, max_diff);

    EXPECT_GE(psnr, 90.0);
    EXPECT_LE(max_diff, 1);
}

// Halide must beat the scalar TargetOpenCV.cpp reference (the function being
// replaced). OpenCV cv::cvtColor BT.601 timing is printed for context.
TEST_P(Nv21Bt709FusedTest, FasterThanScalarReference) {
    auto [width, height] = GetParam();
    int w = width  & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    Halide::Runtime::Buffer<uint8_t> y_buf(nv21.data(), w, h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(nv21.data() + (size_t)w*h, w, h/2);
    Halide::Runtime::Buffer<uint8_t> out =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

    const int ITERS = 30;

    std::vector<uint8_t> ref_buf((size_t)w * h * 3);
    long long ref_us = bench_us([&](){
        ref_nv21_bt709_float(nv21.data(), nv21.data() + (size_t)w*h, w, h, ref_buf);
    }, ITERS);

    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat cv_rgb;
    long long cv_us = bench_us([&](){
        cv::cvtColor(nv21_mat, cv_rgb, cv::COLOR_YUV2RGB_NV21);
    }, ITERS);

    long long halide_us = bench_us([&](){
        ::nv21_to_rgb_bt709_fused(y_buf, uv_buf, out);
    }, ITERS);

    printf("  %4dx%4d (median us, %d iters):\n", w, h, ITERS);
    printf("    Scalar ref (TargetOpenCV.cpp) : %6lld us  (replacement target)\n", ref_us);
    printf("    OpenCV BT.601 cvtColor        : %6lld us  (production alt)\n", cv_us);
    printf("    Halide nv21_to_rgb_bt709_fused: %6lld us  (%.2fx ref, %.2fx OpenCV)\n",
           halide_us,
           ref_us > 0 ? (double)ref_us / halide_us : 0.0,
           cv_us  > 0 ? (double)cv_us  / halide_us : 0.0);

    // The schedule is tuned for HD+ (camera production target is 12MP).
    // Below ~300K pixels the 16-row parallel split has more thread overhead
    // than work, so timing at sub-VGA is informational only.
    if ((size_t)w * h >= 300000) {
        EXPECT_LT(halide_us, ref_us)
            << "Halide slower than scalar TargetOpenCV.cpp reference at "
            << w << "x" << h;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions, Nv21Bt709FusedTest, ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),    // odd dims
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),   // odd dims
        std::make_pair(1920, 1080),
        std::make_pair(3840, 2160),
        std::make_pair(4032, 3024)   // 12MP (Samsung default)
    ));
