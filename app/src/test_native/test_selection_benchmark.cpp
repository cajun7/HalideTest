#include "test_common.h"
#include "halide_ops.h"
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdio>

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct BenchResult {
    double median_us;
    double mean_us;
};

static BenchResult bench(std::function<void()> fn, int iters = 20) {
    fn(); // warmup
    std::vector<double> times(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    double median = times[iters / 2];
    double sum = 0;
    for (auto t : times) sum += t;
    return {median, sum / iters};
}

struct PsnrResult {
    double psnr;
    int max_diff;
};

static PsnrResult compute_psnr_rgb(
    const Halide::Runtime::Buffer<uint8_t>& a,
    const cv::Mat& b, bool bgr = false)
{
    double mse = 0;
    int max_diff = 0;
    int total = a.width() * a.height() * 3;
    for (int y = 0; y < a.height(); y++) {
        for (int x = 0; x < a.width(); x++) {
            for (int c = 0; c < 3; c++) {
                int cv_c = bgr ? (2 - c) : c;
                double diff = (double)a(x, y, c) - (double)b.at<cv::Vec3b>(y, x)[cv_c];
                mse += diff * diff;
                max_diff = std::max(max_diff, (int)std::abs(diff));
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    return {psnr, max_diff};
}

// Compare two Halide buffers directly
static PsnrResult compute_psnr_bufs(
    const Halide::Runtime::Buffer<uint8_t>& a,
    const Halide::Runtime::Buffer<uint8_t>& b)
{
    double mse = 0;
    int max_diff = 0;
    int total = a.width() * a.height() * a.channels();
    for (int y = 0; y < a.height(); y++) {
        for (int x = 0; x < a.width(); x++) {
            for (int c = 0; c < a.channels(); c++) {
                double diff = (double)a(x, y, c) - (double)b(x, y, c);
                mse += diff * diff;
                max_diff = std::max(max_diff, (int)std::abs(diff));
            }
        }
    }
    mse /= total;
    double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    return {psnr, max_diff};
}

// ===========================================================================
// BENCHMARK TESTS
// ===========================================================================

TEST(SelectionBenchmark, RgbBgr) {
    int w = 1920, h = 1080;
    printf("\n=== RGB <-> BGR  (%dx%d, 20 iters) ===\n", w, h);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> out_base =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
    Halide::Runtime::Buffer<uint8_t> out_opt =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

    // Timing
    auto t_base = bench([&]() { halide_ops::rgb_bgr(halide_in, out_base); });
    auto t_opt  = bench([&]() { halide_ops::rgb_bgr_optimized(halide_in, out_opt); });
    cv::Mat cv_out;
    auto t_cv   = bench([&]() { cv::cvtColor(rgb_cv, cv_out, cv::COLOR_RGB2BGR); });

    // PSNR: baseline vs optimized (should be identical = 100 dB)
    auto p_base_vs_opt = compute_psnr_bufs(out_base, out_opt);
    // PSNR vs OpenCV
    auto p_base_vs_cv = compute_psnr_rgb(out_base, cv_out, /*bgr=*/false);
    auto p_opt_vs_cv  = compute_psnr_rgb(out_opt, cv_out, /*bgr=*/false);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized", t_opt.median_us, t_opt.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "OpenCV", t_cv.median_us, t_cv.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR base vs opt:   %.1f dB (max_diff=%d)\n", p_base_vs_opt.psnr, p_base_vs_opt.max_diff);
    printf("  PSNR base vs CV:    %.1f dB (max_diff=%d)\n", p_base_vs_cv.psnr, p_base_vs_cv.max_diff);
    printf("  PSNR opt  vs CV:    %.1f dB (max_diff=%d)\n", p_opt_vs_cv.psnr, p_opt_vs_cv.max_diff);
}

TEST(SelectionBenchmark, Nv21ToRgb) {
    int w = 1920, h = 1080;
    printf("\n=== NV21 -> RGB  (%dx%d, 20 iters) ===\n", w, h);

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(w, h, y_data, uv_data);
    Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), w, h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), w, h / 2);

    // NV21->RGB generators use default (planar) output layout, NOT interleaved
    Halide::Runtime::Buffer<uint8_t> out_base(w, h, 3);
    Halide::Runtime::Buffer<uint8_t> out_opt(w, h, 3);

    // Timing
    auto t_base = bench([&]() { halide_ops::nv21_to_rgb(y_buf, uv_buf, out_base); });
    auto t_opt  = bench([&]() { halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, out_opt); });

    // OpenCV reference
    std::vector<uint8_t> nv21(y_data.size() + uv_data.size());
    std::copy(y_data.begin(), y_data.end(), nv21.begin());
    std::copy(uv_data.begin(), uv_data.end(), nv21.begin() + y_data.size());
    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat cv_out;
    auto t_cv = bench([&]() { cv::cvtColor(nv21_mat, cv_out, cv::COLOR_YUV2RGB_NV21); });

    auto p_base_vs_opt = compute_psnr_bufs(out_base, out_opt);
    auto p_base_vs_cv = compute_psnr_rgb(out_base, cv_out, /*bgr=*/false);
    auto p_opt_vs_cv  = compute_psnr_rgb(out_opt, cv_out, /*bgr=*/false);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized", t_opt.median_us, t_opt.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "OpenCV", t_cv.median_us, t_cv.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR base vs opt:   %.1f dB (max_diff=%d)\n", p_base_vs_opt.psnr, p_base_vs_opt.max_diff);
    printf("  PSNR base vs CV:    %.1f dB (max_diff=%d)\n", p_base_vs_cv.psnr, p_base_vs_cv.max_diff);
    printf("  PSNR opt  vs CV:    %.1f dB (max_diff=%d)\n", p_opt_vs_cv.psnr, p_opt_vs_cv.max_diff);
}

TEST(SelectionBenchmark, RgbToNv21) {
    int w = 1920, h = 1080;
    printf("\n=== RGB -> NV21  (%dx%d, 20 iters) ===\n", w, h);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    // rgb_to_nv21 generators expect planar input (no interleaved stride constraint)
    auto halide_in = mat_to_halide_planar(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> y_base(w, h), uv_base(w, h / 2);
    Halide::Runtime::Buffer<uint8_t> y_opt(w, h), uv_opt(w, h / 2);

    // Timing
    auto t_base = bench([&]() { halide_ops::rgb_to_nv21(halide_in, y_base, uv_base); });
    auto t_opt  = bench([&]() { halide_ops::rgb_to_nv21_optimized(halide_in, y_opt, uv_opt); });

    // OpenCV reference: RGB->YUV_NV21 (OpenCV doesn't have direct RGB->NV21, use round-trip)
    // Instead compare Y and UV planes directly between baseline and optimized
    PsnrResult p_y = compute_psnr_bufs(y_base, y_opt);

    // UV comparison
    double uv_mse = 0;
    int uv_max_diff = 0;
    int uv_total = w * (h / 2);
    for (int yy = 0; yy < h / 2; yy++) {
        for (int x = 0; x < w; x++) {
            double diff = (double)uv_base(x, yy) - (double)uv_opt(x, yy);
            uv_mse += diff * diff;
            uv_max_diff = std::max(uv_max_diff, (int)std::abs(diff));
        }
    }
    uv_mse /= uv_total;
    double uv_psnr = (uv_mse == 0) ? 100.0 : 10.0 * std::log10(255.0 * 255.0 / uv_mse);

    // Round-trip PSNR: RGB -> NV21 -> RGB (using optimized both ways)
    // NV21->RGB generator uses planar output layout
    Halide::Runtime::Buffer<uint8_t> roundtrip(w, h, 3);
    halide_ops::nv21_to_rgb(y_opt, uv_opt, roundtrip);
    auto p_roundtrip = compute_psnr_bufs(
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(
            (uint8_t*)rgb_cv.data, w, h, 3),
        roundtrip);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized", t_opt.median_us, t_opt.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR Y  base vs opt: %.1f dB (max_diff=%d)\n", p_y.psnr, p_y.max_diff);
    printf("  PSNR UV base vs opt: %.1f dB (max_diff=%d)\n", uv_psnr, uv_max_diff);
    printf("  Round-trip PSNR (RGB->NV21->RGB): %.1f dB (max_diff=%d)\n",
           p_roundtrip.psnr, p_roundtrip.max_diff);
}

TEST(SelectionBenchmark, ResizeBilinear) {
    int w = 1920, h = 1080;
    int tw = 960, th = 540;
    printf("\n=== Resize Bilinear  (%dx%d -> %dx%d, 20 iters) ===\n", w, h, tw, th);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> out_base =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    Halide::Runtime::Buffer<uint8_t> out_opt =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    // Timing
    auto t_base = bench([&]() { halide_ops::resize_bilinear_target(halide_in, tw, th, out_base); });
    auto t_opt  = bench([&]() { halide_ops::resize_bilinear_optimized(halide_in, tw, th, out_opt); });
    cv::Mat cv_out;
    auto t_cv   = bench([&]() { cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR); });

    auto p_base_vs_opt = compute_psnr_bufs(out_base, out_opt);
    auto p_base_vs_cv = compute_psnr_rgb(out_base, cv_out, /*bgr=*/false);
    auto p_opt_vs_cv  = compute_psnr_rgb(out_opt, cv_out, /*bgr=*/false);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline (float)", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized (Q11 int)", t_opt.median_us, t_opt.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "OpenCV INTER_LINEAR", t_cv.median_us, t_cv.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR base vs opt:   %.1f dB (max_diff=%d)\n", p_base_vs_opt.psnr, p_base_vs_opt.max_diff);
    printf("  PSNR base vs CV:    %.1f dB (max_diff=%d)\n", p_base_vs_cv.psnr, p_base_vs_cv.max_diff);
    printf("  PSNR opt  vs CV:    %.1f dB (max_diff=%d)\n", p_opt_vs_cv.psnr, p_opt_vs_cv.max_diff);
}

TEST(SelectionBenchmark, ResizeBicubic) {
    int w = 1920, h = 1080;
    int tw = 960, th = 540;
    printf("\n=== Resize Bicubic  (%dx%d -> %dx%d, 20 iters) ===\n", w, h, tw, th);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> out_base =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    Halide::Runtime::Buffer<uint8_t> out_opt =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    // Timing
    auto t_base = bench([&]() { halide_ops::resize_bicubic_target(halide_in, tw, th, out_base); });
    auto t_opt  = bench([&]() { halide_ops::resize_bicubic_optimized(halide_in, tw, th, out_opt); });
    cv::Mat cv_out;
    auto t_cv   = bench([&]() { cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_CUBIC); });

    auto p_base_vs_opt = compute_psnr_bufs(out_base, out_opt);
    // Key comparison: baseline uses Catmull-Rom (a=-0.5), optimized uses OpenCV kernel (a=-0.75)
    auto p_base_vs_cv = compute_psnr_rgb(out_base, cv_out, /*bgr=*/false);
    auto p_opt_vs_cv  = compute_psnr_rgb(out_opt, cv_out, /*bgr=*/false);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline (a=-0.5 Catmull-Rom)", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized (a=-0.75 OpenCV)", t_opt.median_us, t_opt.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "OpenCV INTER_CUBIC", t_cv.median_us, t_cv.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR base vs opt:   %.1f dB (max_diff=%d)  [different kernels!]\n", p_base_vs_opt.psnr, p_base_vs_opt.max_diff);
    printf("  PSNR base vs CV:    %.1f dB (max_diff=%d)  [a=-0.5 vs OpenCV]\n", p_base_vs_cv.psnr, p_base_vs_cv.max_diff);
    printf("  PSNR opt  vs CV:    %.1f dB (max_diff=%d)  [a=-0.75 vs OpenCV]\n", p_opt_vs_cv.psnr, p_opt_vs_cv.max_diff);
}

TEST(SelectionBenchmark, ResizeArea) {
    int w = 1920, h = 1080;
    int tw = 960, th = 540;
    printf("\n=== Resize INTER_AREA  (%dx%d -> %dx%d, 20 iters) ===\n", w, h, tw, th);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    Halide::Runtime::Buffer<uint8_t> out_base =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
    Halide::Runtime::Buffer<uint8_t> out_opt =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    // Timing
    auto t_base = bench([&]() { halide_ops::resize_area_target(halide_in, tw, th, out_base); });
    auto t_opt  = bench([&]() { halide_ops::resize_area_optimized(halide_in, tw, th, out_opt); });
    cv::Mat cv_out;
    auto t_cv   = bench([&]() { cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_AREA); });

    auto p_base_vs_opt = compute_psnr_bufs(out_base, out_opt);
    auto p_base_vs_cv = compute_psnr_rgb(out_base, cv_out, /*bgr=*/false);
    auto p_opt_vs_cv  = compute_psnr_rgb(out_opt, cv_out, /*bgr=*/false);

    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide baseline", t_base.median_us, t_base.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "Halide optimized", t_opt.median_us, t_opt.mean_us);
    printf("  %-40s median=%8.0f us  mean=%8.0f us\n", "OpenCV INTER_AREA", t_cv.median_us, t_cv.mean_us);
    printf("  Speedup (opt/base): %.2fx\n", t_base.median_us / t_opt.median_us);
    printf("  PSNR base vs opt:   %.1f dB (max_diff=%d)\n", p_base_vs_opt.psnr, p_base_vs_opt.max_diff);
    printf("  PSNR base vs CV:    %.1f dB (max_diff=%d)\n", p_base_vs_cv.psnr, p_base_vs_cv.max_diff);
    printf("  PSNR opt  vs CV:    %.1f dB (max_diff=%d)\n", p_opt_vs_cv.psnr, p_opt_vs_cv.max_diff);
}

// ---------------------------------------------------------------------------
// Odd resolution correctness (no crash + valid output)
// ---------------------------------------------------------------------------

TEST(SelectionBenchmark, OddResolution_NoCrash) {
    int w = 641, h = 481;
    int tw = 321, th = 241;
    printf("\n=== Odd Resolution Correctness  (%dx%d -> %dx%d) ===\n", w, h, tw, th);

    cv::Mat rgb_cv = make_test_image_rgb(w, h);
    auto halide_in = mat_to_halide_interleaved(rgb_cv);

    // RGB BGR
    {
        Halide::Runtime::Buffer<uint8_t> out =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
        EXPECT_EQ(0, halide_ops::rgb_bgr_optimized(halide_in, out));
        printf("  rgb_bgr_optimized:            OK\n");
    }
    // NV21 -> RGB
    {
        std::vector<uint8_t> y_data, uv_data;
        make_nv21_data(w, h, y_data, uv_data);
        // NV21 requires even dimensions for UV plane; use w-1, h-1 if odd
        int ew = w - (w % 2), eh = h - (h % 2);
        y_data.resize(ew * eh);
        uv_data.resize(ew * (eh / 2));
        make_nv21_data(ew, eh, y_data, uv_data);
        Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), ew, eh);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), ew, eh / 2);
        // NV21->RGB generator uses planar output layout
        Halide::Runtime::Buffer<uint8_t> out(ew, eh, 3);
        EXPECT_EQ(0, halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, out));
        printf("  nv21_to_rgb_optimized (%dx%d): OK\n", ew, eh);
    }
    // Resize bilinear
    {
        Halide::Runtime::Buffer<uint8_t> out =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        EXPECT_EQ(0, halide_ops::resize_bilinear_optimized(halide_in, tw, th, out));
        printf("  resize_bilinear_optimized:     OK\n");
    }
    // Resize bicubic
    {
        Halide::Runtime::Buffer<uint8_t> out =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        EXPECT_EQ(0, halide_ops::resize_bicubic_optimized(halide_in, tw, th, out));
        printf("  resize_bicubic_optimized:      OK\n");
    }
    // Resize area
    {
        Halide::Runtime::Buffer<uint8_t> out =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        EXPECT_EQ(0, halide_ops::resize_area_optimized(halide_in, tw, th, out));
        printf("  resize_area_optimized:         OK\n");
    }
    printf("  All odd-resolution tests passed.\n");
}
