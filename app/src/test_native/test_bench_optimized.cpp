#include "test_common.h"
#include "halide_ops.h"
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

static void bench(const char* name, std::function<void()> fn, int iters = 10) {
    // warmup
    fn();
    auto start = Clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto end = Clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)iters;
    printf("  %-50s %8.0f us\n", name, us);
}

TEST(Benchmark, OptimizedVsBaseline) {
    int w = 1920, h = 1080;
    int tw = 960, th = 540;

    printf("\n=== Benchmark: 1920x1080 -> 960x540 (10 iterations) ===\n\n");

    // --- RGB Bilinear Resize ---
    {
        cv::Mat rgb_cv = make_test_image_rgb(w, h);
        auto halide_in = mat_to_halide_interleaved(rgb_cv);

        Halide::Runtime::Buffer<uint8_t> out_opt =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        Halide::Runtime::Buffer<uint8_t> out_base =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

        bench("Halide resize_bilinear_optimized", [&]() {
            halide_ops::resize_bilinear_optimized(halide_in, tw, th, out_opt);
        });
        bench("Halide resize_bilinear_target (baseline)", [&]() {
            halide_ops::resize_bilinear_target(halide_in, tw, th, out_base);
        });

        cv::Mat cv_out;
        bench("OpenCV cv::resize INTER_LINEAR", [&]() {
            cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);
        });
    }

    printf("\n");

    // --- RGB Area Resize ---
    {
        cv::Mat rgb_cv = make_test_image_rgb(w, h);
        auto halide_in = mat_to_halide_interleaved(rgb_cv);

        Halide::Runtime::Buffer<uint8_t> out_opt =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        Halide::Runtime::Buffer<uint8_t> out_base =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

        bench("Halide resize_area_optimized", [&]() {
            halide_ops::resize_area_optimized(halide_in, tw, th, out_opt);
        });
        bench("Halide resize_area_target (baseline)", [&]() {
            halide_ops::resize_area_target(halide_in, tw, th, out_base);
        });

        cv::Mat cv_out;
        bench("OpenCV cv::resize INTER_AREA", [&]() {
            cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
        });
    }

    printf("\n");

    // --- RGB Bicubic Resize ---
    {
        cv::Mat rgb_cv = make_test_image_rgb(w, h);
        auto halide_in = mat_to_halide_interleaved(rgb_cv);

        Halide::Runtime::Buffer<uint8_t> out_opt =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        Halide::Runtime::Buffer<uint8_t> out_base =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

        bench("Halide resize_bicubic_optimized", [&]() {
            halide_ops::resize_bicubic_optimized(halide_in, tw, th, out_opt);
        });
        bench("Halide resize_bicubic_target (baseline)", [&]() {
            halide_ops::resize_bicubic_target(halide_in, tw, th, out_base);
        });

        cv::Mat cv_out;
        bench("OpenCV cv::resize INTER_CUBIC", [&]() {
            cv::resize(rgb_cv, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_CUBIC);
        });
    }

    printf("\n");

    // --- RGB <-> BGR ---
    {
        cv::Mat rgb_cv = make_test_image_rgb(w, h);
        auto halide_in = mat_to_halide_interleaved(rgb_cv);

        Halide::Runtime::Buffer<uint8_t> out_opt =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
        Halide::Runtime::Buffer<uint8_t> out_base =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);

        bench("Halide rgb_bgr_optimized", [&]() {
            halide_ops::rgb_bgr_optimized(halide_in, out_opt);
        });
        bench("Halide rgb_bgr (baseline)", [&]() {
            halide_ops::rgb_bgr(halide_in, out_base);
        });

        cv::Mat cv_out;
        bench("OpenCV cvtColor RGB2BGR", [&]() {
            cv::cvtColor(rgb_cv, cv_out, cv::COLOR_RGB2BGR);
        });
    }

    printf("\n");

    // --- NV21 -> RGB ---
    {
        std::vector<uint8_t> y_data, uv_data;
        make_nv21_data(w, h, y_data, uv_data);
        Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), w, h / 2);

        Halide::Runtime::Buffer<uint8_t> out_opt(w, h, 3);
        Halide::Runtime::Buffer<uint8_t> out_base(w, h, 3);

        bench("Halide nv21_to_rgb_optimized", [&]() {
            halide_ops::nv21_to_rgb_optimized(y_buf, uv_buf, out_opt);
        });
        bench("Halide nv21_to_rgb (baseline)", [&]() {
            halide_ops::nv21_to_rgb(y_buf, uv_buf, out_base);
        });

        std::vector<uint8_t> nv21(y_data.size() + uv_data.size());
        std::copy(y_data.begin(), y_data.end(), nv21.begin());
        std::copy(uv_data.begin(), uv_data.end(), nv21.begin() + y_data.size());
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
        cv::Mat cv_out;
        bench("OpenCV cvtColor NV21->RGB", [&]() {
            cv::cvtColor(nv21_mat, cv_out, cv::COLOR_YUV2RGB_NV21);
        });
    }

    printf("\n");

    // --- NV21-domain Bilinear Resize ---
    {
        std::vector<uint8_t> y_data, uv_data;
        make_nv21_data(w, h, y_data, uv_data);
        Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), w, h / 2);

        Halide::Runtime::Buffer<uint8_t> y_out(tw, th);
        Halide::Runtime::Buffer<uint8_t> uv_out(tw, th / 2);

        bench("Halide nv21_resize_bilinear_optimized", [&]() {
            halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, tw, th, y_out, uv_out);
        });

        // OpenCV reference: NV21->RGB->resize (no direct NV21 resize in OpenCV)
        std::vector<uint8_t> nv21(y_data.size() + uv_data.size());
        std::copy(y_data.begin(), y_data.end(), nv21.begin());
        std::copy(uv_data.begin(), uv_data.end(), nv21.begin() + y_data.size());
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
        cv::Mat cv_rgb, cv_resized;
        bench("OpenCV NV21->RGB->resize (reference)", [&]() {
            cv::cvtColor(nv21_mat, cv_rgb, cv::COLOR_YUV2RGB_NV21);
            cv::resize(cv_rgb, cv_resized, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);
        });
    }

    printf("\n");

    // --- Fused NV21 Resize -> RGB ---
    {
        std::vector<uint8_t> y_data, uv_data;
        make_nv21_data(w, h, y_data, uv_data);
        Halide::Runtime::Buffer<uint8_t> y_buf(y_data.data(), w, h);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), w, h / 2);

        Halide::Runtime::Buffer<uint8_t> out_fused =
            Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

        bench("Halide nv21_resize_rgb_bilinear_optimized (fused)", [&]() {
            halide_ops::nv21_resize_rgb_bilinear_optimized(y_buf, uv_buf, tw, th, out_fused);
        });

        // OpenCV reference: NV21->RGB->resize (two steps)
        std::vector<uint8_t> nv21(y_data.size() + uv_data.size());
        std::copy(y_data.begin(), y_data.end(), nv21.begin());
        std::copy(uv_data.begin(), uv_data.end(), nv21.begin() + y_data.size());
        cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
        cv::Mat cv_rgb, cv_out;
        bench("OpenCV NV21->RGB->resize (two steps)", [&]() {
            cv::cvtColor(nv21_mat, cv_rgb, cv::COLOR_YUV2RGB_NV21);
            cv::resize(cv_rgb, cv_out, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);
        });
    }

    printf("\n=== Done ===\n");
}
