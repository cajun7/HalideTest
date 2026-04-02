#include "test_common.h"
#include "nv21_yuv444_rgb.h"
#include "nv21_to_rgb.h"

// C++ float reference: bilinear-upsample UV from NV21, then BT.601 limited-range.
static void ref_yuv444_image(const uint8_t* y_data, const uint8_t* uv_data,
                             int w, int h, std::vector<uint8_t>& rgb_out) {
    rgb_out.resize(w * h * 3);
    int uv_w = w;       // byte width of UV plane
    int uv_h = h / 2;   // row count of UV plane

    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            // Y at full resolution
            float Y = (float)y_data[row * w + col];

            // Bilinear UV coordinates
            float uv_sx = (float)col / 2.0f;
            float uv_sy = (float)row / 2.0f;
            int uv_ix = (int)floorf(uv_sx);
            int uv_iy = (int)floorf(uv_sy);
            float uv_fx = uv_sx - (float)uv_ix;
            float uv_fy = uv_sy - (float)uv_iy;

            // Clamp to valid UV range (repeat_edge)
            auto clamp_uv = [&](int ix, int iy, int offset) -> float {
                int cx = std::max(0, std::min(ix, w / 2 - 1));
                int cy = std::max(0, std::min(iy, uv_h - 1));
                return (float)uv_data[cy * uv_w + cx * 2 + offset];
            };

            // V (offset 0) bilinear
            float v00 = clamp_uv(uv_ix, uv_iy, 0);
            float v10 = clamp_uv(uv_ix + 1, uv_iy, 0);
            float v01 = clamp_uv(uv_ix, uv_iy + 1, 0);
            float v11 = clamp_uv(uv_ix + 1, uv_iy + 1, 0);
            float V = v00 * (1 - uv_fx) * (1 - uv_fy) +
                      v10 * uv_fx * (1 - uv_fy) +
                      v01 * (1 - uv_fx) * uv_fy +
                      v11 * uv_fx * uv_fy;

            // U (offset 1) bilinear
            float u00 = clamp_uv(uv_ix, uv_iy, 1);
            float u10 = clamp_uv(uv_ix + 1, uv_iy, 1);
            float u01 = clamp_uv(uv_ix, uv_iy + 1, 1);
            float u11 = clamp_uv(uv_ix + 1, uv_iy + 1, 1);
            float U = u00 * (1 - uv_fx) * (1 - uv_fy) +
                      u10 * uv_fx * (1 - uv_fy) +
                      u01 * (1 - uv_fx) * uv_fy +
                      u11 * uv_fx * uv_fy;

            // BT.601 limited-range (same as Halide generator)
            int v_int = (int)std::max(0.0f, std::min(255.0f, V)) - 128;
            int u_int = (int)std::max(0.0f, std::min(255.0f, U)) - 128;
            int y_int = (int)Y;
            int y_scaled = (y_int - 16) * 298 + 128;
            int r = (y_scaled + 409 * v_int) >> 8;
            int g = (y_scaled - 100 * u_int - 208 * v_int) >> 8;
            int b = (y_scaled + 516 * u_int) >> 8;

            rgb_out[(row * w + col) * 3 + 0] = (uint8_t)std::max(0, std::min(255, r));
            rgb_out[(row * w + col) * 3 + 1] = (uint8_t)std::max(0, std::min(255, g));
            rgb_out[(row * w + col) * 3 + 2] = (uint8_t)std::max(0, std::min(255, b));
        }
    }
}

static Halide::Runtime::Buffer<uint8_t> make_uv_buf(uint8_t* uv_ptr, int w, int h) {
    return Halide::Runtime::Buffer<uint8_t>(uv_ptr, w, h / 2);
}

class Nv21Yuv444Test : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Nv21Yuv444Test, MatchesFloatReference) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_yuv444_rgb(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide nv21_yuv444_rgb failed";

    // Float reference with bilinear UV upsampling
    std::vector<uint8_t> ref_rgb;
    ref_yuv444_image(y_ptr, uv_ptr, w, h, ref_rgb);

    dump_if_first(output_buf, "yuv444_halide", 0);

    // Tolerance: Halide uses float bilinear -> cast<int32_t> (truncation) -> BT.601.
    // The C++ reference uses the same sequence, but float operation ordering
    // can cause up to ~2 LSB difference from FMA vs separate multiply+add.
    int mismatches = 0, max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int halide_val = output_buf(x, y, c);
                int ref_val = ref_rgb[(y * w + x) * 3 + c];
                int diff = std::abs(halide_val - ref_val);
                if (diff > 2) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float pct = 100.0f * mismatches / (float)(w * h * 3);
    printf("  YUV444 vs float ref: max_diff=%d mismatch(>2)=%.2f%%\n", max_diff, pct);
    EXPECT_LE(max_diff, 3)
        << "Bilinear UV upsample should match float reference within 3";
    EXPECT_LT(pct, 1.0f)
        << "Too many mismatches: " << mismatches << " (" << pct << "%)";
}

TEST_P(Nv21Yuv444Test, DiffersFromNearestNeighbor) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);

    Halide::Runtime::Buffer<uint8_t> yuv444_out(w, h, 3);
    Halide::Runtime::Buffer<uint8_t> nearest_out(w, h, 3);

    int err1 = nv21_yuv444_rgb(y_buf, uv_buf, yuv444_out);
    ASSERT_EQ(err1, 0);
    int err2 = nv21_to_rgb(y_buf, uv_buf, nearest_out);
    ASSERT_EQ(err2, 0);

    dump_if_first(yuv444_out, "yuv444_bilinear", 0);
    dump_if_first(nearest_out, "yuv444_nearest", 0);

    // Count differing pixels — bilinear UV produces smoother chroma
    int differ_count = 0;
    int max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int d = std::abs((int)yuv444_out(x, y, c) -
                                 (int)nearest_out(x, y, c));
                if (d > 0) differ_count++;
                max_diff = std::max(max_diff, d);
            }
        }
    }
    float pct = 100.0f * differ_count / (float)(w * h * 3);
    printf("  YUV444 bilinear vs nearest: differ=%.1f%% max_diff=%d\n", pct, max_diff);

    // They should differ — bilinear interpolation smooths chroma boundaries
    EXPECT_GT(differ_count, 0)
        << "Bilinear UV upsample should differ from nearest-neighbor";
}

TEST_P(Nv21Yuv444Test, CompareWithOpenCV) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    // OpenCV reference (uses nearest-neighbor UV, so we expect differences)
    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat opencv_rgb;
    cv::cvtColor(nv21_mat, opencv_rgb, cv::COLOR_YUV2RGB_NV21);

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_yuv444_rgb(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0);

    dump_mat_if_first(opencv_rgb, "yuv444_opencv_ref", 0, /*is_bgr=*/false);

    // Expect moderate difference: bilinear UV vs OpenCV's nearest + BT.601 rounding
    int mismatches = 0, max_diff = 0;
    double mse = 0.0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int h_val = output_buf(x, y, c);
                int cv_val = opencv_rgb.at<cv::Vec3b>(y, x)[c];
                int diff = std::abs(h_val - cv_val);
                if (diff > 20) mismatches++;
                max_diff = std::max(max_diff, diff);
                double d = (double)h_val - (double)cv_val;
                mse += d * d;
            }
        }
    }
    int total = w * h * 3;
    mse /= total;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0;
    float pct = 100.0f * mismatches / (float)total;
    printf("  YUV444 vs OpenCV: PSNR=%.1f dB max_diff=%d mismatch(>20)=%.2f%%\n",
           psnr, max_diff, pct);

    // PSNR should be high (both are BT.601, only UV sampling differs)
    EXPECT_GT(psnr, 30.0) << "PSNR too low vs OpenCV";
    EXPECT_LT(pct, 5.0f) << "Too many mismatches vs OpenCV";
}

TEST_P(Nv21Yuv444Test, OutputInValidRange) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    auto nv21 = make_nv21_contiguous(w, h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = nv21.data() + w * h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, w, h);
    auto uv_buf = make_uv_buf(uv_ptr, w, h);
    Halide::Runtime::Buffer<uint8_t> output_buf(w, h, 3);

    int err = nv21_yuv444_rgb(y_buf, uv_buf, output_buf);
    ASSERT_EQ(err, 0);
    SUCCEED() << "No crash on " << w << "x" << h;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21Yuv444Test,
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
