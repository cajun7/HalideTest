#include "test_common.h"
#include "nv21_to_rgb.h"
#include "resize_bilinear.h"
#include "rotate_fixed_90cw.h"
#include "rotate_fixed_180.h"
#include "rotate_fixed_270cw.h"
#include "flip_horizontal.h"
#include "flip_vertical.h"
#include "nv21_pipeline_bilinear_none.h"
#include "nv21_pipeline_bilinear_90cw.h"
#include "nv21_pipeline_bilinear_180.h"
#include "nv21_pipeline_bilinear_270cw.h"
#include "nv21_pipeline_area_none.h"
#include "nv21_pipeline_area_90cw.h"
#include "nv21_pipeline_area_180.h"
#include "nv21_pipeline_area_270cw.h"

// Helper: dispatch bilinear fused pipeline by rotation code
static int call_fused_bilinear(Halide::Runtime::Buffer<uint8_t>& y,
                               Halide::Runtime::Buffer<uint8_t>& uv,
                               int rotation_cw, int flip, int tw, int th,
                               Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rotation_cw) {
        case 0:   return nv21_pipeline_bilinear_none(y, uv, flip, tw, th, out);
        case 90:  return nv21_pipeline_bilinear_90cw(y, uv, flip, tw, th, out);
        case 180: return nv21_pipeline_bilinear_180(y, uv, flip, tw, th, out);
        case 270: return nv21_pipeline_bilinear_270cw(y, uv, flip, tw, th, out);
        default:  return -1;
    }
}

// Helper: dispatch area fused pipeline by rotation code
static int call_fused_area(Halide::Runtime::Buffer<uint8_t>& y,
                           Halide::Runtime::Buffer<uint8_t>& uv,
                           int rotation_cw, int flip, int tw, int th,
                           Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rotation_cw) {
        case 0:   return nv21_pipeline_area_none(y, uv, flip, tw, th, out);
        case 90:  return nv21_pipeline_area_90cw(y, uv, flip, tw, th, out);
        case 180: return nv21_pipeline_area_180(y, uv, flip, tw, th, out);
        case 270: return nv21_pipeline_area_270cw(y, uv, flip, tw, th, out);
        default:  return -1;
    }
}

// Helper: build separate reference by doing NV21->RGB + rotate + flip + resize via OpenCV
static cv::Mat make_opencv_reference(const std::vector<uint8_t>& nv21,
                                     int src_w, int src_h,
                                     int rotation_cw, int flip_code,
                                     int tw, int th, int interp) {
    cv::Mat nv21_mat(src_h + src_h / 2, src_w, CV_8UC1,
                     const_cast<uint8_t*>(nv21.data()));
    cv::Mat rgb;
    cv::cvtColor(nv21_mat, rgb, cv::COLOR_YUV2RGB_NV21);

    cv::Mat rotated;
    switch (rotation_cw) {
        case 90:  cv::rotate(rgb, rotated, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(rgb, rotated, cv::ROTATE_180); break;
        case 270: cv::rotate(rgb, rotated, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default:  rotated = rgb; break;
    }

    cv::Mat flipped;
    if (flip_code == 1) cv::flip(rotated, flipped, 1);
    else if (flip_code == 2) cv::flip(rotated, flipped, 0);
    else flipped = rotated;

    cv::Mat resized;
    cv::resize(flipped, resized, cv::Size(tw, th), 0, 0, interp);
    return resized;
}

// Compare interleaved Halide RGB buffer against OpenCV RGB Mat (not BGR)
static void compare_rgb_vs_rgb(const Halide::Runtime::Buffer<uint8_t>& halide_buf,
                               const cv::Mat& cv_rgb, int tolerance) {
    ASSERT_EQ(halide_buf.width(), cv_rgb.cols);
    ASSERT_EQ(halide_buf.height(), cv_rgb.rows);

    int mismatches = 0, max_diff = 0;
    double mse = 0.0;
    for (int y = 0; y < halide_buf.height(); y++) {
        for (int x = 0; x < halide_buf.width(); x++) {
            for (int c = 0; c < 3; c++) {
                uint8_t h_val = halide_buf(x, y, c);
                uint8_t cv_val = cv_rgb.at<cv::Vec3b>(y, x)[c];
                int diff = std::abs((int)h_val - (int)cv_val);
                if (diff > tolerance) mismatches++;
                max_diff = std::max(max_diff, diff);
                double d = (double)h_val - (double)cv_val;
                mse += d * d;
            }
        }
    }
    int total = halide_buf.width() * halide_buf.height() * 3;
    mse /= total;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0;
    float pct = 100.0f * mismatches / total;

    // Print quality metrics for visibility
    printf("  PSNR=%.1f dB  max_diff=%d  mismatch=%.2f%%\n", psnr, max_diff, pct);

    EXPECT_GT(psnr, 30.0)
        << "PSNR too low (quality regression)";
    EXPECT_LT(pct, 5.0f)
        << "Too many pixel mismatches: " << mismatches
        << " (" << pct << "%), max_diff=" << max_diff;
}

// ---------------------------------------------------------------------------
// Fused bilinear pipeline tests
// ---------------------------------------------------------------------------

class FusedBilinearTest : public ::testing::TestWithParam<
    std::tuple<std::pair<int, int>, int, int>> {};

TEST_P(FusedBilinearTest, MatchesOpenCVReference) {
    auto [res, rotation_cw, flip_code] = GetParam();
    auto [src_w, src_h] = res;
    // Ensure even dimensions for NV21
    src_w &= ~1;
    src_h &= ~1;

    // Target is half-size (after rotation, dimensions may swap)
    int rotated_w = (rotation_cw == 90 || rotation_cw == 270) ? src_h : src_w;
    int rotated_h = (rotation_cw == 90 || rotation_cw == 270) ? src_w : src_h;
    int tw = rotated_w / 2;
    int th = rotated_h / 2;
    if (tw < 1 || th < 1) return;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = call_fused_bilinear(y_buf, uv_buf, rotation_cw, flip_code, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Fused bilinear pipeline failed: rotation=" << rotation_cw
                       << " flip=" << flip_code;

    // OpenCV reference (separate steps)
    cv::Mat opencv_ref = make_opencv_reference(nv21, src_w, src_h,
                                               rotation_cw, flip_code,
                                               tw, th, cv::INTER_LINEAR);

    // Tolerance: BT.601 rounding (~20) + bilinear interpolation differences (~3)
    // Fused pipeline samples NV21 directly with interpolation, while reference
    // converts to RGB first then interpolates — different rounding order.
    compare_rgb_vs_rgb(output_buf, opencv_ref, /*tolerance=*/25);
}

INSTANTIATE_TEST_SUITE_P(
    Configs,
    FusedBilinearTest,
    ::testing::Combine(
        ::testing::Values(
            std::make_pair(320, 240),
            std::make_pair(640, 480),
            std::make_pair(1280, 720)
        ),
        ::testing::Values(0, 90, 180, 270),   // rotation
        ::testing::Values(0, 1, 2)             // flip: none, horizontal, vertical
    )
);

// ---------------------------------------------------------------------------
// Fused area pipeline tests
// ---------------------------------------------------------------------------

class FusedAreaTest : public ::testing::TestWithParam<
    std::tuple<std::pair<int, int>, int, int>> {};

TEST_P(FusedAreaTest, MatchesOpenCVReference) {
    auto [res, rotation_cw, flip_code] = GetParam();
    auto [src_w, src_h] = res;
    src_w &= ~1;
    src_h &= ~1;

    int rotated_w = (rotation_cw == 90 || rotation_cw == 270) ? src_h : src_w;
    int rotated_h = (rotation_cw == 90 || rotation_cw == 270) ? src_w : src_h;
    int tw = rotated_w / 2;
    int th = rotated_h / 2;
    if (tw < 1 || th < 1) return;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = call_fused_area(y_buf, uv_buf, rotation_cw, flip_code, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Fused area pipeline failed: rotation=" << rotation_cw
                       << " flip=" << flip_code;

    cv::Mat opencv_ref = make_opencv_reference(nv21, src_w, src_h,
                                               rotation_cw, flip_code,
                                               tw, th, cv::INTER_AREA);

    compare_rgb_vs_rgb(output_buf, opencv_ref, /*tolerance=*/25);
}

INSTANTIATE_TEST_SUITE_P(
    Configs,
    FusedAreaTest,
    ::testing::Combine(
        ::testing::Values(
            std::make_pair(320, 240),
            std::make_pair(640, 480),
            std::make_pair(1280, 720)
        ),
        ::testing::Values(0, 90, 180, 270),
        ::testing::Values(0, 1, 2)             // flip: none, horizontal, vertical
    )
);

// ---------------------------------------------------------------------------
// Fused area: non-uniform scale (asymmetric downscale ratios)
// ---------------------------------------------------------------------------

struct AreaScaleCase {
    int src_w, src_h, tw, th, rotation_cw, flip_code;
};

class FusedAreaScaleTest : public ::testing::TestWithParam<AreaScaleCase> {};

TEST_P(FusedAreaScaleTest, MatchesOpenCVReference) {
    auto p = GetParam();
    int src_w = p.src_w & ~1;
    int src_h = p.src_h & ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(p.tw, p.th, 3);

    int err = call_fused_area(y_buf, uv_buf, p.rotation_cw, p.flip_code,
                              p.tw, p.th, output_buf);
    ASSERT_EQ(err, 0) << "Failed: " << src_w << "x" << src_h
                       << " rot=" << p.rotation_cw << " flip=" << p.flip_code
                       << " -> " << p.tw << "x" << p.th;

    cv::Mat opencv_ref = make_opencv_reference(nv21, src_w, src_h,
                                               p.rotation_cw, p.flip_code,
                                               p.tw, p.th, cv::INTER_AREA);

    compare_rgb_vs_rgb(output_buf, opencv_ref, /*tolerance=*/25);
}

INSTANTIATE_TEST_SUITE_P(
    NonUniform,
    FusedAreaScaleTest,
    ::testing::Values(
        // Asymmetric targets
        AreaScaleCase{640, 480, 480, 320, 0, 0},
        AreaScaleCase{640, 480, 320, 180, 90, 0},
        AreaScaleCase{1280, 720, 960, 540, 0, 0},
        AreaScaleCase{1280, 720, 640, 360, 90, 1},
        // Benchmark-like: 1080p -> 720p with 90° rotation
        AreaScaleCase{1920, 1080, 1280, 720, 90, 0},
        // Odd source dimensions
        AreaScaleCase{642, 482, 320, 240, 0, 0},
        AreaScaleCase{642, 482, 240, 320, 90, 0},
        AreaScaleCase{1278, 718, 640, 360, 180, 0},
        // Stronger downscale (tests robustness beyond max_pool=8)
        AreaScaleCase{640, 480, 160, 120, 0, 0},
        AreaScaleCase{640, 480, 120, 90, 90, 0},
        // At max_pool boundary (8x downscale)
        AreaScaleCase{640, 480, 80, 60, 0, 0},
        // With flip
        AreaScaleCase{640, 480, 320, 240, 0, 1},
        AreaScaleCase{640, 480, 320, 240, 90, 2},
        AreaScaleCase{640, 480, 320, 240, 270, 1}
    )
);

// ---------------------------------------------------------------------------
// Separable correctness: verify Halide separable output matches a C++
// non-separable reference implementing the FULL fused pipeline
// (NV21 area-filter + BT.601 → RGB) without separability.
// ---------------------------------------------------------------------------

// Full C++ non-separable fused pipeline: area-filter NV21 + BT.601 → RGB.
static void ref_fused_pixel(const uint8_t* y_data, int src_w, int src_h,
                            const uint8_t* uv_data,
                            float nv21_left, float nv21_right,
                            float nv21_top, float nv21_bot,
                            uint8_t out_rgb[3]) {
    auto clampf = [](float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); };

    // Area-average Y (non-separable 2D loop)
    float y_sum = 0, y_wt = 0;
    for (int py = (int)std::floor(nv21_top); py < (int)std::ceil(nv21_bot); py++) {
        float wy = std::max(std::min((float)py+1, nv21_bot) - std::max((float)py, nv21_top), 0.0f);
        for (int px = (int)std::floor(nv21_left); px < (int)std::ceil(nv21_right); px++) {
            float wx = std::max(std::min((float)px+1, nv21_right) - std::max((float)px, nv21_left), 0.0f);
            float w = wx * wy;
            int cx = std::max(0, std::min(px, src_w - 1));
            int cy = std::max(0, std::min(py, src_h - 1));
            y_sum += w * (float)y_data[cy * src_w + cx];
            y_wt  += w;
        }
    }

    // Area-average UV at half resolution (non-separable 2D loop)
    float uv_left = nv21_left / 2, uv_right = nv21_right / 2;
    float uv_top = nv21_top / 2,   uv_bot = nv21_bot / 2;
    int uv_w = src_w / 2, uv_h = src_h / 2;
    float v_sum = 0, u_sum = 0, uv_wt = 0;
    for (int py = (int)std::floor(uv_top); py < (int)std::ceil(uv_bot); py++) {
        float wy = std::max(std::min((float)py+1, uv_bot) - std::max((float)py, uv_top), 0.0f);
        for (int px = (int)std::floor(uv_left); px < (int)std::ceil(uv_right); px++) {
            float wx = std::max(std::min((float)px+1, uv_right) - std::max((float)px, uv_left), 0.0f);
            float w = wx * wy;
            int cx = std::max(0, std::min(px, uv_w - 1));
            int cy = std::max(0, std::min(py, uv_h - 1));
            v_sum += w * (float)uv_data[cy * src_w + cx * 2];
            u_sum += w * (float)uv_data[cy * src_w + cx * 2 + 1];
            uv_wt += w;
        }
    }

    float y_avg = (y_wt > 1e-4f) ? y_sum / y_wt : 0;
    float v_avg = (uv_wt > 1e-4f) ? v_sum / uv_wt : 128;
    float u_avg = (uv_wt > 1e-4f) ? u_sum / uv_wt : 128;

    // BT.601 (identical to the Halide generator)
    int yi = (int)clampf(y_avg, 0, 255);
    int vi = (int)clampf(v_avg, 0, 255) - 128;
    int ui = (int)clampf(u_avg, 0, 255) - 128;
    int ys = (yi - 16) * 298 + 128;
    out_rgb[0] = (uint8_t)std::max(0, std::min(255, (ys + 409 * vi) >> 8));
    out_rgb[1] = (uint8_t)std::max(0, std::min(255, (ys - 100 * ui - 208 * vi) >> 8));
    out_rgb[2] = (uint8_t)std::max(0, std::min(255, (ys + 516 * ui) >> 8));
}

TEST(FusedAreaQuality, SeparableMatchesNonSeparable) {
    // 640x480 source, no rotation, no flip, half-size target
    int src_w = 640, src_h = 480, tw = 320, th = 240;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);

    int err = call_fused_area(y_buf, uv_buf, 0, 0, tw, th, output_buf);
    ASSERT_EQ(err, 0);

    int max_diff = 0, mismatches = 0;
    for (int oy = 0; oy < th; oy++) {
        for (int ox = 0; ox < tw; ox++) {
            float left  = (float)ox * src_w / tw;
            float right = ((float)ox + 1.0f) * src_w / tw;
            float top   = (float)oy * src_h / th;
            float bot   = ((float)oy + 1.0f) * src_h / th;

            uint8_t ref_rgb[3];
            ref_fused_pixel(y_ptr, src_w, src_h, uv_ptr,
                            left, right, top, bot, ref_rgb);

            for (int c = 0; c < 3; c++) {
                int d = std::abs((int)output_buf(ox, oy, c) - (int)ref_rgb[c]);
                max_diff = std::max(max_diff, d);
                if (d > 1) mismatches++;
            }
        }
    }
    float pct = 100.0f * mismatches / (tw * th * 3);
    printf("  Separable vs non-separable RGB: max_diff=%d, mismatch(>1)=%.2f%%\n",
           max_diff, pct);
    // Separable decomposition is mathematically exact.
    // Only float rounding order differs: ≤1 per channel expected.
    EXPECT_LE(max_diff, 2)
        << "Separable and non-separable differ by more than float rounding";
    EXPECT_LT(pct, 1.0f)
        << "Too many mismatches: " << mismatches;
}

// ---------------------------------------------------------------------------
// NoCrash tests for edge cases
// ---------------------------------------------------------------------------

class FusedNoCrashTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(FusedNoCrashTest, Bilinear_AllRotations_NoCrash) {
    auto [src_w, src_h] = GetParam();
    src_w &= ~1;
    src_h &= ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    int rotations[] = {0, 90, 180, 270};
    for (int rot : rotations) {
        int rw = (rot == 90 || rot == 270) ? src_h : src_w;
        int rh = (rot == 90 || rot == 270) ? src_w : src_h;
        int tw = std::max(rw / 3, 1);
        int th = std::max(rh / 3, 1);

        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        int err = call_fused_bilinear(y_buf, uv_buf, rot, 0, tw, th, out);
        ASSERT_EQ(err, 0) << "Crashed: " << src_w << "x" << src_h
                          << " rot=" << rot << " -> " << tw << "x" << th;
    }
}

TEST_P(FusedNoCrashTest, Bilinear_WithFlip_NoCrash) {
    auto [src_w, src_h] = GetParam();
    src_w &= ~1;
    src_h &= ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    int tw = src_w / 2, th = src_h / 2;
    for (int flip = 0; flip <= 2; flip++) {
        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        int err = call_fused_bilinear(y_buf, uv_buf, 90, flip, tw, th, out);
        ASSERT_EQ(err, 0) << "Crashed: rot=90 flip=" << flip;
    }
}

// Extreme downscale beyond max_pool: must not crash (quality is approximate)
TEST(FusedAreaRobustness, ExtremeDownscale_NoCrash) {
    int src_w = 640, src_h = 480;
    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    // 20x downscale (exceeds max_pool=8)
    int targets[][4] = {{32, 24, 0, 0}, {32, 24, 90, 0}, {64, 48, 180, 0},
                        {32, 24, 270, 0}, {32, 24, 90, 1}, {32, 24, 0, 2}};
    for (auto& t : targets) {
        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(t[0], t[1], 3);
        int err = call_fused_area(y_buf, uv_buf, t[2], t[3], t[0], t[1], out);
        ASSERT_EQ(err, 0) << "Crashed: " << t[0] << "x" << t[1]
                          << " rot=" << t[2] << " flip=" << t[3];
    }
}

TEST_P(FusedNoCrashTest, Area_AllRotations_NoCrash) {
    auto [src_w, src_h] = GetParam();
    src_w &= ~1;
    src_h &= ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    int rotations[] = {0, 90, 180, 270};
    for (int rot : rotations) {
        int rw = (rot == 90 || rot == 270) ? src_h : src_w;
        int rh = (rot == 90 || rot == 270) ? src_w : src_h;
        int tw = std::max(rw / 3, 1);
        int th = std::max(rh / 3, 1);

        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        int err = call_fused_area(y_buf, uv_buf, rot, 0, tw, th, out);
        ASSERT_EQ(err, 0) << "Crashed: " << src_w << "x" << src_h
                          << " rot=" << rot << " -> " << tw << "x" << th;
    }
}

TEST_P(FusedNoCrashTest, Area_WithFlip_NoCrash) {
    auto [src_w, src_h] = GetParam();
    src_w &= ~1;
    src_h &= ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;
    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    int tw = src_w / 2, th = src_h / 2;
    for (int flip = 0; flip <= 2; flip++) {
        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(tw, th, 3);
        int err = call_fused_area(y_buf, uv_buf, 90, flip, tw, th, out);
        ASSERT_EQ(err, 0) << "Crashed: rot=90 flip=" << flip;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    FusedNoCrashTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(642, 482),
        std::make_pair(1280, 720)
    )
);
