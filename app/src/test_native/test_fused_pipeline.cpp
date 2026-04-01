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
    for (int y = 0; y < halide_buf.height(); y++) {
        for (int x = 0; x < halide_buf.width(); x++) {
            for (int c = 0; c < 3; c++) {
                uint8_t h_val = halide_buf(x, y, c);
                uint8_t cv_val = cv_rgb.at<cv::Vec3b>(y, x)[c];
                int diff = std::abs((int)h_val - (int)cv_val);
                if (diff > tolerance) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float pct = 100.0f * mismatches / (halide_buf.width() * halide_buf.height() * 3);
    EXPECT_LT(pct, 2.0f)
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
    std::tuple<std::pair<int, int>, int>> {};

TEST_P(FusedAreaTest, NoFlip_MatchesOpenCVReference) {
    auto [res, rotation_cw] = GetParam();
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

    int err = call_fused_area(y_buf, uv_buf, rotation_cw, 0, tw, th, output_buf);
    ASSERT_EQ(err, 0) << "Fused area pipeline failed: rotation=" << rotation_cw;

    cv::Mat opencv_ref = make_opencv_reference(nv21, src_w, src_h,
                                               rotation_cw, 0,
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
        ::testing::Values(0, 90, 180, 270)
    )
);

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
