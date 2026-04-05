// =============================================================================
// Tests for seg_bg_replace Halide generator
// =============================================================================
//
// Test strategy:
//   Correctness: Compare fused Halide output against a C++ reference that
//   chains mask upsample → bg resize → alpha blend.
//
//   Quality metrics:
//   - Per-pixel tolerance: 3 (bilinear rounding + alpha blend rounding)
//   - PSNR > 45 dB against C++ reference
//   - Mismatch < 1% of pixels
//
//   Performance: Time Halide fused vs OpenCV chained across resolutions.
//   Expected ~3-4x speedup (no heavy blur, memory-bandwidth bound).
//
// Test cases:
//   1. MatchesReference      - Compare against C++ reference
//   2. AllForeground         - fg_class covers all → output = fg_image
//   3. AllBackground         - fg_class matches nothing → output = resized bg_image
//   4. DifferentBgResolution - bg_image at different resolution from fg
//   5. OddResolutions        - Edge case verification
//   6. TimingBenchmark        - Halide vs OpenCV
//
// =============================================================================

#include "test_common.h"
#include "seg_bg_replace.h"
#include "opencv_ops.h"
#include <chrono>

static const int FG_CLASS = 1;
static const float EDGE_SOFTNESS = 3.0f;
static const int TOLERANCE = 3;
static const int NUM_WARMUP = 3;
static const int NUM_ITERS = 10;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

static Halide::Runtime::Buffer<uint8_t> make_seg_mask_rect(int mask_w, int mask_h,
                                                            int fg_class,
                                                            float fg_ratio = 0.4f) {
    Halide::Runtime::Buffer<uint8_t> mask(mask_w, mask_h);
    int x0 = (int)(mask_w * (0.5f - fg_ratio / 2));
    int x1 = (int)(mask_w * (0.5f + fg_ratio / 2));
    int y0 = (int)(mask_h * (0.5f - fg_ratio / 2));
    int y1 = (int)(mask_h * (0.5f + fg_ratio / 2));
    for (int y = 0; y < mask_h; y++)
        for (int x = 0; x < mask_w; x++)
            mask(x, y) = (x >= x0 && x < x1 && y >= y0 && y < y1) ?
                          (uint8_t)fg_class : (uint8_t)0;
    return mask;
}

static Halide::Runtime::Buffer<uint8_t> make_uniform_mask(int w, int h, uint8_t cls) {
    Halide::Runtime::Buffer<uint8_t> mask(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            mask(x, y) = cls;
    return mask;
}

// C++ reference matching the Halide generator
static void reference_bg_replace(
    const Halide::Runtime::Buffer<uint8_t>& fg,
    const Halide::Runtime::Buffer<uint8_t>& bg,
    const Halide::Runtime::Buffer<uint8_t>& seg_mask,
    int fg_class, float edge_softness,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    int w = fg.dim(0).extent();
    int h = fg.dim(1).extent();
    int mw = seg_mask.dim(0).extent();
    int mh = seg_mask.dim(1).extent();
    int bw = bg.dim(0).extent();
    int bh = bg.dim(1).extent();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Alpha mask
            float mask_x = (x + 0.5f) * mw / (float)w - 0.5f;
            float mask_y = (y + 0.5f) * mh / (float)h - 0.5f;
            int mx = (int)floorf(mask_x);
            int my = (int)floorf(mask_y);
            float mfx = mask_x - mx;
            float mfy = mask_y - my;

            auto msamp = [&](int px, int py) -> float {
                px = std::max(0, std::min(px, mw - 1));
                py = std::max(0, std::min(py, mh - 1));
                return (seg_mask(px, py) == fg_class) ? 1.0f : 0.0f;
            };

            float raw = msamp(mx, my) * (1 - mfx) * (1 - mfy) +
                         msamp(mx + 1, my) * mfx * (1 - mfy) +
                         msamp(mx, my + 1) * (1 - mfx) * mfy +
                         msamp(mx + 1, my + 1) * mfx * mfy;

            float alpha = std::max(0.0f, std::min(1.0f,
                (raw - 0.5f) * edge_softness + 0.5f));

            // Background bilinear sample
            float bg_src_x = (x + 0.5f) * bw / (float)w - 0.5f;
            float bg_src_y = (y + 0.5f) * bh / (float)h - 0.5f;
            int bx = (int)floorf(bg_src_x);
            int by = (int)floorf(bg_src_y);
            float bfx = bg_src_x - bx;
            float bfy = bg_src_y - by;

            for (int c = 0; c < 3; c++) {
                auto bsamp = [&](int px, int py) -> float {
                    px = std::max(0, std::min(px, bw - 1));
                    py = std::max(0, std::min(py, bh - 1));
                    return (float)bg(px, py, c);
                };

                float bg_val = bsamp(bx, by) * (1 - bfx) * (1 - bfy) +
                               bsamp(bx + 1, by) * bfx * (1 - bfy) +
                               bsamp(bx, by + 1) * (1 - bfx) * bfy +
                               bsamp(bx + 1, by + 1) * bfx * bfy;

                float fg_val = (float)fg(x, y, c);
                float val = alpha * fg_val + (1.0f - alpha) * bg_val + 0.5f;
                output(x, y, c) = (uint8_t)std::max(0.0f, std::min(255.0f, val));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

struct BgReplaceTestCase {
    int fg_w, fg_h;
    int bg_w, bg_h;
    int mask_w, mask_h;
};

class SegBgReplaceTest : public ::testing::TestWithParam<BgReplaceTestCase> {};

// ---------------------------------------------------------------------------
// Test 1: Matches reference
// ---------------------------------------------------------------------------
TEST_P(SegBgReplaceTest, MatchesReference) {
    auto tc = GetParam();

    cv::Mat fg_mat = make_test_image_rgb(tc.fg_w, tc.fg_h);
    auto fg = mat_to_halide_interleaved(fg_mat);

    // Background: different gradient pattern
    cv::Mat bg_mat(tc.bg_h, tc.bg_w, CV_8UC3);
    for (int y = 0; y < tc.bg_h; y++)
        for (int x = 0; x < tc.bg_w; x++)
            bg_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uint8_t)(255 - x * 255 / std::max(tc.bg_w - 1, 1)),
                (uint8_t)((x * y) % 256),
                (uint8_t)(y * 255 / std::max(tc.bg_h - 1, 1)));
    auto bg = mat_to_halide_interleaved(bg_mat);

    auto seg_mask = make_seg_mask_rect(tc.mask_w, tc.mask_h, FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.fg_w, tc.fg_h, 3);
    Halide::Runtime::Buffer<uint8_t> ref =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.fg_w, tc.fg_h, 3);

    int err = seg_bg_replace(fg, bg, seg_mask, FG_CLASS, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0) << "Halide seg_bg_replace failed";

    reference_bg_replace(fg, bg, seg_mask, FG_CLASS, EDGE_SOFTNESS, ref);

    int mismatches = 0;
    int max_diff = 0;
    double mse_sum = 0;
    int total = tc.fg_w * tc.fg_h * 3;
    for (int y = 0; y < tc.fg_h; y++)
        for (int x = 0; x < tc.fg_w; x++)
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)output(x, y, c) - (int)ref(x, y, c));
                mse_sum += diff * diff;
                if (diff > TOLERANCE) mismatches++;
                max_diff = std::max(max_diff, diff);
            }

    double mse = mse_sum / total;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    float mismatch_pct = 100.0f * mismatches / total;

    printf("  [fg=%dx%d, bg=%dx%d, mask=%dx%d] PSNR=%.1f dB, max_diff=%d, mismatch=%.2f%%\n",
           tc.fg_w, tc.fg_h, tc.bg_w, tc.bg_h, tc.mask_w, tc.mask_h,
           psnr, max_diff, mismatch_pct);

    EXPECT_GT(psnr, 45.0) << "PSNR too low";
    EXPECT_LT(mismatch_pct, 1.0f) << "Too many pixel mismatches";
}

// ---------------------------------------------------------------------------
// Test 2: All foreground → output = fg_image
// ---------------------------------------------------------------------------
TEST_P(SegBgReplaceTest, AllForeground_OutputEqualsFg) {
    auto tc = GetParam();

    cv::Mat fg_mat = make_test_image_rgb(tc.fg_w, tc.fg_h);
    auto fg = mat_to_halide_interleaved(fg_mat);
    cv::Mat bg_mat = make_test_image_rgb(tc.bg_w, tc.bg_h);  // different from fg
    auto bg = mat_to_halide_interleaved(bg_mat);
    auto seg_mask = make_uniform_mask(tc.mask_w, tc.mask_h, (uint8_t)FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.fg_w, tc.fg_h, 3);

    int err = seg_bg_replace(fg, bg, seg_mask, FG_CLASS, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < tc.fg_h; y++)
        for (int x = 0; x < tc.fg_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)fg(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "All-foreground should produce fg_image unchanged (±1)";
}

// ---------------------------------------------------------------------------
// Test 3: All background → output = resized bg_image
// ---------------------------------------------------------------------------
TEST_P(SegBgReplaceTest, AllBackground_OutputEqualsBg) {
    auto tc = GetParam();

    cv::Mat fg_mat = make_test_image_rgb(tc.fg_w, tc.fg_h);
    auto fg = mat_to_halide_interleaved(fg_mat);
    cv::Mat bg_mat(tc.bg_h, tc.bg_w, CV_8UC3, cv::Scalar(100, 150, 200));  // solid color
    auto bg = mat_to_halide_interleaved(bg_mat);
    auto seg_mask = make_uniform_mask(tc.mask_w, tc.mask_h, (uint8_t)255);  // no match

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.fg_w, tc.fg_h, 3);

    int err = seg_bg_replace(fg, bg, seg_mask, FG_CLASS, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0);

    // Alpha = 0 everywhere → output should be the resized bg (solid color)
    // Allow tolerance for bilinear rounding
    int mismatches = 0;
    for (int y = 0; y < tc.fg_h; y++)
        for (int x = 0; x < tc.fg_w; x++) {
            // BG is solid (100,150,200) → after bilinear resize, still solid
            if (std::abs((int)output(x, y, 0) - 100) > 2) mismatches++;
            if (std::abs((int)output(x, y, 1) - 150) > 2) mismatches++;
            if (std::abs((int)output(x, y, 2) - 200) > 2) mismatches++;
        }

    EXPECT_EQ(mismatches, 0) << "All-background with solid BG should produce solid color (±2)";
}

// ---------------------------------------------------------------------------
// Test 4: Timing benchmark — Halide vs OpenCV
// ---------------------------------------------------------------------------
TEST_P(SegBgReplaceTest, TimingBenchmark) {
    auto tc = GetParam();

    cv::Mat fg_mat = make_test_image_rgb(tc.fg_w, tc.fg_h);
    auto fg = mat_to_halide_interleaved(fg_mat);
    cv::Mat bg_mat = make_test_image_rgb(tc.bg_w, tc.bg_h);
    auto bg = mat_to_halide_interleaved(bg_mat);
    auto seg_mask_halide = make_seg_mask_rect(tc.mask_w, tc.mask_h, FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.fg_w, tc.fg_h, 3);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        seg_bg_replace(fg, bg, seg_mask_halide, FG_CLASS, EDGE_SOFTNESS, output);

    // Halide timing
    std::vector<long> halide_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        seg_bg_replace(fg, bg, seg_mask_halide, FG_CLASS, EDGE_SOFTNESS, output);
        auto end = std::chrono::high_resolution_clock::now();
        halide_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    // OpenCV reference
    cv::Mat seg_mask_cv(tc.mask_h, tc.mask_w, CV_8UC1);
    for (int y = 0; y < tc.mask_h; y++)
        for (int x = 0; x < tc.mask_w; x++)
            seg_mask_cv.at<uint8_t>(y, x) = seg_mask_halide(x, y);

    cv::Mat cv_output;
    for (int i = 0; i < NUM_WARMUP; i++)
        opencv_ops::seg_bg_replace(fg_mat, bg_mat, seg_mask_cv, FG_CLASS,
                                   EDGE_SOFTNESS, cv_output);

    std::vector<long> opencv_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        opencv_ops::seg_bg_replace(fg_mat, bg_mat, seg_mask_cv, FG_CLASS,
                                   EDGE_SOFTNESS, cv_output);
        auto end = std::chrono::high_resolution_clock::now();
        opencv_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    std::sort(halide_times.begin(), halide_times.end());
    std::sort(opencv_times.begin(), opencv_times.end());
    long halide_median = halide_times[NUM_ITERS / 2];
    long opencv_median = opencv_times[NUM_ITERS / 2];
    float speedup = (float)opencv_median / std::max(halide_median, 1L);

    printf("  [fg=%dx%d, bg=%dx%d, mask=%dx%d] Halide: %ld us, OpenCV: %ld us, speedup: %.2fx\n",
           tc.fg_w, tc.fg_h, tc.bg_w, tc.bg_h, tc.mask_w, tc.mask_h,
           halide_median, opencv_median, speedup);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    SegBgReplaceTest,
    ::testing::Values(
        BgReplaceTestCase{320, 240, 640, 480, 64, 64},    // bg larger than fg
        BgReplaceTestCase{640, 480, 640, 480, 128, 128},   // bg same size
        BgReplaceTestCase{641, 481, 320, 240, 128, 128},   // odd fg, bg smaller
        BgReplaceTestCase{1280, 720, 1920, 1080, 256, 256},
        BgReplaceTestCase{1279, 719, 640, 480, 256, 256},  // odd resolution
        BgReplaceTestCase{1920, 1080, 1280, 720, 512, 512}
    )
);
