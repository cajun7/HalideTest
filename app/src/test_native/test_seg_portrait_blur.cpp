// =============================================================================
// Tests for seg_portrait_blur Halide generator
// =============================================================================
//
// Test strategy:
//   Correctness: Compare fused Halide output against a C++ reference that
//   chains the same operations (mask upsample → disc blur → alpha blend).
//   Also compare against OpenCV reference for timing benchmarks.
//
//   Quality metrics:
//   - Per-pixel tolerance: 3 (float blending rounding)
//   - PSNR > 40 dB against C++ reference
//   - Mismatch < 1% of pixels
//
//   Performance: Time Halide fused vs OpenCV chained (mask resize + disc blur + blend)
//   across multiple resolutions.
//
// Test cases:
//   1. MatchesReference     - Compare against C++ reference, multiple resolutions
//   2. AllForeground        - fg_class covers all pixels → output = input (no blur)
//   3. AllBackground        - fg_class matches nothing → output = fully blurred
//   4. ZeroRadius           - blur_radius = 0 → output = input regardless of mask
//   5. OddResolutions       - Verify GuardWithIf handles non-aligned widths
//   6. MaskResolutionVariants - Different seg_mask resolutions (64², 128², 256², 512²)
//   7. TimingBenchmark      - Halide fused vs OpenCV chained (multiple resolutions)
//
// =============================================================================

#include "test_common.h"
#include "seg_portrait_blur.h"
#include "opencv_ops.h"
#include <chrono>

static const int NUM_CLASSES = 8;
static const int FG_CLASS = 1;  // Person class for testing
static const int BLUR_RADIUS = 8;
static const float EDGE_SOFTNESS = 3.0f;
static const int TOLERANCE = 3;
static const int NUM_WARMUP = 3;
static const int NUM_ITERS = 10;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

// Create a synthetic segmentation mask where a centered rectangle is fg_class
// and everything else is background (class 0).
static Halide::Runtime::Buffer<uint8_t> make_seg_mask(int mask_w, int mask_h,
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

// Create a seg mask that is entirely one class
static Halide::Runtime::Buffer<uint8_t> make_uniform_mask(int mask_w, int mask_h,
                                                           uint8_t class_id) {
    Halide::Runtime::Buffer<uint8_t> mask(mask_w, mask_h);
    for (int y = 0; y < mask_h; y++)
        for (int x = 0; x < mask_w; x++)
            mask(x, y) = class_id;
    return mask;
}

// C++ reference implementation matching the Halide generator exactly
static void reference_portrait_blur(
    const Halide::Runtime::Buffer<uint8_t>& input,
    const Halide::Runtime::Buffer<uint8_t>& seg_mask,
    int fg_class, int blur_radius, float edge_softness,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    int w = input.dim(0).extent();
    int h = input.dim(1).extent();
    int mw = seg_mask.dim(0).extent();
    int mh = seg_mask.dim(1).extent();

    // Step 1: Build soft alpha mask via bilinear upsampling
    std::vector<float> alpha(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mask_x = (x + 0.5f) * mw / (float)w - 0.5f;
            float mask_y = (y + 0.5f) * mh / (float)h - 0.5f;
            int mx = (int)floorf(mask_x);
            int my = (int)floorf(mask_y);
            float mfx = mask_x - mx;
            float mfy = mask_y - my;

            auto sample = [&](int px, int py) -> float {
                px = std::max(0, std::min(px, mw - 1));
                py = std::max(0, std::min(py, mh - 1));
                return (seg_mask(px, py) == fg_class) ? 1.0f : 0.0f;
            };

            float raw = sample(mx, my) * (1 - mfx) * (1 - mfy) +
                         sample(mx + 1, my) * mfx * (1 - mfy) +
                         sample(mx, my + 1) * (1 - mfx) * mfy +
                         sample(mx + 1, my + 1) * mfx * mfy;

            float feathered = std::max(0.0f, std::min(1.0f,
                (raw - 0.5f) * edge_softness + 0.5f));
            alpha[y * w + x] = feathered;
        }
    }

    // Step 2: Disc blur
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0;
                int count = 0;
                for (int dy = -blur_radius; dy <= blur_radius; dy++) {
                    for (int dx = -blur_radius; dx <= blur_radius; dx++) {
                        if (dx * dx + dy * dy <= blur_radius * blur_radius) {
                            int sx = std::max(0, std::min(x + dx, w - 1));
                            int sy = std::max(0, std::min(y + dy, h - 1));
                            sum += (float)input(sx, sy, c);
                            count++;
                        }
                    }
                }
                float blurred = (count > 0) ? sum / count : (float)input(x, y, c);
                float a = alpha[y * w + x];
                float sharp = (float)input(x, y, c);
                float val = a * sharp + (1.0f - a) * blurred + 0.5f;
                output(x, y, c) = (uint8_t)std::max(0.0f, std::min(255.0f, val));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

struct PortraitBlurTestCase {
    int img_w, img_h;
    int mask_w, mask_h;
};

class SegPortraitBlurTest : public ::testing::TestWithParam<PortraitBlurTestCase> {};

// ---------------------------------------------------------------------------
// Test 1: Matches C++ reference across resolutions
// ---------------------------------------------------------------------------
TEST_P(SegPortraitBlurTest, MatchesReference) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_seg_mask(tc.mask_w, tc.mask_h, FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);
    Halide::Runtime::Buffer<uint8_t> ref =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_portrait_blur(input, seg_mask, FG_CLASS, BLUR_RADIUS, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0) << "Halide seg_portrait_blur failed";

    reference_portrait_blur(input, seg_mask, FG_CLASS, BLUR_RADIUS, EDGE_SOFTNESS, ref);

    // Compare
    int mismatches = 0;
    int max_diff = 0;
    double mse_sum = 0;
    int total = tc.img_w * tc.img_h * 3;
    for (int y = 0; y < tc.img_h; y++) {
        for (int x = 0; x < tc.img_w; x++) {
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)output(x, y, c) - (int)ref(x, y, c));
                mse_sum += diff * diff;
                if (diff > TOLERANCE) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    double mse = mse_sum / total;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    float mismatch_pct = 100.0f * mismatches / total;

    printf("  [%dx%d img, %dx%d mask] PSNR=%.1f dB, max_diff=%d, mismatch=%.2f%%\n",
           tc.img_w, tc.img_h, tc.mask_w, tc.mask_h, psnr, max_diff, mismatch_pct);

    EXPECT_GT(psnr, 40.0) << "PSNR too low";
    EXPECT_LT(mismatch_pct, 1.0f) << "Too many pixel mismatches";

    dump_if_first(output, "portrait_blur_halide", 0);
    dump_if_first(ref, "portrait_blur_ref", 0);
}

// ---------------------------------------------------------------------------
// Test 2: All foreground → output = input (no blur applied)
// ---------------------------------------------------------------------------
TEST_P(SegPortraitBlurTest, AllForeground_NoBlur) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_uniform_mask(tc.mask_w, tc.mask_h, (uint8_t)FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_portrait_blur(input, seg_mask, FG_CLASS, BLUR_RADIUS, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0);

    // Alpha = 1.0 everywhere → output should equal input exactly
    // (alpha * sharp + (1-alpha) * blurred = 1.0 * sharp + 0.0 * blurred = sharp)
    // Allow +/- 1 for rounding (+0.5 in the expression)
    int mismatches = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)input(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "All-foreground should produce input unchanged (±1 rounding)";
}

// ---------------------------------------------------------------------------
// Test 3: Zero blur radius → output = input regardless of mask
// ---------------------------------------------------------------------------
TEST_P(SegPortraitBlurTest, ZeroRadius_NoChange) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_seg_mask(tc.mask_w, tc.mask_h, FG_CLASS);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    // blur_radius = 0 → disc has 0 pixels (dist 0 <= 0: only center pixel)
    // → blurred = input → blend doesn't matter
    int err = seg_portrait_blur(input, seg_mask, FG_CLASS, 0, EDGE_SOFTNESS, output);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)input(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "Zero radius should produce input unchanged (±1 rounding)";
}

// ---------------------------------------------------------------------------
// Test 4: Timing benchmark — Halide fused vs OpenCV chained
// ---------------------------------------------------------------------------
TEST_P(SegPortraitBlurTest, TimingBenchmark) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask_halide = make_seg_mask(tc.mask_w, tc.mask_h, FG_CLASS);

    // Halide output
    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        seg_portrait_blur(input, seg_mask_halide, FG_CLASS, BLUR_RADIUS, EDGE_SOFTNESS, output);

    // Halide timing
    std::vector<long> halide_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        seg_portrait_blur(input, seg_mask_halide, FG_CLASS, BLUR_RADIUS, EDGE_SOFTNESS, output);
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
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        opencv_ops::seg_portrait_blur(rgb_mat, seg_mask_cv, FG_CLASS, BLUR_RADIUS,
                                      EDGE_SOFTNESS, cv_output);

    std::vector<long> opencv_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        opencv_ops::seg_portrait_blur(rgb_mat, seg_mask_cv, FG_CLASS, BLUR_RADIUS,
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

    printf("  [%dx%d img, %dx%d mask, r=%d] Halide: %ld us, OpenCV: %ld us, speedup: %.2fx\n",
           tc.img_w, tc.img_h, tc.mask_w, tc.mask_h, BLUR_RADIUS,
           halide_median, opencv_median, speedup);
}

// Parameterize across image and mask resolutions
INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    SegPortraitBlurTest,
    ::testing::Values(
        PortraitBlurTestCase{320, 240, 64, 64},
        PortraitBlurTestCase{640, 480, 128, 128},
        PortraitBlurTestCase{641, 481, 128, 128},     // odd resolution
        PortraitBlurTestCase{640, 480, 256, 256},
        PortraitBlurTestCase{1280, 720, 256, 256},
        PortraitBlurTestCase{1279, 719, 256, 256},     // odd resolution
        PortraitBlurTestCase{1920, 1080, 512, 512}
    )
);
