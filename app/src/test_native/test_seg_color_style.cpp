// =============================================================================
// Tests for seg_color_style Halide generator
// =============================================================================
//
// Test strategy:
//   Correctness: Compare fused Halide output against a C++ reference that
//   chains mask nearest-neighbor upsample → LUT lookup → color grade → blend.
//
//   Quality metrics:
//   - Per-pixel tolerance: 2 (nearest-neighbor + float rounding)
//   - PSNR > 50 dB against C++ reference (very lightweight pipeline)
//   - Mismatch < 1% of pixels
//
//   Performance: Time Halide fused vs OpenCV chained across resolutions.
//   Expected ~4-5x speedup (lightest of the three seg pipelines).
//
// Test cases:
//   1. MatchesReference       - Compare against C++ reference
//   2. IdentityLUT            - All classes use identity transform → output = input
//   3. SingleClassDominant    - One class covers all, verify its transform applied
//   4. VisualizationMode      - Each class mapped to distinct color
//   5. OddResolutions         - Edge case verification
//   6. TimingBenchmark         - Halide vs OpenCV
//
// =============================================================================

#include "test_common.h"
#include "seg_color_style.h"
#include "opencv_ops.h"
#include <chrono>

static const int NUM_CLASSES = 8;
static const int TOLERANCE = 2;
static const int NUM_WARMUP = 3;
static const int NUM_ITERS = 10;

// ---------------------------------------------------------------------------
// LUT helpers
// ---------------------------------------------------------------------------

// Create identity LUT: gain=1, bias=0, alpha=1 for all classes
static Halide::Runtime::Buffer<float> make_identity_lut() {
    Halide::Runtime::Buffer<float> lut(NUM_CLASSES, 7);
    for (int c = 0; c < NUM_CLASSES; c++) {
        lut(c, 0) = 1.0f; lut(c, 1) = 1.0f; lut(c, 2) = 1.0f;  // R,G,B gain
        lut(c, 3) = 0.0f; lut(c, 4) = 0.0f; lut(c, 5) = 0.0f;  // R,G,B bias
        lut(c, 6) = 1.0f;  // blend_alpha
    }
    return lut;
}

// Create a stylized LUT with different transforms per class
static Halide::Runtime::Buffer<float> make_styled_lut() {
    Halide::Runtime::Buffer<float> lut(NUM_CLASSES, 7);
    // Class 0 (background): desaturate (darken by 70%)
    lut(0, 0) = 0.3f; lut(0, 1) = 0.3f; lut(0, 2) = 0.3f;
    lut(0, 3) = 0.0f; lut(0, 4) = 0.0f; lut(0, 5) = 0.0f;
    lut(0, 6) = 0.8f;
    // Class 1 (person): unchanged
    lut(1, 0) = 1.0f; lut(1, 1) = 1.0f; lut(1, 2) = 1.0f;
    lut(1, 3) = 0.0f; lut(1, 4) = 0.0f; lut(1, 5) = 0.0f;
    lut(1, 6) = 1.0f;
    // Class 2 (sky): enhance blue
    lut(2, 0) = 0.8f; lut(2, 1) = 0.9f; lut(2, 2) = 1.2f;
    lut(2, 3) = 0.0f; lut(2, 4) = 0.0f; lut(2, 5) = 20.0f;
    lut(2, 6) = 0.9f;
    // Class 3 (vegetation): enhance green
    lut(3, 0) = 0.9f; lut(3, 1) = 1.1f; lut(3, 2) = 0.8f;
    lut(3, 3) = 0.0f; lut(3, 4) = 10.0f; lut(3, 5) = 0.0f;
    lut(3, 6) = 0.9f;
    // Classes 4-7: moderate adjustments
    for (int c = 4; c < NUM_CLASSES; c++) {
        lut(c, 0) = 0.9f + (c % 3) * 0.1f;
        lut(c, 1) = 1.0f;
        lut(c, 2) = 0.9f + ((c + 1) % 3) * 0.1f;
        lut(c, 3) = (float)(c * 3); lut(c, 4) = 0.0f; lut(c, 5) = (float)(c * 2);
        lut(c, 6) = 0.7f;
    }
    return lut;
}

// Create visualization LUT: each class gets a distinct bright color
static Halide::Runtime::Buffer<float> make_visualization_lut() {
    Halide::Runtime::Buffer<float> lut(NUM_CLASSES, 7);
    // Distinct colors for visualization
    float colors[][3] = {
        {0, 0, 0},         // Class 0: black (background)
        {255, 0, 0},       // Class 1: red (person)
        {0, 0, 255},       // Class 2: blue (sky)
        {0, 200, 0},       // Class 3: green (vegetation)
        {255, 255, 0},     // Class 4: yellow
        {255, 0, 255},     // Class 5: magenta
        {0, 255, 255},     // Class 6: cyan
        {200, 200, 200},   // Class 7: gray
    };
    for (int c = 0; c < NUM_CLASSES; c++) {
        lut(c, 0) = 0.0f; lut(c, 1) = 0.0f; lut(c, 2) = 0.0f;  // zero gain
        lut(c, 3) = colors[c][0]; lut(c, 4) = colors[c][1]; lut(c, 5) = colors[c][2];
        lut(c, 6) = 0.5f;  // 50% overlay
    }
    return lut;
}

// ---------------------------------------------------------------------------
// Mask generators
// ---------------------------------------------------------------------------

// Stripe pattern: alternating classes horizontally
static Halide::Runtime::Buffer<uint8_t> make_striped_mask(int mask_w, int mask_h,
                                                           int num_classes) {
    Halide::Runtime::Buffer<uint8_t> mask(mask_w, mask_h);
    int stripe_w = std::max(1, mask_w / num_classes);
    for (int y = 0; y < mask_h; y++)
        for (int x = 0; x < mask_w; x++)
            mask(x, y) = (uint8_t)std::min(x / stripe_w, num_classes - 1);
    return mask;
}

static Halide::Runtime::Buffer<uint8_t> make_uniform_mask(int w, int h, uint8_t cls) {
    Halide::Runtime::Buffer<uint8_t> mask(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            mask(x, y) = cls;
    return mask;
}

// C++ reference implementation
static void reference_color_style(
    const Halide::Runtime::Buffer<uint8_t>& input,
    const Halide::Runtime::Buffer<uint8_t>& seg_mask,
    const Halide::Runtime::Buffer<float>& lut,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    int w = input.dim(0).extent();
    int h = input.dim(1).extent();
    int mw = seg_mask.dim(0).extent();
    int mh = seg_mask.dim(1).extent();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Nearest-neighbor upsample
            int mx = (int)std::max(0.0f, std::min((float)(mw - 1),
                (x + 0.5f) * mw / (float)w - 0.5f + 0.5f));
            int my = (int)std::max(0.0f, std::min((float)(mh - 1),
                (y + 0.5f) * mh / (float)h - 0.5f + 0.5f));

            int cls = seg_mask(std::max(0, std::min(mx, mw - 1)),
                               std::max(0, std::min(my, mh - 1)));

            float blend_alpha = lut(cls, 6);

            for (int c = 0; c < 3; c++) {
                float orig = (float)input(x, y, c);
                float gain = lut(cls, c);
                float bias = lut(cls, c + 3);
                float styled = std::max(0.0f, std::min(255.0f, orig * gain + bias));
                float blended = blend_alpha * styled + (1.0f - blend_alpha) * orig + 0.5f;
                output(x, y, c) = (uint8_t)std::max(0.0f, std::min(255.0f, blended));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

struct ColorStyleTestCase {
    int img_w, img_h;
    int mask_w, mask_h;
};

class SegColorStyleTest : public ::testing::TestWithParam<ColorStyleTestCase> {};

// ---------------------------------------------------------------------------
// Test 1: Matches reference with styled LUT
// ---------------------------------------------------------------------------
TEST_P(SegColorStyleTest, MatchesReference) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_striped_mask(tc.mask_w, tc.mask_h, NUM_CLASSES);
    auto lut = make_styled_lut();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);
    Halide::Runtime::Buffer<uint8_t> ref =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_color_style(input, seg_mask, lut, output);
    ASSERT_EQ(err, 0) << "Halide seg_color_style failed";

    reference_color_style(input, seg_mask, lut, ref);

    int mismatches = 0;
    int max_diff = 0;
    double mse_sum = 0;
    int total = tc.img_w * tc.img_h * 3;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)output(x, y, c) - (int)ref(x, y, c));
                mse_sum += diff * diff;
                if (diff > TOLERANCE) mismatches++;
                max_diff = std::max(max_diff, diff);
            }

    double mse = mse_sum / total;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    float mismatch_pct = 100.0f * mismatches / total;

    printf("  [%dx%d img, %dx%d mask] PSNR=%.1f dB, max_diff=%d, mismatch=%.2f%%\n",
           tc.img_w, tc.img_h, tc.mask_w, tc.mask_h, psnr, max_diff, mismatch_pct);

    EXPECT_GT(psnr, 50.0) << "PSNR too low";
    EXPECT_LT(mismatch_pct, 1.0f) << "Too many pixel mismatches";
}

// ---------------------------------------------------------------------------
// Test 2: Identity LUT → output = input exactly
// ---------------------------------------------------------------------------
TEST_P(SegColorStyleTest, IdentityLUT_OutputEqualsInput) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_striped_mask(tc.mask_w, tc.mask_h, NUM_CLASSES);
    auto lut = make_identity_lut();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_color_style(input, seg_mask, lut, output);
    ASSERT_EQ(err, 0);

    // Identity: gain=1, bias=0, alpha=1 → styled = input, blend = input
    // Allow ±1 for rounding (+0.5f)
    int mismatches = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)input(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "Identity LUT should produce input unchanged (±1)";
}

// ---------------------------------------------------------------------------
// Test 3: Single class uniform → verify that class's transform applied
// ---------------------------------------------------------------------------
TEST_P(SegColorStyleTest, SingleClassTransform) {
    auto tc = GetParam();

    // Use a solid-color input for easy verification
    cv::Mat rgb_mat(tc.img_h, tc.img_w, CV_8UC3, cv::Scalar(100, 100, 100));
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask = make_uniform_mask(tc.mask_w, tc.mask_h, 2);  // class 2 everywhere
    auto lut = make_styled_lut();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_color_style(input, seg_mask, lut, output);
    ASSERT_EQ(err, 0);

    // Class 2: R_gain=0.8, G_gain=0.9, B_gain=1.2, R_bias=0, G_bias=0, B_bias=20, alpha=0.9
    // styled_r = clamp(100 * 0.8 + 0, 0, 255) = 80
    // styled_g = clamp(100 * 0.9 + 0, 0, 255) = 90
    // styled_b = clamp(100 * 1.2 + 20, 0, 255) = 140
    // blended_r = 0.9 * 80 + 0.1 * 100 + 0.5 = 82.5 → 82 or 83
    // blended_g = 0.9 * 90 + 0.1 * 100 + 0.5 = 91.5 → 91 or 92
    // blended_b = 0.9 * 140 + 0.1 * 100 + 0.5 = 136.5 → 136 or 137
    uint8_t expected_r = 82;  // floor of 82.5
    uint8_t expected_g = 91;  // floor of 91.5
    uint8_t expected_b = 136; // floor of 136.5

    // Check center pixel (avoid potential edge effects)
    int cx = tc.img_w / 2;
    int cy = tc.img_h / 2;
    EXPECT_NEAR((int)output(cx, cy, 0), (int)expected_r, 2)
        << "R channel mismatch for class 2 transform";
    EXPECT_NEAR((int)output(cx, cy, 1), (int)expected_g, 2)
        << "G channel mismatch for class 2 transform";
    EXPECT_NEAR((int)output(cx, cy, 2), (int)expected_b, 2)
        << "B channel mismatch for class 2 transform";
}

// ---------------------------------------------------------------------------
// Test 4: Timing benchmark — Halide vs OpenCV
// ---------------------------------------------------------------------------
TEST_P(SegColorStyleTest, TimingBenchmark) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto seg_mask_halide = make_striped_mask(tc.mask_w, tc.mask_h, NUM_CLASSES);
    auto lut = make_styled_lut();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        seg_color_style(input, seg_mask_halide, lut, output);

    // Halide timing
    std::vector<long> halide_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        seg_color_style(input, seg_mask_halide, lut, output);
        auto end = std::chrono::high_resolution_clock::now();
        halide_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    // OpenCV reference
    cv::Mat seg_mask_cv(tc.mask_h, tc.mask_w, CV_8UC1);
    for (int y = 0; y < tc.mask_h; y++)
        for (int x = 0; x < tc.mask_w; x++)
            seg_mask_cv.at<uint8_t>(y, x) = seg_mask_halide(x, y);

    // Flatten LUT for OpenCV reference
    std::vector<float> lut_vec(NUM_CLASSES * 7);
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int p = 0; p < 7; p++)
            lut_vec[c * 7 + p] = lut(c, p);

    cv::Mat cv_output;
    for (int i = 0; i < NUM_WARMUP; i++)
        opencv_ops::seg_color_style(rgb_mat, seg_mask_cv, lut_vec, NUM_CLASSES, cv_output);

    std::vector<long> opencv_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        opencv_ops::seg_color_style(rgb_mat, seg_mask_cv, lut_vec, NUM_CLASSES, cv_output);
        auto end = std::chrono::high_resolution_clock::now();
        opencv_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    std::sort(halide_times.begin(), halide_times.end());
    std::sort(opencv_times.begin(), opencv_times.end());
    long halide_median = halide_times[NUM_ITERS / 2];
    long opencv_median = opencv_times[NUM_ITERS / 2];
    float speedup = (float)opencv_median / std::max(halide_median, 1L);

    printf("  [%dx%d img, %dx%d mask] Halide: %ld us, OpenCV: %ld us, speedup: %.2fx\n",
           tc.img_w, tc.img_h, tc.mask_w, tc.mask_h,
           halide_median, opencv_median, speedup);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    SegColorStyleTest,
    ::testing::Values(
        ColorStyleTestCase{320, 240, 64, 64},
        ColorStyleTestCase{640, 480, 128, 128},
        ColorStyleTestCase{641, 481, 128, 128},     // odd resolution
        ColorStyleTestCase{640, 480, 256, 256},
        ColorStyleTestCase{1280, 720, 256, 256},
        ColorStyleTestCase{1279, 719, 256, 256},     // odd resolution
        ColorStyleTestCase{1920, 1080, 512, 512}
    )
);
