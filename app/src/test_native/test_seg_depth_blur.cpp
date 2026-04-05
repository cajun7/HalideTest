// =============================================================================
// Tests for seg_depth_blur Halide generator
// =============================================================================
//
// Test strategy:
//   Correctness: Compare fused Halide output against a C++ reference that
//   chains the same operations (depth upsample → per-layer disc blur → select).
//
//   Quality metrics:
//   - Per-pixel tolerance: 3 (float blending rounding)
//   - PSNR > 40 dB against C++ reference
//   - Mismatch < 1% of pixels
//
//   Edge cases:
//   - Uniform depth (single layer): should match standalone disc blur
//   - Zero-radius layers: output should equal input
//   - Odd resolutions: verify GuardWithIf handles non-aligned widths
//
//   Performance: Halide fused vs OpenCV chained across resolutions.
//
// =============================================================================

#include "test_common.h"
#include "seg_depth_blur.h"
#include "opencv_ops.h"
#include <chrono>

static const int TOLERANCE = 3;
static const int NUM_WARMUP = 3;
static const int NUM_ITERS = 10;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

// Create a depth map with horizontal gradient (left=near, right=far)
static Halide::Runtime::Buffer<uint8_t> make_gradient_depth_map(int w, int h) {
    Halide::Runtime::Buffer<uint8_t> depth(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            depth(x, y) = (uint8_t)(x * 255 / std::max(w - 1, 1));
    return depth;
}

// Create a uniform depth map (all pixels at same depth)
static Halide::Runtime::Buffer<uint8_t> make_uniform_depth_map(int w, int h, uint8_t value) {
    Halide::Runtime::Buffer<uint8_t> depth(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            depth(x, y) = value;
    return depth;
}

// Create a step-function depth map (left half = near, right half = far)
static Halide::Runtime::Buffer<uint8_t> make_step_depth_map(int w, int h) {
    Halide::Runtime::Buffer<uint8_t> depth(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            depth(x, y) = (x < w / 2) ? (uint8_t)0 : (uint8_t)255;
    return depth;
}

// Create kernel config buffer: each row is [min_depth_norm, max_depth_norm, blur_radius]
static Halide::Runtime::Buffer<float> make_kernel_config(
    const std::vector<std::array<float, 3>>& zones) {
    // Pad to max_layers (5) rows even if fewer zones
    int rows = 5;
    Halide::Runtime::Buffer<float> config(rows, 3);
    for (int k = 0; k < rows; k++) {
        if (k < (int)zones.size()) {
            config(k, 0) = zones[k][0];  // min_depth
            config(k, 1) = zones[k][1];  // max_depth
            config(k, 2) = zones[k][2];  // blur_radius
        } else {
            config(k, 0) = 0.0f;
            config(k, 1) = 0.0f;
            config(k, 2) = 0.0f;
        }
    }
    return config;
}

// C++ reference implementation matching the Halide generator exactly
static void reference_depth_blur(
    const Halide::Runtime::Buffer<uint8_t>& input,
    const Halide::Runtime::Buffer<uint8_t>& depth_map,
    const Halide::Runtime::Buffer<float>& kernel_config,
    int num_kernels,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    int w = input.dim(0).extent();
    int h = input.dim(1).extent();
    int dw = depth_map.dim(0).extent();
    int dh = depth_map.dim(1).extent();

    // Step 1: Bilinear upsample depth map
    std::vector<float> depth_up(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float map_x = (x + 0.5f) * dw / (float)w - 0.5f;
            float map_y = (y + 0.5f) * dh / (float)h - 0.5f;
            int dxi = (int)floorf(map_x);
            int dyi = (int)floorf(map_y);
            float dfx = map_x - dxi;
            float dfy = map_y - dyi;

            auto sample = [&](int px, int py) -> float {
                px = std::max(0, std::min(px, dw - 1));
                py = std::max(0, std::min(py, dh - 1));
                return depth_map(px, py) / 255.0f;
            };

            depth_up[y * w + x] = sample(dxi, dyi) * (1 - dfx) * (1 - dfy)
                                 + sample(dxi + 1, dyi) * dfx * (1 - dfy)
                                 + sample(dxi, dyi + 1) * (1 - dfx) * dfy
                                 + sample(dxi + 1, dyi + 1) * dfx * dfy;
        }
    }

    // Step 2: Pre-blur at each layer's radius using disc kernel with repeat_edge
    struct LayerBlur {
        std::vector<float> data; // w * h * 3
    };
    std::vector<LayerBlur> layers(num_kernels);

    for (int k = 0; k < num_kernels; k++) {
        int radius = (int)kernel_config(k, 2);
        layers[k].data.resize(w * h * 3);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    float sum = 0;
                    int count = 0;
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            if (dx * dx + dy * dy <= radius * radius) {
                                int sx = std::max(0, std::min(x + dx, w - 1));
                                int sy = std::max(0, std::min(y + dy, h - 1));
                                sum += (float)input(sx, sy, c);
                                count++;
                            }
                        }
                    }
                    layers[k].data[(y * w + x) * 3 + c] =
                        (count > 0) ? sum / count : (float)input(x, y, c);
                }
            }
        }
    }

    // Step 3: Per-pixel depth-based layer selection
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float d = depth_up[y * w + x];
            int selected = -1;
            for (int k = 0; k < num_kernels; k++) {
                float min_d = kernel_config(k, 0);
                float max_d = kernel_config(k, 1);
                if (d >= min_d && d <= max_d)
                    selected = k;
            }
            for (int c = 0; c < 3; c++) {
                float val;
                if (selected >= 0)
                    val = layers[selected].data[(y * w + x) * 3 + c];
                else
                    val = (float)input(x, y, c);
                output(x, y, c) = (uint8_t)std::max(0.0f, std::min(255.0f, val + 0.5f));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

struct DepthBlurTestCase {
    int img_w, img_h;
    int depth_w, depth_h;
};

class SegDepthBlurTest : public ::testing::TestWithParam<DepthBlurTestCase> {};

// Standard 3-zone kernel config: near (sharp), mid (light blur), far (heavy blur)
static std::vector<std::array<float, 3>> standard_zones() {
    return {
        {0.0f,  0.33f, 0.0f},   // near: no blur
        {0.33f, 0.66f, 4.0f},   // mid: radius 4
        {0.66f, 1.0f,  8.0f},   // far: radius 8
    };
}

// ---------------------------------------------------------------------------
// Test 1: Matches C++ reference across resolutions
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, MatchesReference) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto depth = make_gradient_depth_map(tc.depth_w, tc.depth_h);
    auto zones = standard_zones();
    auto config = make_kernel_config(zones);
    int nk = (int)zones.size();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);
    Halide::Runtime::Buffer<uint8_t> ref =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_depth_blur(input, depth, config, nk, output);
    ASSERT_EQ(err, 0) << "Halide seg_depth_blur failed";

    reference_depth_blur(input, depth, config, nk, ref);

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

    printf("  [%dx%d img, %dx%d depth] PSNR=%.1f dB, max_diff=%d, mismatch=%.2f%%\n",
           tc.img_w, tc.img_h, tc.depth_w, tc.depth_h, psnr, max_diff, mismatch_pct);

    EXPECT_GT(psnr, 40.0) << "PSNR too low";
    EXPECT_LT(mismatch_pct, 1.0f) << "Too many pixel mismatches";

    dump_if_first(output, "depth_blur_halide", 0);
    dump_if_first(ref, "depth_blur_ref", 0);
}

// ---------------------------------------------------------------------------
// Test 2: Uniform depth, single layer — output = disc blur at that radius
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, UniformDepth_SingleLayer) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    // All pixels at depth 128 (0.502 normalized)
    auto depth = make_uniform_depth_map(tc.depth_w, tc.depth_h, 128);
    // Single kernel covering full range with radius 4
    auto config = make_kernel_config({{0.0f, 1.0f, 4.0f}});

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);
    Halide::Runtime::Buffer<uint8_t> ref =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_depth_blur(input, depth, config, 1, output);
    ASSERT_EQ(err, 0);

    reference_depth_blur(input, depth, config, 1, ref);

    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++) {
                int diff = std::abs((int)output(x, y, c) - (int)ref(x, y, c));
                if (diff > TOLERANCE) mismatches++;
                max_diff = std::max(max_diff, diff);
            }

    float mismatch_pct = 100.0f * mismatches / (tc.img_w * tc.img_h * 3);
    printf("  [%dx%d uniform depth, r=4] max_diff=%d, mismatch=%.2f%%\n",
           tc.img_w, tc.img_h, max_diff, mismatch_pct);
    EXPECT_LT(mismatch_pct, 1.0f);
}

// ---------------------------------------------------------------------------
// Test 3: Foreground sharp — depth=0 with radius=0 → output = input
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, ForegroundSharp) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto depth = make_uniform_depth_map(tc.depth_w, tc.depth_h, 0);
    // Near depth zone: radius 0 (sharp)
    auto config = make_kernel_config({{0.0f, 0.1f, 0.0f}});

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_depth_blur(input, depth, config, 1, output);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)input(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "Foreground sharp (r=0) should produce input unchanged (±1 rounding)";
}

// ---------------------------------------------------------------------------
// Test 4: All zero radius → output = input regardless of depth
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, ZeroRadius_AllLayers) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto depth = make_gradient_depth_map(tc.depth_w, tc.depth_h);
    auto config = make_kernel_config({
        {0.0f,  0.5f, 0.0f},
        {0.5f,  1.0f, 0.0f},
    });

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_depth_blur(input, depth, config, 2, output);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < tc.img_h; y++)
        for (int x = 0; x < tc.img_w; x++)
            for (int c = 0; c < 3; c++)
                if (std::abs((int)output(x, y, c) - (int)input(x, y, c)) > 1)
                    mismatches++;

    EXPECT_EQ(mismatches, 0) << "All-zero radius should produce input unchanged (±1 rounding)";
}

// ---------------------------------------------------------------------------
// Test 5: Multiple depth zones — verify distinct blur levels
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, MultipleDepthZones) {
    auto tc = GetParam();
    if (tc.img_w < 640) return;  // Need enough resolution for zone distinction

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto depth = make_step_depth_map(tc.depth_w, tc.depth_h);
    auto zones = standard_zones();
    auto config = make_kernel_config(zones);

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    int err = seg_depth_blur(input, depth, config, (int)zones.size(), output);
    ASSERT_EQ(err, 0);

    // Left quarter (near, depth ~0, r=0) should be close to input
    // Right quarter (far, depth ~1, r=8) should differ significantly
    double near_diff_sum = 0, far_diff_sum = 0;
    int near_count = 0, far_count = 0;
    int quarter_x = tc.img_w / 4;
    int three_quarter_x = 3 * tc.img_w / 4;

    for (int y = 0; y < tc.img_h; y++) {
        for (int c = 0; c < 3; c++) {
            // Sample well inside the near zone
            int diff_near = std::abs((int)output(quarter_x, y, c) - (int)input(quarter_x, y, c));
            near_diff_sum += diff_near;
            near_count++;

            // Sample well inside the far zone
            int diff_far = std::abs((int)output(three_quarter_x, y, c) - (int)input(three_quarter_x, y, c));
            far_diff_sum += diff_far;
            far_count++;
        }
    }

    double near_avg = near_diff_sum / near_count;
    double far_avg = far_diff_sum / far_count;

    printf("  [%dx%d step depth] near_avg_diff=%.2f, far_avg_diff=%.2f\n",
           tc.img_w, tc.img_h, near_avg, far_avg);

    // Near zone should be close to original (small diff)
    // Far zone should have significant blur (larger diff)
    EXPECT_LT(near_avg, 2.0) << "Near zone should be nearly sharp";
    EXPECT_GT(far_avg, near_avg) << "Far zone should be blurrier than near zone";
}

// ---------------------------------------------------------------------------
// Test 6: Timing benchmark — Halide fused vs OpenCV chained
// ---------------------------------------------------------------------------
TEST_P(SegDepthBlurTest, TimingBenchmark) {
    auto tc = GetParam();

    cv::Mat rgb_mat = make_test_image_rgb(tc.img_w, tc.img_h);
    auto input = mat_to_halide_interleaved(rgb_mat);
    auto depth = make_gradient_depth_map(tc.depth_w, tc.depth_h);
    auto zones = standard_zones();
    auto config = make_kernel_config(zones);
    int nk = (int)zones.size();

    Halide::Runtime::Buffer<uint8_t> output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(tc.img_w, tc.img_h, 3);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        seg_depth_blur(input, depth, config, nk, output);

    // Halide timing
    std::vector<long> halide_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        seg_depth_blur(input, depth, config, nk, output);
        auto end = std::chrono::high_resolution_clock::now();
        halide_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    // OpenCV reference
    cv::Mat depth_cv(tc.depth_h, tc.depth_w, CV_8UC1);
    for (int y = 0; y < tc.depth_h; y++)
        for (int x = 0; x < tc.depth_w; x++)
            depth_cv.at<uint8_t>(y, x) = depth(x, y);

    std::vector<float> config_vec;
    for (int k = 0; k < nk; k++) {
        config_vec.push_back(zones[k][0]);
        config_vec.push_back(zones[k][1]);
        config_vec.push_back(zones[k][2]);
    }

    cv::Mat cv_output;
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++)
        opencv_ops::seg_depth_blur(rgb_mat, depth_cv, config_vec, nk, cv_output);

    std::vector<long> opencv_times;
    for (int i = 0; i < NUM_ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        opencv_ops::seg_depth_blur(rgb_mat, depth_cv, config_vec, nk, cv_output);
        auto end = std::chrono::high_resolution_clock::now();
        opencv_times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    std::sort(halide_times.begin(), halide_times.end());
    std::sort(opencv_times.begin(), opencv_times.end());
    long halide_median = halide_times[NUM_ITERS / 2];
    long opencv_median = opencv_times[NUM_ITERS / 2];
    float speedup = (float)opencv_median / std::max(halide_median, 1L);

    printf("  [%dx%d img, %dx%d depth, %d zones] Halide: %ld us, OpenCV: %ld us, speedup: %.2fx\n",
           tc.img_w, tc.img_h, tc.depth_w, tc.depth_h, nk,
           halide_median, opencv_median, speedup);
}

// Parameterize across image and depth map resolutions
INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    SegDepthBlurTest,
    ::testing::Values(
        DepthBlurTestCase{320, 240, 64, 64},
        DepthBlurTestCase{640, 480, 128, 128},
        DepthBlurTestCase{641, 481, 128, 128},     // odd resolution
        DepthBlurTestCase{640, 480, 256, 256},
        DepthBlurTestCase{1280, 720, 256, 256},
        DepthBlurTestCase{1279, 719, 256, 256},     // odd resolution
        DepthBlurTestCase{1920, 1080, 512, 512}
    )
);
