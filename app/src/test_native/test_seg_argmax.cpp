#include "test_common.h"
#include "seg_argmax.h"

#include <cstdlib>
#include <cmath>

static const int NUM_CLASSES = 8;

// Create planar float32 buffer with random logits
static Halide::Runtime::Buffer<float> make_random_logits(int width, int height, unsigned seed) {
    Halide::Runtime::Buffer<float> buf(width, height, NUM_CLASSES);
    std::srand(seed);
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                buf(x, y, c) = (float)(std::rand() % 10000) / 1000.0f - 5.0f;
    return buf;
}

// C++ reference argmax (exact same tie-breaking as Halide: strict >, first class wins)
static Halide::Runtime::Buffer<uint8_t> reference_argmax(
    const Halide::Runtime::Buffer<float>& input, int num_classes) {
    int w = input.dim(0).extent();
    int h = input.dim(1).extent();
    Halide::Runtime::Buffer<uint8_t> result(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float max_val = input(x, y, 0);
            uint8_t max_idx = 0;
            for (int c = 1; c < num_classes; c++) {
                if (input(x, y, c) > max_val) {
                    max_val = input(x, y, c);
                    max_idx = static_cast<uint8_t>(c);
                }
            }
            result(x, y) = max_idx;
        }
    }
    return result;
}

class SegArgmaxTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// Random logits: Halide output must exactly match C++ reference
TEST_P(SegArgmaxTest, Argmax_MatchesReference) {
    auto [width, height] = GetParam();

    auto input_buf = make_random_logits(width, height, 42);
    auto ref = reference_argmax(input_buf, NUM_CLASSES);
    Halide::Runtime::Buffer<uint8_t> output_buf(width, height);

    int err = seg_argmax(input_buf, output_buf);
    ASSERT_EQ(err, 0) << "Halide seg_argmax failed with error " << err;

    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (output_buf(x, y) != ref(x, y)) {
                mismatches++;
            }
        }
    }
    EXPECT_EQ(mismatches, 0)
        << "Argmax should be exact (no rounding), got " << mismatches << " mismatches";
}

// Deterministic pattern: channel (x+y)%8 has the highest value
TEST_P(SegArgmaxTest, Argmax_KnownPattern) {
    auto [width, height] = GetParam();

    Halide::Runtime::Buffer<float> input_buf(width, height, NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                input_buf(x, y, c) = (c == (x + y) % NUM_CLASSES) ? 10.0f : -1.0f;

    Halide::Runtime::Buffer<uint8_t> output_buf(width, height);
    int err = seg_argmax(input_buf, output_buf);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t expected = static_cast<uint8_t>((x + y) % NUM_CLASSES);
            if (output_buf(x, y) != expected) {
                mismatches++;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Known pattern mismatch";
}

// All channels equal: class 0 wins (strict > tie-breaking)
TEST_P(SegArgmaxTest, Argmax_UniformInput) {
    auto [width, height] = GetParam();

    Halide::Runtime::Buffer<float> input_buf(width, height, NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                input_buf(x, y, c) = 1.0f;

    Halide::Runtime::Buffer<uint8_t> output_buf(width, height);
    int err = seg_argmax(input_buf, output_buf);
    ASSERT_EQ(err, 0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            ASSERT_EQ(output_buf(x, y), 0)
                << "Uniform input should yield class 0 at (" << x << "," << y << ")";
        }
    }
}

// Single dominant channel: sweep each class
TEST_P(SegArgmaxTest, Argmax_SingleDominantChannel) {
    auto [width, height] = GetParam();

    for (int dominant = 0; dominant < NUM_CLASSES; dominant++) {
        Halide::Runtime::Buffer<float> input_buf(width, height, NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    input_buf(x, y, c) = (c == dominant) ? 100.0f : 0.0f;

        Halide::Runtime::Buffer<uint8_t> output_buf(width, height);
        int err = seg_argmax(input_buf, output_buf);
        ASSERT_EQ(err, 0) << "Failed for dominant class " << dominant;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                ASSERT_EQ(output_buf(x, y), dominant)
                    << "Expected class " << dominant << " at (" << x << "," << y << ")";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    SegArgmaxTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),
        std::make_pair(1920, 1080)
    )
);
