// =============================================================================
// Equivalence tests for rotate_fixed_1c_{90cw,180,270cw}.
//
// 1-channel planar rotation is a pure index remap — no interpolation, no
// arithmetic, so output must be bit-exact vs a trivial scalar oracle. No
// tolerance margin.
// =============================================================================

#include "test_common.h"
#include "rotate_fixed_1c_90cw.h"
#include "rotate_fixed_1c_180.h"
#include "rotate_fixed_1c_270cw.h"

#include <cstdio>

namespace {

enum class Rot { CW90, ROT180, CW270 };

// Scalar oracle — mirrors the formulas in rotate_generator.cpp:
//   90 CW:  output(x, y) = input(y, H-1-x)     (output dims H x W)
//   180:    output(x, y) = input(W-1-x, H-1-y) (output dims W x H)
//   270 CW: output(x, y) = input(W-1-y, x)     (output dims H x W)
void scalar_rotate_1c(const uint8_t* src, int W, int H, Rot rot,
                      uint8_t* dst, int& out_w, int& out_h) {
    switch (rot) {
    case Rot::CW90:
        out_w = H; out_h = W;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                // Point (x, y) in src maps to (H-1-y, x) in 90-CW dst.
                dst[x * out_w + (H - 1 - y)] = src[y * W + x];
            }
        break;
    case Rot::ROT180:
        out_w = W; out_h = H;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                dst[(H - 1 - y) * out_w + (W - 1 - x)] = src[y * W + x];
        break;
    case Rot::CW270:
        out_w = H; out_h = W;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                // Point (x, y) in src maps to (y, W-1-x) in 270-CW dst.
                dst[(W - 1 - x) * out_w + y] = src[y * W + x];
            }
        break;
    }
}

std::vector<uint8_t> random_plane(int w, int h, uint64_t seed) {
    std::vector<uint8_t> data(w * h);
    uint64_t s = seed;
    for (auto& b : data) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        b = (uint8_t)(s >> 56);
    }
    return data;
}

int run_halide(Rot rot, Halide::Runtime::Buffer<uint8_t>& in,
               Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rot) {
    case Rot::CW90:  return rotate_fixed_1c_90cw (in, out);
    case Rot::ROT180: return rotate_fixed_1c_180  (in, out);
    case Rot::CW270: return rotate_fixed_1c_270cw(in, out);
    }
    return -1;
}

}  // namespace

struct Rotate1CParams { int w, h; Rot rot; const char* name; };

class Rotate1CTest : public ::testing::TestWithParam<Rotate1CParams> {};

TEST_P(Rotate1CTest, BitExactVsScalar) {
    auto p = GetParam();
    auto src = random_plane(p.w, p.h, 0xFEEDFACE + p.w * 1000 + p.h + (int)p.rot);

    int ow, oh;
    std::vector<uint8_t> dst_ref(p.w * p.h);  // max size, we'll only fill ow*oh
    scalar_rotate_1c(src.data(), p.w, p.h, p.rot, dst_ref.data(), ow, oh);

    Halide::Runtime::Buffer<uint8_t> in(src.data(), p.w, p.h);
    Halide::Runtime::Buffer<uint8_t> out(ow, oh);
    ASSERT_EQ(run_halide(p.rot, in, out), 0) << "rotate_fixed_1c_" << p.name << " failed";

    int mismatches = 0, first_bad = -1;
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            int h_val = out(x, y);
            int r_val = dst_ref[y * ow + x];
            if (h_val != r_val) {
                if (first_bad < 0) first_bad = y * ow + x;
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) {
        int px = first_bad % ow, py = first_bad / ow;
        printf("  FAIL %s %dx%d @ (%d,%d): halide=%d scalar=%d (%d total)\n",
               p.name, p.w, p.h, px, py,
               (int)out(px, py), dst_ref[py * ow + px], mismatches);
    }
    EXPECT_EQ(mismatches, 0) << "1-ch rotation must be bit-exact (pure index remap)";
}

INSTANTIATE_TEST_SUITE_P(
    Shapes,
    Rotate1CTest,
    ::testing::Values(
        Rotate1CParams{640,  480,  Rot::CW90,  "90cw"},
        Rotate1CParams{640,  480,  Rot::ROT180, "180"},
        Rotate1CParams{640,  480,  Rot::CW270, "270cw"},
        Rotate1CParams{1920, 1080, Rot::CW90,  "90cw"},
        Rotate1CParams{1920, 1080, Rot::ROT180, "180"},
        Rotate1CParams{1920, 1080, Rot::CW270, "270cw"},
        // Odd / degenerate sizes
        Rotate1CParams{641,  481,  Rot::CW90,  "90cw"},
        Rotate1CParams{641,  481,  Rot::ROT180, "180"},
        Rotate1CParams{641,  481,  Rot::CW270, "270cw"},
        Rotate1CParams{1,    1080, Rot::CW90,  "90cw"},
        Rotate1CParams{1080, 1,    Rot::CW270, "270cw"},
        Rotate1CParams{17,   33,   Rot::ROT180, "180"}
    )
);
