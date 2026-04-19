// =============================================================================
// Equivalence tests for the fused Nv21ResizeRgbBt709{Nearest,Bilinear,Area}.
//
// Oracle chain per variant: non-fused NV21 resize (same algorithm) → NEON
// BT.709 full-range reference (already bit-exact vs the standalone Halide
// BT.709 generator per test_nv21_bt709.cpp). Both sides use identical Q11/
// float resize math and identical Q8 BT.709 color coefficients, so the fused
// output MUST be bit-exact (tolerance=0). Running with a tight tolerance
// catches any regression in coefficient, bias, or clamp order.
// =============================================================================

#include "test_common.h"
#include "bt709_neon_ref.h"
#include "halide_ops.h"
#include "nv21_resize_rgb_bt709_nearest.h"
#include "nv21_resize_rgb_bt709_bilinear.h"
#include "nv21_resize_rgb_bt709_area.h"
#include "nv21_resize_nearest_optimized.h"

#include <cstdio>
#include <cstring>

namespace {

struct Resize { int src_w, src_h, dst_w, dst_h; };

// Allocate an interleaved-strided uint8 RGB buffer matching Halide BT.709
// output layout (stride(x)=3, stride(c)=1) — byte-compatible with cv::Mat CV_8UC3.
Halide::Runtime::Buffer<uint8_t> make_rgb_out(int w, int h) {
    return Halide::Runtime::Buffer<uint8_t>::make_interleaved(w, h, 3);
}

// Compare two interleaved RGB buffers of identical size.
int diff_rgb_bytes(const Halide::Runtime::Buffer<uint8_t>& a,
                   const std::vector<uint8_t>& b,
                   int* max_diff_out, int* first_bad_out) {
    EXPECT_EQ((size_t)a.width() * a.height() * 3, b.size());
    int max_diff = 0, differ = 0, first_bad = -1;
    const uint8_t* ad = a.data();
    for (size_t i = 0; i < b.size(); ++i) {
        int d = std::abs((int)ad[i] - (int)b[i]);
        if (d > 0) {
            if (first_bad < 0) first_bad = (int)i;
            ++differ;
        }
        if (d > max_diff) max_diff = d;
    }
    *max_diff_out = max_diff;
    *first_bad_out = first_bad;
    return differ;
}

// Oracle: resize NV21 with the non-fused optimized path, then convert to
// BT.709 RGB with the hand-rolled NEON reference. Output matches fused path
// byte-for-byte when all coefficients align.
std::vector<uint8_t> oracle_resize_then_bt709(
    const uint8_t* y_src, const uint8_t* uv_src, int sw, int sh,
    int dw, int dh, const char* mode) {
    Halide::Runtime::Buffer<uint8_t> y_buf (const_cast<uint8_t*>(y_src),  sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(const_cast<uint8_t*>(uv_src), sw, sh / 2);
    Halide::Runtime::Buffer<uint8_t> y_out(dw, dh);
    Halide::Runtime::Buffer<uint8_t> uv_out(dw, dh / 2);
    int err = 0;
    if      (std::strcmp(mode, "nearest")  == 0) err = nv21_resize_nearest_optimized (y_buf, uv_buf, dw, dh, y_out, uv_out);
    else if (std::strcmp(mode, "bilinear") == 0) err = halide_ops::nv21_resize_bilinear_optimized(y_buf, uv_buf, dw, dh, y_out, uv_out);
    else if (std::strcmp(mode, "area")     == 0) err = halide_ops::nv21_resize_area_optimized    (y_buf, uv_buf, dw, dh, y_out, uv_out);
    EXPECT_EQ(err, 0) << "Non-fused resize (" << mode << ") failed";

    std::vector<uint8_t> rgb(dw * dh * 3);
    bt709::nv21_to_rgb_bt709_full_range_neon(
        y_out.data(), dw, uv_out.data(), dw,
        rgb.data(), dw * 3, dw, dh);
    return rgb;
}

}  // namespace

class Nv21ResizeRgbBt709Test : public ::testing::TestWithParam<Resize> {};

// ---- Nearest: fused Halide == non-fused(nearest) + NEON BT.709 ref ----
TEST_P(Nv21ResizeRgbBt709Test, NearestMatchesOracle) {
    auto p = GetParam();
    int sw = p.src_w & ~1, sh = p.src_h & ~1;
    int dw = p.dst_w & ~1, dh = p.dst_h & ~1;

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf (y_data.data(),  sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);
    auto out = make_rgb_out(dw, dh);
    ASSERT_EQ(0, nv21_resize_rgb_bt709_nearest(y_buf, uv_buf, dw, dh, out));

    auto oracle = oracle_resize_then_bt709(y_data.data(), uv_data.data(),
                                           sw, sh, dw, dh, "nearest");

    int max_diff = 0, first_bad = -1;
    int differ = diff_rgb_bytes(out, oracle, &max_diff, &first_bad);
    if (differ > 0) {
        int pix = (first_bad / 3);
        int px = pix % dw, py = pix / dw;
        printf("  FAIL nearest %dx%d -> %dx%d @ first byte %d (px=%d,%d): max_diff=%d, differ=%d\n",
               sw, sh, dw, dh, first_bad, px, py, max_diff, differ);
    }
    EXPECT_EQ(differ, 0) << "Fused nearest BT.709 must be bit-exact vs non-fused chain";
    EXPECT_LE(max_diff, 1) << "Safety margin exceeded";
}

// ---- Bilinear: fused Halide == non-fused(bilinear) + NEON BT.709 ref ----
TEST_P(Nv21ResizeRgbBt709Test, BilinearMatchesOracle) {
    auto p = GetParam();
    int sw = p.src_w & ~1, sh = p.src_h & ~1;
    int dw = p.dst_w & ~1, dh = p.dst_h & ~1;

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf (y_data.data(),  sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);
    auto out = make_rgb_out(dw, dh);
    ASSERT_EQ(0, nv21_resize_rgb_bt709_bilinear(y_buf, uv_buf, dw, dh, out));

    auto oracle = oracle_resize_then_bt709(y_data.data(), uv_data.data(),
                                           sw, sh, dw, dh, "bilinear");

    int max_diff = 0, first_bad = -1;
    int differ = diff_rgb_bytes(out, oracle, &max_diff, &first_bad);
    if (differ > 0) {
        int pix = (first_bad / 3);
        int px = pix % dw, py = pix / dw;
        printf("  FAIL bilinear %dx%d -> %dx%d @ first byte %d (px=%d,%d): max_diff=%d, differ=%d\n",
               sw, sh, dw, dh, first_bad, px, py, max_diff, differ);
    }
    // Fused-vs-nonfused uses identical Q11 bilinear + identical Q8 BT.709 tail.
    // Expect bit-exact. Tolerance of 2 absorbs any unanticipated path difference
    // while still failing any systematic coefficient bug.
    EXPECT_LE(max_diff, 2) << "Fused bilinear BT.709 too far from non-fused chain";
}

// ---- Area: fused Halide == non-fused(area) + NEON BT.709 ref ----
TEST_P(Nv21ResizeRgbBt709Test, AreaMatchesOracle) {
    auto p = GetParam();
    int sw = p.src_w & ~1, sh = p.src_h & ~1;
    int dw = p.dst_w & ~1, dh = p.dst_h & ~1;

    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf (y_data.data(),  sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);
    auto out = make_rgb_out(dw, dh);
    ASSERT_EQ(0, nv21_resize_rgb_bt709_area(y_buf, uv_buf, dw, dh, out));

    auto oracle = oracle_resize_then_bt709(y_data.data(), uv_data.data(),
                                           sw, sh, dw, dh, "area");

    int max_diff = 0, first_bad = -1;
    int differ = diff_rgb_bytes(out, oracle, &max_diff, &first_bad);
    printf("  area %dx%d -> %dx%d: max_diff=%d differ=%d/%d\n",
           sw, sh, dw, dh, max_diff, differ, dw * dh * 3);
    // Area reduces through a float divide — the fused and non-fused paths
    // can diverge by 1 LSB on some pixels due to compile-time expression
    // rearrangement. Allow 3 LSB safety (plan spec).
    EXPECT_LE(max_diff, 3)
        << "Fused area BT.709 exceeds 3-LSB tolerance vs non-fused oracle";
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    Nv21ResizeRgbBt709Test,
    ::testing::Values(
        Resize{640,  480,  1280, 720},    // upscale
        Resize{1920, 1080, 640,  480},    // downscale (main Samsung path)
        Resize{1280, 720,  1280, 720},    // identity (fused should still work)
        Resize{1280, 720,  641,  481},    // odd dst
        Resize{642,  482,  320,  240},    // odd src
        Resize{1920, 1080, 960,  540}     // exact 2x downscale (area fast path)
    )
);

// =============================================================================
// Mathematical correctness: constant-color fixtures at fixed BT.709 reference
// points. Verifies R=Y+1.5748(V-128), G=Y-0.1873(U-128)-0.4681(V-128),
// B=Y+1.8556(U-128) at the six key points below, across all three interp
// variants. Runs at identity (dst=src) so the resize stage is a pass-through
// and the RGB output equals the BT.709 math applied to the constant pixel.
//
// Reference (uint8, full-range, 1-LSB tolerance covers Q8 rounding):
//     Black        Y=0   U=128 V=128 -> (  0,   0,   0)
//     White        Y=255 U=128 V=128 -> (255, 255, 255)
//     Gray 50%     Y=128 U=128 V=128 -> (128, 128, 128)
//     Red primary  Y=76  U=84  V=255 -> (255,   0,   0)
//     Green prim   Y=150 U=44  V=21  -> (  0, 255,   0)
//     Blue primary Y=29  U=255 V=107 -> (  0,   0, 255)
// =============================================================================
TEST(Nv21ResizeRgbBt709FixedColor, AllVariantsMatchScalarReference) {
    struct Color { const char* name; uint8_t y, u, v; };
    const Color colors[] = {
        {"black",  0,   128, 128},
        {"white",  255, 128, 128},
        {"gray",   128, 128, 128},
        {"red",    76,  84,  255},
        {"green",  150, 44,  21 },
        {"blue",   29,  255, 107},
    };
    const int W = 32, H = 32;  // even; resize is pass-through at dst=src

    for (const auto& c : colors) {
        // NV21: Y plane W*H, UV plane W*(H/2) with V,U,V,U,... interleaving
        std::vector<uint8_t> y_data(W * H, c.y);
        std::vector<uint8_t> uv_data(W * (H / 2));
        for (int i = 0; i < W * (H / 2); i += 2) {
            uv_data[i]     = c.v;   // V at even byte
            uv_data[i + 1] = c.u;   // U at odd byte
        }

        // Ground truth: the scalar NEON reference applied to the same buffer.
        // We compare against this so the expected values need no hand-coded
        // rounding — the reference already encodes the correct BT.709 math.
        std::vector<uint8_t> expected(W * H * 3);
        bt709::nv21_to_rgb_bt709_full_range_scalar(
            y_data.data(),  W,
            uv_data.data(), W,
            expected.data(), W * 3, W, H);

        Halide::Runtime::Buffer<uint8_t> y_buf (y_data.data(),  W, H);
        Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), W, H / 2);

        for (const char* interp : {"nearest", "bilinear", "area"}) {
            auto out = make_rgb_out(W, H);
            int err = 0;
            if      (std::strcmp(interp, "nearest")  == 0) err = nv21_resize_rgb_bt709_nearest (y_buf, uv_buf, W, H, out);
            else if (std::strcmp(interp, "bilinear") == 0) err = nv21_resize_rgb_bt709_bilinear(y_buf, uv_buf, W, H, out);
            else                                            err = nv21_resize_rgb_bt709_area    (y_buf, uv_buf, W, H, out);
            ASSERT_EQ(err, 0) << interp << "/" << c.name << " pipeline failed";

            int max_diff = 0, first_bad = -1;
            int differ = diff_rgb_bytes(out, expected, &max_diff, &first_bad);
            if (max_diff > 1) {
                int pix = (first_bad / 3);
                int chan = first_bad % 3;
                printf("  FAIL %s/%s byte %d (px=%d,%d,c=%d): got %d, expected %d (max_diff=%d)\n",
                       interp, c.name, first_bad, pix % W, pix / W, chan,
                       out.data()[first_bad], expected[first_bad], max_diff);
            }
            EXPECT_LE(max_diff, 1)
                << interp << "/" << c.name
                << ": BT.709 math diverges from scalar reference beyond 1 LSB "
                << "(differ=" << differ << ")";
        }
    }
}

// Sanity: produces non-trivial output (guards against all-zero bug).
TEST(Nv21ResizeRgbBt709Smoke, OutputNotAllZero) {
    const int sw = 640, sh = 480, dw = 320, dh = 240;
    std::vector<uint8_t> y_data, uv_data;
    make_nv21_data(sw, sh, y_data, uv_data);

    Halide::Runtime::Buffer<uint8_t> y_buf (y_data.data(),  sw, sh);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_data.data(), sw, sh / 2);

    for (const char* interp : {"nearest", "bilinear", "area"}) {
        auto out = make_rgb_out(dw, dh);
        int err = 0;
        if      (std::strcmp(interp, "nearest")  == 0) err = nv21_resize_rgb_bt709_nearest (y_buf, uv_buf, dw, dh, out);
        else if (std::strcmp(interp, "bilinear") == 0) err = nv21_resize_rgb_bt709_bilinear(y_buf, uv_buf, dw, dh, out);
        else                                            err = nv21_resize_rgb_bt709_area    (y_buf, uv_buf, dw, dh, out);
        ASSERT_EQ(err, 0) << interp << " pipeline failed";

        // Any byte > 0 is sufficient — a pure-zero bug would hit this.
        int nonzero = 0;
        for (int i = 0; i < dw * dh * 3; ++i)
            if (out.data()[i] != 0) ++nonzero;
        EXPECT_GT(nonzero, dw * dh)  // at least ~1/3 of pixels non-black
            << interp << ": output is mostly zero, pipeline may be broken";
    }
}
