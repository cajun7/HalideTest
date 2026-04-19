// =============================================================================
// Host self-test for bt709_neon_ref.cpp.
//
// Verifies that the NEON implementation produces bit-exact output vs. the
// portable scalar reference, across a random sample of the input space plus
// a set of corner cases (odd widths, 1xN, Nx1, very large).
//
// Build (Apple Silicon / ARM64 Linux):
//   clang++ -O2 -std=c++17 -march=armv8-a \
//       app/src/main/jni/bt709_neon_ref.cpp \
//       app/src/main/jni/bt709_selftest.cpp \
//       -o /tmp/bt709_host && /tmp/bt709_host
//
// On x86 hosts the NEON path degenerates to scalar; the test still runs but
// only proves scalar is consistent with itself. Real verification must happen
// on-device.
// =============================================================================

#include "bt709_neon_ref.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace {

#if defined(__aarch64__) || defined(__ARM_NEON)
constexpr bool kHostHasNeon = true;
#else
constexpr bool kHostHasNeon = false;
#endif

struct NV21 {
    std::vector<uint8_t> y;   // width * height
    std::vector<uint8_t> uv;  // width * (height/2)   (rounded down)
    int width, height;
    int y_stride;
    int uv_stride;
};

NV21 random_nv21(int w, int h, uint64_t seed) {
    NV21 img;
    img.width = w;
    img.height = h;
    img.y_stride = w;
    img.uv_stride = w;
    img.y.resize(w * h);
    img.uv.resize(w * (h / 2) + (h % 2 == 0 ? 0 : 0));  // h/2 rows of UV
    // For odd height we still keep floor(h/2) UV rows; the last Y row
    // reuses the last UV row (handled by scalar+neon identically).
    if (img.uv.empty()) img.uv.resize(w);  // 1xH=1 case: allocate something

    std::mt19937_64 rng(seed);
    for (auto& b : img.y)  b = (uint8_t)rng();
    for (auto& b : img.uv) b = (uint8_t)rng();
    return img;
}

// Compare two RGB buffers pixel-by-pixel; return index of first mismatch
// or -1 if equal. Also capture the mismatching triple for diagnostics.
int compare_rgb(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                int* out_r_a, int* out_g_a, int* out_b_a,
                int* out_r_b, int* out_g_b, int* out_b_b) {
    for (size_t i = 0; i + 3 <= a.size(); i += 3) {
        if (a[i] != b[i] || a[i+1] != b[i+1] || a[i+2] != b[i+2]) {
            *out_r_a = a[i]; *out_g_a = a[i+1]; *out_b_a = a[i+2];
            *out_r_b = b[i]; *out_g_b = b[i+1]; *out_b_b = b[i+2];
            return (int)(i / 3);
        }
    }
    return -1;
}

bool test_size(int w, int h, uint64_t seed, int* total_pixels_out) {
    NV21 img = random_nv21(w, h, seed);
    std::vector<uint8_t> rgb_scalar(w * h * 3);
    std::vector<uint8_t> rgb_neon(w * h * 3);

    bt709::nv21_to_rgb_bt709_full_range_scalar(
        img.y.data(), img.y_stride,
        img.uv.data(), img.uv_stride,
        rgb_scalar.data(), w * 3,
        w, h);
    bt709::nv21_to_rgb_bt709_full_range_neon(
        img.y.data(), img.y_stride,
        img.uv.data(), img.uv_stride,
        rgb_neon.data(), w * 3,
        w, h);

    int ra, ga, ba, rb, gb, bb;
    int idx = compare_rgb(rgb_scalar, rgb_neon, &ra, &ga, &ba, &rb, &gb, &bb);
    *total_pixels_out = w * h;
    if (idx >= 0) {
        int px_x = idx % w, px_y = idx / w;
        std::printf("  FAIL %dx%d @ pixel (%d,%d): "
                    "scalar=(%3d,%3d,%3d) neon=(%3d,%3d,%3d)\n",
                    w, h, px_x, px_y, ra, ga, ba, rb, gb, bb);
        return false;
    }
    return true;
}

// Dense sweep over all 256^3 (Y, U, V) triples: for each of the 65536 (U, V)
// chroma combinations, run a 256x2 image whose Y rows contain every Y value
// [0..255] once. Catches saturation-edge divergences that random sampling
// can miss.
bool test_exhaustive() {
    const int W = 256;   // 256 distinct Y columns => 128 VU pairs per UV row
    const int H = 2;     // 1 UV row, shared by 2 Y rows

    std::vector<uint8_t> y_plane(W * H);
    std::vector<uint8_t> uv_plane(W);  // 1 UV row of W bytes (128 VU pairs)
    std::vector<uint8_t> rgb_scalar(W * H * 3);
    std::vector<uint8_t> rgb_neon(W * H * 3);

    // Both Y rows: 0, 1, 2, ..., 255.
    for (int row = 0; row < H; ++row) {
        for (int x = 0; x < W; ++x) y_plane[row * W + x] = (uint8_t)x;
    }

    int64_t total_mismatches = 0;
    int64_t total_triples = 0;
    for (int V = 0; V < 256; ++V) {
        for (int U = 0; U < 256; ++U) {
            // Fill UV row with constant (V, U).
            for (int i = 0; i < W / 2; ++i) {
                uv_plane[2 * i + 0] = (uint8_t)V;
                uv_plane[2 * i + 1] = (uint8_t)U;
            }
            bt709::nv21_to_rgb_bt709_full_range_scalar(
                y_plane.data(), W, uv_plane.data(), W,
                rgb_scalar.data(), W * 3, W, H);
            bt709::nv21_to_rgb_bt709_full_range_neon(
                y_plane.data(), W, uv_plane.data(), W,
                rgb_neon.data(), W * 3, W, H);
            for (int i = 0; i < W * H * 3; ++i) {
                if (rgb_scalar[i] != rgb_neon[i]) ++total_mismatches;
            }
            total_triples += W * H;  // 512 Y samples per (V, U) combo
        }
    }

    if (total_mismatches != 0) {
        std::printf("  FAIL exhaustive: %lld mismatching bytes across %lld pixels\n",
                    (long long)total_mismatches, (long long)total_triples);
        return false;
    }
    std::printf("  OK exhaustive: %lld pixels across all 65536 (U,V) pairs bit-exact\n",
                (long long)total_triples);
    return true;
}

}  // namespace

int main() {
    std::printf("bt709_neon_ref self-test\n");
    std::printf("  host NEON: %s\n", kHostHasNeon ? "yes (real test)" : "no (scalar-only)");

    struct Size { int w, h; };
    const Size sizes[] = {
        // Typical
        { 320,  240}, { 640,  480}, {1280,  720}, {1920, 1080},
        // Odd widths / heights
        { 641,  481}, {1279,  719}, {  15,   15}, {  17,   33},
        // Degenerate
        {   1,    1}, {   2,    2}, {  16,    1}, {   1,   16},
        // Single row / column
        {1920,    1}, {   1, 1080},
        // Large (skip by default; uncomment for bandwidth test)
        // {3840, 2160},
    };

    bool ok = true;
    int total_pixels = 0;
    uint64_t seed = 0xDEADBEEF;
    for (const auto& s : sizes) {
        int px = 0;
        if (!test_size(s.w, s.h, seed++, &px)) ok = false;
        total_pixels += px;
    }
    std::printf("  %s sizes: %zu shapes, %d pixels total\n",
                ok ? "OK" : "FAIL", sizeof(sizes) / sizeof(sizes[0]), total_pixels);

    ok = test_exhaustive() && ok;

    std::printf("%s\n", ok ? "SELF-TEST PASSED" : "SELF-TEST FAILED");
    return ok ? 0 : 1;
}
