#include "test_common.h"
#include "nv21_resize_pad_rotate_none.h"
#include "nv21_resize_pad_rotate_90cw.h"
#include "nv21_resize_pad_rotate_180.h"
#include "nv21_resize_pad_rotate_270cw.h"

// Dispatch by rotation code
static int call_resize_pad_rotate(Halide::Runtime::Buffer<uint8_t>& y,
                                  Halide::Runtime::Buffer<uint8_t>& uv,
                                  int rotation_cw, int target_size,
                                  Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rotation_cw) {
        case 0:   return nv21_resize_pad_rotate_none(y, uv, target_size, out);
        case 90:  return nv21_resize_pad_rotate_90cw(y, uv, target_size, out);
        case 180: return nv21_resize_pad_rotate_180(y, uv, target_size, out);
        case 270: return nv21_resize_pad_rotate_270cw(y, uv, target_size, out);
        default:  return -1;
    }
}

// C++ float reference that mirrors nv21_resize_pad_rotate_generator.cpp exactly.
// Same inverse mapping, bilinear Y+UV sampling, and full-range BT.601 fixed-point.
// Returns RGB cv::Mat (target_size x target_size).
static cv::Mat make_float_reference(const std::vector<uint8_t>& nv21,
                                    int src_w, int src_h,
                                    int rotation_cw, int target_size) {
    const uint8_t* y_data = nv21.data();
    const uint8_t* uv_data = nv21.data() + src_w * src_h;
    int uv_w = src_w;           // byte width of UV plane
    int uv_h = src_h / 2;      // row count of UV plane
    int ts = target_size;
    float tsf = (float)ts;
    float src_wf = (float)src_w;
    float src_hf = (float)src_h;

    // Aspect-ratio-preserving resize dimensions (same as generator)
    float scale_x = tsf / src_wf;
    float scale_y = tsf / src_hf;
    float uniform_scale = std::min(scale_x, scale_y);
    int scaled_w = (int)roundf(src_wf * uniform_scale);
    int scaled_h = (int)roundf(src_hf * uniform_scale);
    float scaled_wf = (float)scaled_w;
    float scaled_hf = (float)scaled_h;
    int pad_x = (ts - scaled_w) / 2;
    int pad_y = (ts - scaled_h) / 2;
    float pad_xf = (float)pad_x;
    float pad_yf = (float)pad_y;

    // repeat_edge helpers
    auto clamp_y_sample = [&](int ix, int iy) -> float {
        int cx = std::max(0, std::min(ix, src_w - 1));
        int cy = std::max(0, std::min(iy, src_h - 1));
        return (float)y_data[cy * src_w + cx];
    };
    int uv_w_half = src_w / 2;
    auto clamp_uv_px = [&](int px) -> int {
        return std::max(0, std::min(px, uv_w_half - 1));
    };
    auto clamp_uv_row = [&](int py) -> int {
        return std::max(0, std::min(py, uv_h - 1));
    };

    cv::Mat out(ts, ts, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int oy = 0; oy < ts; oy++) {
        for (int ox = 0; ox < ts; ox++) {
            // Step 1: Inverse rotation (same as generator lines 78-93)
            float rx, ry;
            switch (rotation_cw) {
                case 0:   rx = (float)ox; ry = (float)oy; break;
                case 90:  rx = (float)oy; ry = tsf - 1.0f - (float)ox; break;
                case 180: rx = tsf - 1.0f - (float)ox; ry = tsf - 1.0f - (float)oy; break;
                case 270: rx = tsf - 1.0f - (float)oy; ry = (float)ox; break;
                default:  rx = (float)ox; ry = (float)oy; break;
            }

            // Step 2: Check padding region
            bool in_region = (rx >= pad_xf) && (rx < pad_xf + scaled_wf) &&
                             (ry >= pad_yf) && (ry < pad_yf + scaled_hf);
            if (!in_region) {
                out.at<cv::Vec3b>(oy, ox) = cv::Vec3b(0, 0, 0);
                continue;
            }

            // Step 3: Inverse pad
            float img_x = rx - pad_xf;
            float img_y = ry - pad_yf;

            // Step 4: Inverse resize (same as generator lines 110-115)
            float sx = (img_x + 0.5f) * src_wf / scaled_wf - 0.5f;
            float sy = (img_y + 0.5f) * src_hf / scaled_hf - 0.5f;
            sx = std::max(0.0f, std::min(src_wf - 1.0f, sx));
            sy = std::max(0.0f, std::min(src_hf - 1.0f, sy));

            // Step 5: Bilinear sample Y at full-res
            int y_ix = (int)floorf(sx);
            int y_iy = (int)floorf(sy);
            float y_fx = sx - (float)y_ix;
            float y_fy = sy - (float)y_iy;

            float y_val = clamp_y_sample(y_ix, y_iy) * (1.0f - y_fx) * (1.0f - y_fy) +
                          clamp_y_sample(y_ix + 1, y_iy) * y_fx * (1.0f - y_fy) +
                          clamp_y_sample(y_ix, y_iy + 1) * (1.0f - y_fx) * y_fy +
                          clamp_y_sample(y_ix + 1, y_iy + 1) * y_fx * y_fy;

            // Step 6: Bilinear sample UV at half-res (same as generator lines 144-170)
            float uv_sx = sx / 2.0f;
            float uv_sy = sy / 2.0f;
            int uv_ix = (int)floorf(uv_sx);
            int uv_iy = (int)floorf(uv_sy);
            float uv_fx = uv_sx - (float)uv_ix;
            float uv_fy = uv_sy - (float)uv_iy;

            // Clamp UV pixel indices before computing byte offsets (matches generator)
            int cix0 = clamp_uv_px(uv_ix);
            int cix1 = clamp_uv_px(uv_ix + 1);
            int ciy0 = clamp_uv_row(uv_iy);
            int ciy1 = clamp_uv_row(uv_iy + 1);

            // V at even byte offsets, U at odd byte offsets
            float v00 = (float)uv_data[ciy0 * uv_w + cix0 * 2];
            float v10 = (float)uv_data[ciy0 * uv_w + cix1 * 2];
            float v01 = (float)uv_data[ciy1 * uv_w + cix0 * 2];
            float v11 = (float)uv_data[ciy1 * uv_w + cix1 * 2];
            float v_interp = v00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                             v10 * uv_fx * (1.0f - uv_fy) +
                             v01 * (1.0f - uv_fx) * uv_fy +
                             v11 * uv_fx * uv_fy;

            float u00 = (float)uv_data[ciy0 * uv_w + cix0 * 2 + 1];
            float u10 = (float)uv_data[ciy0 * uv_w + cix1 * 2 + 1];
            float u01 = (float)uv_data[ciy1 * uv_w + cix0 * 2 + 1];
            float u11 = (float)uv_data[ciy1 * uv_w + cix1 * 2 + 1];
            float u_interp = u00 * (1.0f - uv_fx) * (1.0f - uv_fy) +
                             u10 * uv_fx * (1.0f - uv_fy) +
                             u01 * (1.0f - uv_fx) * uv_fy +
                             u11 * uv_fx * uv_fy;

            // Step 7: Full-range BT.601 fixed-point (same as generator lines 175-183)
            int y_int = (int)std::max(0.0f, std::min(255.0f, y_val));
            int v_int = (int)std::max(0.0f, std::min(255.0f, v_interp)) - 128;
            int u_int = (int)std::max(0.0f, std::min(255.0f, u_interp)) - 128;

            int y_scaled = y_int * 256 + 128;
            int r = (y_scaled + 359 * v_int) >> 8;
            int g = (y_scaled - 88 * u_int - 183 * v_int) >> 8;
            int b = (y_scaled + 454 * u_int) >> 8;

            out.at<cv::Vec3b>(oy, ox) = cv::Vec3b(
                (uint8_t)std::max(0, std::min(255, r)),
                (uint8_t)std::max(0, std::min(255, g)),
                (uint8_t)std::max(0, std::min(255, b)));
        }
    }
    return out;
}

// Compare interleaved Halide RGB buffer against OpenCV RGB Mat
static void compare_rgb_vs_rgb(const Halide::Runtime::Buffer<uint8_t>& halide_buf,
                               const cv::Mat& cv_rgb, int tolerance,
                               const char* label) {
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

    printf("  %s: PSNR=%.1f dB max_diff=%d mismatch=%.2f%%\n",
           label, psnr, max_diff, pct);

    // Float reference mirrors generator's exact math — only FMA/rounding diffs.
    EXPECT_GT(psnr, 45.0)
        << "PSNR too low vs float reference";
    EXPECT_LT(pct, 1.0f)
        << "Too many pixel mismatches: " << mismatches
        << " (" << pct << "%), max_diff=" << max_diff;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

struct ResizePadRotateCase {
    int src_w, src_h, target_size, rotation_cw;
};

class NV21ResizePadRotateTest : public ::testing::TestWithParam<ResizePadRotateCase> {};

TEST_P(NV21ResizePadRotateTest, MatchesFloatReference) {
    auto p = GetParam();
    int src_w = p.src_w & ~1;
    int src_h = p.src_h & ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
        p.target_size, p.target_size, 3);

    int err = call_resize_pad_rotate(y_buf, uv_buf, p.rotation_cw,
                                     p.target_size, output_buf);
    ASSERT_EQ(err, 0) << "Failed: " << src_w << "x" << src_h
                       << " -> " << p.target_size << "x" << p.target_size
                       << " rot=" << p.rotation_cw;

    char label[128];
    snprintf(label, sizeof(label), "resize_pad_rot_%dx%d_%d_rot%d",
             src_w, src_h, p.target_size, p.rotation_cw);
    dump_if_first(output_buf, label, 0);

    // Float reference mirrors generator's exact math (full-range BT.601 + bilinear NV21)
    cv::Mat float_ref = make_float_reference(nv21, src_w, src_h,
                                             p.rotation_cw, p.target_size);
    dump_mat_if_first(float_ref, (std::string(label) + "_ref").c_str(), 0, false);

    // Tolerance: only FMA vs multiply+add float ordering diffs (~1-2 LSB)
    compare_rgb_vs_rgb(output_buf, float_ref, /*tolerance=*/3, label);
}

TEST_P(NV21ResizePadRotateTest, PaddingIsBlack) {
    auto p = GetParam();
    int src_w = p.src_w & ~1;
    int src_h = p.src_h & ~1;

    // Only test non-rotated to simplify padding region detection
    if (p.rotation_cw != 0) return;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);
    auto output_buf = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
        p.target_size, p.target_size, 3);

    int err = call_resize_pad_rotate(y_buf, uv_buf, 0, p.target_size, output_buf);
    ASSERT_EQ(err, 0);

    // Compute expected padding region
    float scale = std::min((float)p.target_size / src_w,
                           (float)p.target_size / src_h);
    int scaled_w = (int)roundf(src_w * scale);
    int scaled_h = (int)roundf(src_h * scale);
    int pad_x = (p.target_size - scaled_w) / 2;
    int pad_y = (p.target_size - scaled_h) / 2;

    int padding_errors = 0;
    for (int y = 0; y < p.target_size; y++) {
        for (int x = 0; x < p.target_size; x++) {
            bool in_image = (x >= pad_x && x < pad_x + scaled_w &&
                             y >= pad_y && y < pad_y + scaled_h);
            if (!in_image) {
                for (int c = 0; c < 3; c++) {
                    if (output_buf(x, y, c) != 0) padding_errors++;
                }
            }
        }
    }
    EXPECT_EQ(padding_errors, 0)
        << "Padding region should be black (0,0,0), found " << padding_errors << " non-zero pixels";
}

INSTANTIATE_TEST_SUITE_P(
    Configs,
    NV21ResizePadRotateTest,
    ::testing::Values(
        // Primary use cases
        ResizePadRotateCase{1920, 1080, 384, 0},
        ResizePadRotateCase{1920, 1080, 384, 90},
        ResizePadRotateCase{1920, 1080, 384, 180},
        ResizePadRotateCase{1920, 1080, 384, 270},
        // Samsung sensor resolutions
        ResizePadRotateCase{4000, 3000, 1408, 0},
        ResizePadRotateCase{4000, 3000, 1408, 90},
        ResizePadRotateCase{3840, 2160, 1408, 0},
        ResizePadRotateCase{3840, 2160, 1408, 90},
        ResizePadRotateCase{4032, 3024, 1408, 0},
        ResizePadRotateCase{4032, 3024, 1408, 90},
        ResizePadRotateCase{4608, 3456, 1408, 0},
        ResizePadRotateCase{4608, 3456, 1408, 90},
        // Smaller sizes for fast CI
        ResizePadRotateCase{640, 480, 384, 0},
        ResizePadRotateCase{640, 480, 384, 90},
        ResizePadRotateCase{640, 480, 384, 180},
        ResizePadRotateCase{640, 480, 384, 270},
        ResizePadRotateCase{1280, 720, 384, 0},
        ResizePadRotateCase{1280, 720, 384, 90}
    )
);

// ---------------------------------------------------------------------------
// NoCrash: all rotation codes for all resolutions
// ---------------------------------------------------------------------------

class ResizePadRotateNoCrashTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(ResizePadRotateNoCrashTest, AllRotations_NoCrash) {
    auto [width, height] = GetParam();
    int src_w = width & ~1;
    int src_h = height & ~1;

    auto nv21 = make_nv21_contiguous(src_w, src_h);
    uint8_t* y_ptr = nv21.data();
    uint8_t* uv_ptr = y_ptr + src_w * src_h;

    Halide::Runtime::Buffer<uint8_t> y_buf(y_ptr, src_w, src_h);
    Halide::Runtime::Buffer<uint8_t> uv_buf(uv_ptr, src_w, src_h / 2);

    int target = 384;
    int rotations[] = {0, 90, 180, 270};
    for (int rot : rotations) {
        auto out = Halide::Runtime::Buffer<uint8_t>::make_interleaved(target, target, 3);
        int err = call_resize_pad_rotate(y_buf, uv_buf, rot, target, out);
        ASSERT_EQ(err, 0) << "Crashed: " << src_w << "x" << src_h
                          << " rot=" << rot << " -> " << target << "x" << target;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    ResizePadRotateNoCrashTest,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),
        std::make_pair(1280, 720),
        std::make_pair(1920, 1080)
    )
);
