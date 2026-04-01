#include "test_common.h"
#include "rgb_to_nv21.h"
#include "nv21_to_rgb.h"

class RgbToNv21Test : public ::testing::TestWithParam<std::pair<int, int>> {};

// Helper: set up UV output buffer as raw 2D bytes (width x height/2).
// Returns a buffer backed by the provided storage vector.
static Halide::Runtime::Buffer<uint8_t> make_uv_output_buf(
    std::vector<uint8_t>& storage, int w, int h)
{
    storage.resize(w * (h / 2));
    return Halide::Runtime::Buffer<uint8_t>(storage.data(), w, h / 2);
}

// RGB -> NV21 -> RGB round-trip should be near-identity.
// Loss is expected from integer arithmetic rounding and 4:2:0 chroma subsampling.
TEST_P(RgbToNv21Test, RoundTrip_NearIdentity) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    cv::Mat rgb = make_test_image_rgb(w, h);
    auto input_buf = mat_to_halide_planar(rgb);

    // Forward: RGB -> NV21
    Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
    std::vector<uint8_t> uv_storage;
    auto uv_buf = make_uv_output_buf(uv_storage, w, h);

    int err = rgb_to_nv21(input_buf, y_buf, uv_buf);
    ASSERT_EQ(err, 0) << "rgb_to_nv21 failed";

    // Inverse: NV21 -> RGB
    Halide::Runtime::Buffer<uint8_t> recovered(w, h, 3);

    err = nv21_to_rgb(y_buf, uv_buf, recovered);
    ASSERT_EQ(err, 0) << "nv21_to_rgb failed";

    // Compare recovered vs original
    compare_buffers_rgb(recovered, rgb, /*tolerance=*/5, /*opencv_is_bgr=*/false);
}

// Y plane output should match BT.601 forward transform computed manually.
TEST_P(RgbToNv21Test, YPlane_MatchesBT601) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    cv::Mat rgb = make_test_image_rgb(w, h);
    auto input_buf = mat_to_halide_planar(rgb);

    Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
    std::vector<uint8_t> uv_storage;
    auto uv_buf = make_uv_output_buf(uv_storage, w, h);

    int err = rgb_to_nv21(input_buf, y_buf, uv_buf);
    ASSERT_EQ(err, 0);

    // Verify Y against manual BT.601: Y = ((66*R + 129*G + 25*B + 128) >> 8) + 16
    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int r = rgb.at<cv::Vec3b>(y, x)[0];
            int g = rgb.at<cv::Vec3b>(y, x)[1];
            int b = rgb.at<cv::Vec3b>(y, x)[2];
            int ref_y = std::min(255, std::max(0, ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16));
            int diff = std::abs((int)y_buf(x, y) - ref_y);
            if (diff > 1) mismatches++;
            max_diff = std::max(max_diff, diff);
        }
    }
    float total = (float)(w * h);
    float pct = 100.0f * mismatches / total;
    EXPECT_LT(pct, 1.0f)
        << "Y plane mismatches: " << mismatches << " (" << pct << "%), max_diff=" << max_diff;
}

// UV plane output should match BT.601 Cb/Cr with 2x2 block averaging.
TEST_P(RgbToNv21Test, UVPlane_MatchesBT601) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    cv::Mat rgb = make_test_image_rgb(w, h);
    auto input_buf = mat_to_halide_planar(rgb);

    Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
    std::vector<uint8_t> uv_storage;
    auto uv_buf = make_uv_output_buf(uv_storage, w, h);

    int err = rgb_to_nv21(input_buf, y_buf, uv_buf);
    ASSERT_EQ(err, 0);

    int mismatches = 0;
    int max_diff = 0;
    for (int by = 0; by < h / 2; by++) {
        for (int bx = 0; bx < w / 2; bx++) {
            // Compute reference Cb and Cr for the 2x2 block
            int cr_sum = 0, cb_sum = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int px = 2 * bx + dx;
                    int py = 2 * by + dy;
                    int r = rgb.at<cv::Vec3b>(py, px)[0];
                    int g = rgb.at<cv::Vec3b>(py, px)[1];
                    int b = rgb.at<cv::Vec3b>(py, px)[2];
                    cr_sum += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    cb_sum += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                }
            }
            int ref_cr = std::min(255, std::max(0, (cr_sum + 2) / 4));
            int ref_cb = std::min(255, std::max(0, (cb_sum + 2) / 4));

            // uv_buf is 2D raw bytes: V at even offsets, U at odd offsets
            int diff_cr = std::abs((int)uv_buf(bx * 2, by) - ref_cr);
            int diff_cb = std::abs((int)uv_buf(bx * 2 + 1, by) - ref_cb);
            if (diff_cr > 2) mismatches++;
            if (diff_cb > 2) mismatches++;
            max_diff = std::max(max_diff, std::max(diff_cr, diff_cb));
        }
    }
    float total = (float)(w / 2 * h / 2 * 2);
    float pct = 100.0f * mismatches / total;
    EXPECT_LT(pct, 1.0f)
        << "UV plane mismatches: " << mismatches << " (" << pct << "%), max_diff=" << max_diff;
}

// Cross-validate: RGB -> NV21 (Halide), then decode NV21 via OpenCV, compare against original.
TEST_P(RgbToNv21Test, RoundTrip_ViaOpenCV) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    cv::Mat rgb = make_test_image_rgb(w, h);
    auto input_buf = mat_to_halide_planar(rgb);

    // Forward: RGB -> NV21 via Halide
    Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
    std::vector<uint8_t> uv_storage;
    auto uv_buf = make_uv_output_buf(uv_storage, w, h);

    int err = rgb_to_nv21(input_buf, y_buf, uv_buf);
    ASSERT_EQ(err, 0);

    // Assemble contiguous NV21 buffer: Y plane then interleaved VU
    std::vector<uint8_t> nv21(w * h + w * (h / 2));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            nv21[y * w + x] = y_buf(x, y);
    std::copy(uv_storage.begin(), uv_storage.end(), nv21.begin() + w * h);

    // Decode NV21 -> RGB via OpenCV
    cv::Mat nv21_mat(h + h / 2, w, CV_8UC1, nv21.data());
    cv::Mat opencv_rgb;
    cv::cvtColor(nv21_mat, opencv_rgb, cv::COLOR_YUV2RGB_NV21);

    // Compare OpenCV-decoded RGB against original
    ASSERT_EQ(opencv_rgb.cols, w);
    ASSERT_EQ(opencv_rgb.rows, h);

    int mismatches = 0;
    int max_diff = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int orig = rgb.at<cv::Vec3b>(y, x)[c];
                int decoded = opencv_rgb.at<cv::Vec3b>(y, x)[c];
                int diff = std::abs(orig - decoded);
                if (diff > 5) mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    float total = (float)(w * h * 3);
    float pct = 100.0f * mismatches / total;
    EXPECT_LT(pct, 2.0f)
        << "Round-trip via OpenCV mismatches: " << mismatches
        << " (" << pct << "%), max_diff=" << max_diff;
}

// Verify no crash on all parameterized resolutions.
TEST_P(RgbToNv21Test, NoCrash_AllResolutions) {
    auto [width, height] = GetParam();
    int w = width & ~1;
    int h = height & ~1;

    cv::Mat rgb = make_test_image_rgb(w, h);
    auto input_buf = mat_to_halide_planar(rgb);

    Halide::Runtime::Buffer<uint8_t> y_buf(w, h);
    std::vector<uint8_t> uv_storage;
    auto uv_buf = make_uv_output_buf(uv_storage, w, h);

    int err = rgb_to_nv21(input_buf, y_buf, uv_buf);
    ASSERT_EQ(err, 0) << "rgb_to_nv21 crashed on " << w << "x" << h;
    SUCCEED() << "No crash on " << w << "x" << h;
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions,
    RgbToNv21Test,
    ::testing::Values(
        std::make_pair(320, 240),
        std::make_pair(640, 480),
        std::make_pair(641, 481),    // odd -> forced to 640x480
        std::make_pair(642, 482),    // even non-standard
        std::make_pair(1280, 720),
        std::make_pair(1279, 719),   // odd -> forced to 1278x718
        std::make_pair(1920, 1080)
    )
);
