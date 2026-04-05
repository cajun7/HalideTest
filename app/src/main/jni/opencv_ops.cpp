#include "opencv_ops.h"
#include <opencv2/imgproc.hpp>

namespace opencv_ops {

void rgb_bgr(const cv::Mat& input, cv::Mat& output) {
    cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
}

void nv21_to_rgb(const cv::Mat& nv21, cv::Mat& output) {
    cv::cvtColor(nv21, output, cv::COLOR_YUV2RGB_NV21);
}

void nv21_yuv444_rgb(const cv::Mat& nv21, cv::Mat& output) {
    // OpenCV uses nearest-neighbor UV upsampling; timing comparison only.
    cv::cvtColor(nv21, output, cv::COLOR_YUV2RGB_NV21);
}

void nv21_to_rgb_full_range(const cv::Mat& nv21, cv::Mat& output) {
    // OpenCV's COLOR_YUV2RGB_NV21 uses limited-range coefficients.
    // No direct full-range variant in OpenCV 3.x; timing comparison only.
    cv::cvtColor(nv21, output, cv::COLOR_YUV2RGB_NV21);
}

void gaussian_blur_gray(const cv::Mat& input, cv::Mat& output, int kernel_size) {
    cv::GaussianBlur(input, output, cv::Size(kernel_size, kernel_size), 0);
}

void gaussian_blur_rgb(const cv::Mat& input, cv::Mat& output, int kernel_size) {
    cv::GaussianBlur(input, output, cv::Size(kernel_size, kernel_size), 0);
}

void lens_blur(const cv::Mat& input, cv::Mat& output, int radius) {
    // Build disc kernel
    int ksize = 2 * radius + 1;
    cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy <= radius * radius) {
                kernel.at<float>(dy + radius, dx + radius) = 1.0f;
                count++;
            }
        }
    }
    if (count > 0) kernel /= (float)count;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
}

void resize_bilinear(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
}

void resize_bicubic(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
}

void rotate_90(const cv::Mat& input, cv::Mat& output) {
    cv::rotate(input, output, cv::ROTATE_90_CLOCKWISE);
}

void rotate_180(const cv::Mat& input, cv::Mat& output) {
    cv::rotate(input, output, cv::ROTATE_180);
}

void rotate_270(const cv::Mat& input, cv::Mat& output) {
    cv::rotate(input, output, cv::ROTATE_90_COUNTERCLOCKWISE);
}

void rotate_angle(const cv::Mat& input, cv::Mat& output, float angle_degrees) {
    cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_degrees, 1.0);
    cv::warpAffine(input, output, rot_mat, input.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void rgb_to_nv21(const cv::Mat& rgb, cv::Mat& nv21_output) {
    int w = rgb.cols, h = rgb.rows;
    // OpenCV provides COLOR_RGB2YUV_YV12 which produces planar Y + V + U
    cv::Mat yv12;
    cv::cvtColor(rgb, yv12, cv::COLOR_RGB2YUV_YV12);

    // YV12 layout: Y plane (w*h), V plane (w/2 * h/2), U plane (w/2 * h/2)
    // NV21 layout: Y plane (w*h), interleaved VU pairs (w * h/2 bytes)
    int y_size = w * h;
    int uv_half = (w / 2) * (h / 2);
    nv21_output = cv::Mat(h + h / 2, w, CV_8UC1);

    // Copy Y plane
    uint8_t* src = yv12.data;
    uint8_t* dst = nv21_output.data;
    memcpy(dst, src, y_size);

    // Interleave V and U into NV21 VU pairs
    uint8_t* v_plane = src + y_size;
    uint8_t* u_plane = v_plane + uv_half;
    uint8_t* vu_dst = dst + y_size;
    for (int i = 0; i < uv_half; i++) {
        vu_dst[2 * i + 0] = v_plane[i];
        vu_dst[2 * i + 1] = u_plane[i];
    }
}

void resize_area(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
}

void resize_letterbox(const cv::Mat& input, cv::Mat& output, int target_w, int target_h) {
    // Compute uniform scale to fit entire image
    float sx = (float)target_w / input.cols;
    float sy = (float)target_h / input.rows;
    float scale = std::min(sx, sy);

    int scaled_w = (int)std::round(input.cols * scale);
    int scaled_h = (int)std::round(input.rows * scale);

    // Resize with bilinear interpolation
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);

    // Create black canvas and paste centered
    output = cv::Mat::zeros(target_h, target_w, input.type());
    int offset_x = (target_w - scaled_w) / 2;
    int offset_y = (target_h - scaled_h) / 2;
    resized.copyTo(output(cv::Rect(offset_x, offset_y, scaled_w, scaled_h)));
}

void flip_horizontal(const cv::Mat& input, cv::Mat& output) {
    cv::flip(input, output, 1);  // flipCode > 0 = horizontal (flip around y-axis)
}

void flip_vertical(const cv::Mat& input, cv::Mat& output) {
    cv::flip(input, output, 0);  // flipCode == 0 = vertical (flip around x-axis)
}

void nv21_rotate_flip_resize_rgb(const cv::Mat& nv21, cv::Mat& output,
                                 int rotation_degrees_cw, int flip_code,
                                 int target_w, int target_h, int interp) {
    // Step 1: NV21 -> RGB
    cv::Mat rgb;
    cv::cvtColor(nv21, rgb, cv::COLOR_YUV2RGB_NV21);

    // Step 2: Rotate
    cv::Mat rotated;
    switch (rotation_degrees_cw) {
        case 90:  cv::rotate(rgb, rotated, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(rgb, rotated, cv::ROTATE_180); break;
        case 270: cv::rotate(rgb, rotated, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default:  rotated = rgb; break;
    }

    // Step 3: Flip
    cv::Mat flipped;
    if (flip_code == 1) {
        cv::flip(rotated, flipped, 1);   // horizontal
    } else if (flip_code == 2) {
        cv::flip(rotated, flipped, 0);   // vertical
    } else {
        flipped = rotated;
    }

    // Step 4: Resize
    cv::resize(flipped, output, cv::Size(target_w, target_h), 0, 0, interp);
}

void nv21_resize_pad_rotate(const cv::Mat& nv21, cv::Mat& output,
                            int rotation_degrees_cw, int target_size) {
    // Step 1: NV21 -> RGB
    cv::Mat rgb;
    cv::cvtColor(nv21, rgb, cv::COLOR_YUV2RGB_NV21);

    // Step 2: Aspect-ratio-preserving resize
    float scale = std::min((float)target_size / rgb.cols,
                           (float)target_size / rgb.rows);
    int scaled_w = (int)std::round(rgb.cols * scale);
    int scaled_h = (int)std::round(rgb.rows * scale);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);

    // Step 3: Pad to square with black
    cv::Mat padded = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    int offset_x = (target_size - scaled_w) / 2;
    int offset_y = (target_size - scaled_h) / 2;
    resized.copyTo(padded(cv::Rect(offset_x, offset_y, scaled_w, scaled_h)));

    // Step 4: Rotate
    switch (rotation_degrees_cw) {
        case 90:  cv::rotate(padded, output, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(padded, output, cv::ROTATE_180); break;
        case 270: cv::rotate(padded, output, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default:  output = padded; break;
    }
}

void seg_argmax(const cv::Mat& input, cv::Mat& output, int num_classes) {
    // input: planar float data reinterpreted as single-channel rows
    // Each class plane is h*w floats laid out contiguously.
    int h = input.rows;
    int w = input.cols;
    output = cv::Mat(h, w, CV_8UC1);

    const float* data = (const float*)input.data;
    for (int y = 0; y < h; y++) {
        uint8_t* out_row = output.ptr<uint8_t>(y);
        for (int x = 0; x < w; x++) {
            float max_val = data[0 * h * w + y * w + x];
            int max_idx = 0;
            for (int c = 1; c < num_classes; c++) {
                float val = data[c * h * w + y * w + x];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            out_row[x] = (uint8_t)max_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized Operation References
// ---------------------------------------------------------------------------

void rgb_bgr_optimized(const cv::Mat& input, cv::Mat& output) {
    cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
}

void nv21_to_rgb_optimized(const cv::Mat& nv21, cv::Mat& output) {
    cv::cvtColor(nv21, output, cv::COLOR_YUV2RGB_NV21);
}

void rgb_to_nv21_optimized(const cv::Mat& rgb, cv::Mat& nv21_output) {
    rgb_to_nv21(rgb, nv21_output);
}

void resize_bilinear_optimized(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
}

void resize_area_optimized(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
}

void resize_bicubic_optimized(const cv::Mat& input, cv::Mat& output, int out_w, int out_h) {
    cv::resize(input, output, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
}

void nv21_resize_optimized(const cv::Mat& nv21, cv::Mat& nv21_out,
                           int target_w, int target_h, int interp) {
    // Reference: NV21 -> RGB -> resize -> RGB -> NV21
    cv::Mat rgb;
    cv::cvtColor(nv21, rgb, cv::COLOR_YUV2RGB_NV21);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_w, target_h), 0, 0, interp);
    rgb_to_nv21(resized, nv21_out);
}

void nv21_resize_rgb_optimized(const cv::Mat& nv21, cv::Mat& rgb_out,
                               int target_w, int target_h, int interp) {
    // Reference: NV21 -> RGB -> resize
    cv::Mat rgb;
    cv::cvtColor(nv21, rgb, cv::COLOR_YUV2RGB_NV21);
    cv::resize(rgb, rgb_out, cv::Size(target_w, target_h), 0, 0, interp);
}

} // namespace opencv_ops
