#include "opencv_ops.h"
#include <opencv2/imgproc.hpp>

namespace opencv_ops {

void rgb_bgr(const cv::Mat& input, cv::Mat& output) {
    cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
}

void nv21_to_rgb(const cv::Mat& nv21, cv::Mat& output) {
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

} // namespace opencv_ops
