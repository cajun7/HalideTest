#include "halide_ops.h"

// Generated Halide headers
#include "rgb_bgr_convert.h"
#include "nv21_to_rgb.h"
#include "gaussian_blur_y.h"
#include "gaussian_blur_rgb.h"
#include "lens_blur.h"
#include "resize_bilinear.h"
#include "resize_bicubic.h"
#include "rotate_fixed.h"
#include "rotate_arbitrary.h"
#include "rgb_to_nv21.h"
#include "resize_area.h"
#include "resize_letterbox.h"

namespace halide_ops {

int rgb_bgr(Halide::Runtime::Buffer<uint8_t>& input,
            Halide::Runtime::Buffer<uint8_t>& output) {
    return rgb_bgr_convert(input, output);
}

int nv21_to_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                Halide::Runtime::Buffer<uint8_t>& output) {
    return ::nv21_to_rgb(y_plane, uv_plane, output);
}

int gaussian_blur_gray(Halide::Runtime::Buffer<uint8_t>& input,
                       Halide::Runtime::Buffer<uint8_t>& output) {
    return gaussian_blur_y(input, output);
}

int gaussian_blur_rgb(Halide::Runtime::Buffer<uint8_t>& input,
                      Halide::Runtime::Buffer<uint8_t>& output) {
    return ::gaussian_blur_rgb(input, output);
}

int lens_blur(Halide::Runtime::Buffer<uint8_t>& input,
              int radius,
              Halide::Runtime::Buffer<uint8_t>& output) {
    return ::lens_blur(input, radius, output);
}

int resize_bilinear(Halide::Runtime::Buffer<uint8_t>& input,
                    float scale_x, float scale_y,
                    Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_bilinear(input, scale_x, scale_y, output);
}

int resize_bicubic(Halide::Runtime::Buffer<uint8_t>& input,
                   float scale_x, float scale_y,
                   Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_bicubic(input, scale_x, scale_y, output);
}

int rotate_90(Halide::Runtime::Buffer<uint8_t>& input,
              Halide::Runtime::Buffer<uint8_t>& output) {
    return rotate_fixed(input, output);
}

int rotate_angle(Halide::Runtime::Buffer<uint8_t>& input,
                 float angle_rad,
                 Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_arbitrary(input, angle_rad, output);
}

int rgb_to_nv21(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& y_output,
                Halide::Runtime::Buffer<uint8_t>& uv_output) {
    return ::rgb_to_nv21(input, y_output, uv_output);
}

int resize_area(Halide::Runtime::Buffer<uint8_t>& input,
                float scale_x, float scale_y,
                Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_area(input, scale_x, scale_y, output);
}

int resize_letterbox(Halide::Runtime::Buffer<uint8_t>& input,
                     int target_w, int target_h,
                     Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_letterbox(input, target_w, target_h, output);
}

} // namespace halide_ops
