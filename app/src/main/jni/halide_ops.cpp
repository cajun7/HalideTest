#include "halide_ops.h"

// Generated Halide headers
#include "rgb_bgr_convert.h"
#include "nv21_to_rgb.h"
#include "gaussian_blur_y.h"
#include "gaussian_blur_rgb.h"
#include "lens_blur.h"
#include "resize_bilinear.h"
#include "resize_bicubic.h"
#include "resize_bilinear_target.h"
#include "resize_bicubic_target.h"
#include "rotate_fixed_90cw.h"
#include "rotate_fixed_180.h"
#include "rotate_fixed_270cw.h"
#include "rotate_arbitrary.h"
#include "rgb_to_nv21.h"
#include "resize_area.h"
#include "resize_area_target.h"
#include "resize_letterbox.h"
#include "flip_horizontal.h"
#include "flip_vertical.h"
#include "nv21_pipeline_bilinear_none.h"
#include "nv21_pipeline_bilinear_90cw.h"
#include "nv21_pipeline_bilinear_180.h"
#include "nv21_pipeline_bilinear_270cw.h"
#include "nv21_pipeline_area_none.h"
#include "nv21_pipeline_area_90cw.h"
#include "nv21_pipeline_area_180.h"
#include "nv21_pipeline_area_270cw.h"

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

int rotate_90cw(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_90cw(input, output);
}

int rotate_180(Halide::Runtime::Buffer<uint8_t>& input,
               Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_180(input, output);
}

int rotate_270cw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_270cw(input, output);
}

int rotate_90ccw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_270cw(input, output);
}

int rotate_270ccw(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_90cw(input, output);
}

int rotate_90(Halide::Runtime::Buffer<uint8_t>& input,
              Halide::Runtime::Buffer<uint8_t>& output) {
    return ::rotate_fixed_90cw(input, output);
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

int resize_bilinear_target(Halide::Runtime::Buffer<uint8_t>& input,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_bilinear_target(input, target_w, target_h, output);
}

int resize_bicubic_target(Halide::Runtime::Buffer<uint8_t>& input,
                          int target_w, int target_h,
                          Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_bicubic_target(input, target_w, target_h, output);
}

int resize_area_target(Halide::Runtime::Buffer<uint8_t>& input,
                       int target_w, int target_h,
                       Halide::Runtime::Buffer<uint8_t>& output) {
    return ::resize_area_target(input, target_w, target_h, output);
}

int flip_horizontal(Halide::Runtime::Buffer<uint8_t>& input,
                    Halide::Runtime::Buffer<uint8_t>& output) {
    return ::flip_horizontal(input, output);
}

int flip_vertical(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output) {
    return ::flip_vertical(input, output);
}

// Helper: dispatch bilinear fused pipeline by rotation code
static int dispatch_nv21_bilinear(Halide::Runtime::Buffer<uint8_t>& y,
                                  Halide::Runtime::Buffer<uint8_t>& uv,
                                  int rotation_cw, int flip,
                                  int tw, int th,
                                  Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rotation_cw) {
        case 0:   return ::nv21_pipeline_bilinear_none(y, uv, flip, tw, th, out);
        case 90:  return ::nv21_pipeline_bilinear_90cw(y, uv, flip, tw, th, out);
        case 180: return ::nv21_pipeline_bilinear_180(y, uv, flip, tw, th, out);
        case 270: return ::nv21_pipeline_bilinear_270cw(y, uv, flip, tw, th, out);
        default:  return -1;
    }
}

// Helper: dispatch area fused pipeline by rotation code
static int dispatch_nv21_area(Halide::Runtime::Buffer<uint8_t>& y,
                              Halide::Runtime::Buffer<uint8_t>& uv,
                              int rotation_cw, int flip,
                              int tw, int th,
                              Halide::Runtime::Buffer<uint8_t>& out) {
    switch (rotation_cw) {
        case 0:   return ::nv21_pipeline_area_none(y, uv, flip, tw, th, out);
        case 90:  return ::nv21_pipeline_area_90cw(y, uv, flip, tw, th, out);
        case 180: return ::nv21_pipeline_area_180(y, uv, flip, tw, th, out);
        case 270: return ::nv21_pipeline_area_270cw(y, uv, flip, tw, th, out);
        default:  return -1;
    }
}

int nv21_rotate_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                           Halide::Runtime::Buffer<uint8_t>& uv_plane,
                           int rotation_degrees_cw,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output) {
    return dispatch_nv21_bilinear(y_plane, uv_plane, rotation_degrees_cw,
                                 0, target_w, target_h, output);
}

int nv21_rotate_flip_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw, int flip_code,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output) {
    return dispatch_nv21_bilinear(y_plane, uv_plane, rotation_degrees_cw,
                                 flip_code, target_w, target_h, output);
}

int nv21_rotate_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output) {
    return dispatch_nv21_area(y_plane, uv_plane, rotation_degrees_cw,
                              0, target_w, target_h, output);
}

int nv21_rotate_flip_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                     Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                     int rotation_degrees_cw, int flip_code,
                                     int target_w, int target_h,
                                     Halide::Runtime::Buffer<uint8_t>& output) {
    return dispatch_nv21_area(y_plane, uv_plane, rotation_degrees_cw,
                              flip_code, target_w, target_h, output);
}

} // namespace halide_ops
