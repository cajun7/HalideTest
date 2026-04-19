#pragma once

#include "HalideBuffer.h"
#include <cstdint>

// =============================================================================
// Halide Operation Wrappers — Public API
// =============================================================================
//
// This header declares the C++ wrapper layer around Halide AOT-compiled
// (Ahead-Of-Time compiled) image processing functions.
//
// ## Why This Layer Exists
//
// Halide generators produce plain C functions with raw `halide_buffer_t*`
// parameters. This wrapper layer provides:
//   1. Type-safe C++ interface using `Halide::Runtime::Buffer<T>`
//   2. A single `halide_ops` namespace for all operations
//   3. Name disambiguation (e.g., `halide_ops::nv21_to_rgb` vs `::nv21_to_rgb`)
//   4. Convenience aliases (e.g., CCW rotation mapped to CW variants)
//
// ## Buffer Types
//
// `Halide::Runtime::Buffer<T>` is Halide's C++ buffer wrapper. It:
//   - Owns or borrows a pointer to pixel data
//   - Stores dimension metadata (extent, stride, min) for each dimension
//   - Automatically converts to `halide_buffer_t*` (zero-copy) when passed
//     to AOT-generated functions. No memcpy — just passes the internal pointer.
//
// Common buffer shapes used in this API:
//   - Buffer<uint8_t, 3>: RGB image (width x height x 3, interleaved)
//   - Buffer<uint8_t, 2>: Grayscale or NV21 plane (width x height)
//   - Buffer<float, 3>:   ML model output (width x height x classes, planar)
//
// ## Return Values
//
// All functions return int:
//   - 0 = success
//   - Non-zero = Halide runtime error (e.g., buffer dimension mismatch,
//     out-of-memory, or assertion failure in the generated code)
//
// =============================================================================
namespace halide_ops {

// ---------------------------------------------------------------------------
// Color Space Conversions
// ---------------------------------------------------------------------------

// Swap R and B channels in a 3-channel interleaved image.
// Works for both RGB->BGR and BGR->RGB (the operation is symmetric).
//
// Input:  Buffer<uint8_t, 3> — width x height x 3, interleaved (stride: x=3, c=1)
// Output: Buffer<uint8_t, 3> — same dimensions and layout
int rgb_bgr(Halide::Runtime::Buffer<uint8_t>& input,
            Halide::Runtime::Buffer<uint8_t>& output);

// Convert NV21 (YUV 4:2:0 semi-planar) to RGB using BT.601 limited-range.
//
// NV21 is the standard Android camera YUV format:
//   - Y plane:  full resolution (width x height), one luma byte per pixel
//   - UV plane: half resolution, interleaved V,U byte pairs
//               Layout: V0 U0 V1 U1 ... (width bytes per row, height/2 rows)
//               V is at even byte offsets, U at odd offsets
//
// BT.601 limited-range: Y range [16, 235], UV range [16, 240]
//
// y_plane:  Buffer<uint8_t, 2> — width x height
// uv_plane: Buffer<uint8_t, 2> — width x (height/2) raw bytes
// output:   Buffer<uint8_t, 3> — width x height x 3 (RGB interleaved)
int nv21_to_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                Halide::Runtime::Buffer<uint8_t>& output);

// Convert NV21 to RGB with bilinear UV upsampling (YUV444 quality).
// Interpolates UV from half-res to full-res before BT.601 conversion,
// producing smoother chroma transitions than nearest-neighbor.
//
// Same inputs/outputs as nv21_to_rgb.
int nv21_yuv444_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                    Halide::Runtime::Buffer<uint8_t>& uv_plane,
                    Halide::Runtime::Buffer<uint8_t>& output);

// Convert NV21 to RGB using full-range BT.601 (JFIF/Android Camera).
// Uses Y:[0,255] UV:[0,255] instead of limited-range Y:[16,235] UV:[16,240].
//
// Same inputs/outputs as nv21_to_rgb.
int nv21_to_rgb_full_range(Halide::Runtime::Buffer<uint8_t>& y_plane,
                           Halide::Runtime::Buffer<uint8_t>& uv_plane,
                           Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Blur Operations
// ---------------------------------------------------------------------------

// Gaussian blur on a single-channel (grayscale) image.
// Uses a separable 5x5 kernel (radius=2) with repeat_edge boundary.
//
// Input:  Buffer<uint8_t, 2> — width x height (single channel)
// Output: Buffer<uint8_t, 2> — same dimensions
int gaussian_blur_gray(Halide::Runtime::Buffer<uint8_t>& input,
                       Halide::Runtime::Buffer<uint8_t>& output);

// Gaussian blur on a 3-channel RGB interleaved image.
// Same separable 5x5 kernel, applied independently to each channel.
//
// Input:  Buffer<uint8_t, 3> — width x height x 3 (interleaved)
// Output: Buffer<uint8_t, 3> — same dimensions and layout
int gaussian_blur_rgb(Halide::Runtime::Buffer<uint8_t>& input,
                      Halide::Runtime::Buffer<uint8_t>& output);

// Lens blur (bokeh effect) using a disc-shaped kernel.
// Averages all pixels within a circular region of the given radius.
// Uses constant_exterior boundary (black outside image bounds).
//
// Input:  Buffer<uint8_t, 3> — width x height x 3 (RGB interleaved)
// radius: Blur radius in pixels (must be <= 8, the compile-time max_radius)
// Output: Buffer<uint8_t, 3> — same dimensions and layout
int lens_blur(Halide::Runtime::Buffer<uint8_t>& input,
              int radius,
              Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Resize Operations
// ---------------------------------------------------------------------------

// Bilinear resize using scale factors.
// Uses pixel-center alignment: src = (out + 0.5) / scale - 0.5
//
// Input:   Buffer<uint8_t, 3> — source RGB interleaved image
// scale_x: output_width / input_width  (e.g., 2.0 = double width)
// scale_y: output_height / input_height
// Output:  Buffer<uint8_t, 3> — caller must allocate at target dimensions
int resize_bilinear(Halide::Runtime::Buffer<uint8_t>& input,
                    float scale_x, float scale_y,
                    Halide::Runtime::Buffer<uint8_t>& output);

// Bicubic resize (Catmull-Rom, alpha=-0.5) using scale factors.
// Higher quality than bilinear (4x4 tap vs 2x2), but ~2x slower.
// Implemented as separable 2-pass (horizontal then vertical).
//
// Parameters same as resize_bilinear.
int resize_bicubic(Halide::Runtime::Buffer<uint8_t>& input,
                   float scale_x, float scale_y,
                   Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Fixed Rotation — clockwise direction
// ---------------------------------------------------------------------------
// All fixed rotations are pure index remapping (no interpolation).
// Output dimensions swap for 90/270: (W x H) input -> (H x W) output.

// Rotate 90 degrees clockwise. Output(x,y) = Input(y, H-1-x).
int rotate_90cw(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& output);

// Rotate 180 degrees. Output(x,y) = Input(W-1-x, H-1-y).
int rotate_180(Halide::Runtime::Buffer<uint8_t>& input,
               Halide::Runtime::Buffer<uint8_t>& output);

// Rotate 270 degrees clockwise. Output(x,y) = Input(W-1-y, x).
int rotate_270cw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Counter-clockwise rotation convenience aliases
// ---------------------------------------------------------------------------
// These map to the CW equivalents: 90 CCW = 270 CW, 270 CCW = 90 CW.

// Rotate 90 degrees counter-clockwise (= 270 CW).
int rotate_90ccw(Halide::Runtime::Buffer<uint8_t>& input,
                 Halide::Runtime::Buffer<uint8_t>& output);

// Rotate 270 degrees counter-clockwise (= 90 CW).
int rotate_270ccw(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output);

// Backward-compatible alias for rotate_90cw.
int rotate_90(Halide::Runtime::Buffer<uint8_t>& input,
              Halide::Runtime::Buffer<uint8_t>& output);

// Arbitrary angle rotation with bilinear interpolation.
// Rotates around the image center. Out-of-bounds pixels become black.
//
// Input:     Buffer<uint8_t, 3> — RGB interleaved
// angle_rad: Rotation angle in radians (positive = counter-clockwise)
// Output:    Buffer<uint8_t, 3> — same dimensions as input
int rotate_angle(Halide::Runtime::Buffer<uint8_t>& input,
                 float angle_rad,
                 Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// RGB to NV21 (inverse color space conversion)
// ---------------------------------------------------------------------------

// Convert RGB to NV21 using BT.601 limited-range forward transform.
// Produces separate Y and UV planes. UV is subsampled by averaging
// each 2x2 pixel block's chroma values.
//
// Input:     Buffer<uint8_t, 3> — width x height x 3 (RGB interleaved)
// y_output:  Buffer<uint8_t, 2> — width x height
// uv_output: Buffer<uint8_t, 2> — width x (height/2) raw bytes (V,U pairs)
int rgb_to_nv21(Halide::Runtime::Buffer<uint8_t>& input,
                Halide::Runtime::Buffer<uint8_t>& y_output,
                Halide::Runtime::Buffer<uint8_t>& uv_output);

// ---------------------------------------------------------------------------
// Area Resize (INTER_AREA — box-filter downsampling)
// ---------------------------------------------------------------------------

// INTER_AREA resize using scale factors.
// Optimal for downscaling — each output pixel is a weighted average of all
// source pixels whose footprint overlaps it (box filter). Separable 2-pass.
// Supports up to 8x downscale (compile-time max_kernel=8).
//
// scale_x/scale_y: output_dim / input_dim (< 1.0 for downscale)
int resize_area(Halide::Runtime::Buffer<uint8_t>& input,
                float scale_x, float scale_y,
                Halide::Runtime::Buffer<uint8_t>& output);

// Aspect-ratio-preserving resize with black padding (letterbox/pillarbox).
// Computes a uniform scale so the entire image fits within target dimensions
// without cropping. Centers the image; fills remaining space with black (0).
//
// target_w, target_h: Desired output dimensions in pixels
int resize_letterbox(Halide::Runtime::Buffer<uint8_t>& input,
                     int target_w, int target_h,
                     Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Target-size resize APIs
// ---------------------------------------------------------------------------
// These take explicit pixel dimensions instead of scale factors.
// Avoids float precision loss from intermediate scale factor computation.
// The coordinate mapping is: src = (out + 0.5) * src_dim / target_dim - 0.5

// Bilinear resize to exact target dimensions.
int resize_bilinear_target(Halide::Runtime::Buffer<uint8_t>& input,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output);

// Bicubic (Catmull-Rom) resize to exact target dimensions.
int resize_bicubic_target(Halide::Runtime::Buffer<uint8_t>& input,
                          int target_w, int target_h,
                          Halide::Runtime::Buffer<uint8_t>& output);

// INTER_AREA resize to exact target dimensions.
int resize_area_target(Halide::Runtime::Buffer<uint8_t>& input,
                       int target_w, int target_h,
                       Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Flip Operations (3-channel RGB interleaved)
// ---------------------------------------------------------------------------

// Horizontal flip (mirror left-right). Output(x,y) = Input(W-1-x, y).
int flip_horizontal(Halide::Runtime::Buffer<uint8_t>& input,
                    Halide::Runtime::Buffer<uint8_t>& output);

// Vertical flip (mirror top-bottom). Output(x,y) = Input(x, H-1-y).
int flip_vertical(Halide::Runtime::Buffer<uint8_t>& input,
                  Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Fused NV21 Pipelines
// ---------------------------------------------------------------------------
// These perform NV21 -> Rotate -> [Flip] -> Resize -> RGB in a single pass.
//
// Benefits over chaining individual operations:
//   1. Single interpolation — no quality loss from double-interpolation
//   2. Single memory pass — read NV21 once, write RGB once (cache friendly)
//   3. Maximum fusion — Halide optimizer schedules the entire pipeline
//
// Each rotation angle (0, 90, 180, 270) is a separate AOT-compiled variant
// because rotation_code is a compile-time GeneratorParam. The dispatch
// functions in halide_ops.cpp select the correct variant at runtime.
//
// rotation_degrees_cw: Must be exactly 0, 90, 180, or 270 (returns -1 otherwise)
// flip_code: 0=none, 1=horizontal flip, 2=vertical flip

// Fused NV21 -> Rotate -> Resize (Bilinear) -> RGB
int nv21_rotate_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                           Halide::Runtime::Buffer<uint8_t>& uv_plane,
                           int rotation_degrees_cw,
                           int target_w, int target_h,
                           Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Flip -> Resize (Bilinear) -> RGB
int nv21_rotate_flip_resize_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw, int flip_code,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Resize (INTER_AREA) -> RGB
int nv21_rotate_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                int rotation_degrees_cw,
                                int target_w, int target_h,
                                Halide::Runtime::Buffer<uint8_t>& output);

// Fused NV21 -> Rotate -> Flip -> Resize (INTER_AREA) -> RGB
int nv21_rotate_flip_resize_area_rgb(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                     Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                     int rotation_degrees_cw, int flip_code,
                                     int target_w, int target_h,
                                     Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Fused NV21 -> Resize -> Pad -> Rotate (ML Preprocessing)
// ---------------------------------------------------------------------------
// Produces a square RGB output (target_size x target_size x 3).
// Uses full-range BT.601, bilinear resize, aspect-ratio-preserving
// padding (letterbox), and optional rotation.
//
// rotation_degrees_cw: Must be exactly 0, 90, 180, or 270 (returns -1 otherwise)
// target_size: side length of the square output
int nv21_resize_pad_rotate(Halide::Runtime::Buffer<uint8_t>& y_plane,
                           Halide::Runtime::Buffer<uint8_t>& uv_plane,
                           int rotation_degrees_cw,
                           int target_size,
                           Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Segmentation Post-processing
// ---------------------------------------------------------------------------

// Argmax across class logits for semantic segmentation.
//
// This is the only float-input operation in halide_ops. Segmentation models
// (e.g., DeepLab, U-Net) output float32 logits per class per pixel.
// This function finds the class with the highest logit (argmax) for each pixel.
//
// Softmax is intentionally skipped: since exp() is monotonically increasing,
// argmax(softmax(x)) == argmax(x), so we save the expensive exp() and division.
//
// Input:  Buffer<float, 3> — planar float32, shape (width, height, num_classes)
//         Planar layout: each class is a separate contiguous 2D plane.
//         [plane_0: H*W floats][plane_1: H*W floats]...[plane_7: H*W floats]
//
// Output: Buffer<uint8_t, 2> — shape (width, height)
//         Each pixel contains the winning class index (0 to num_classes-1).
//         Tie-breaking: lowest index wins (strict > comparison).
//
// Returns 0 on success, non-zero on Halide runtime error.
int seg_argmax(Halide::Runtime::Buffer<float>& input,
               Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Optimized Operations
// ---------------------------------------------------------------------------
// These are optimized versions of existing operations with improved scheduling,
// wider vectorization, better tiling, and OpenCV-matching algorithms.

// --- Optimized Color Space Conversions ---

// NV21 -> RGB optimized: tiled Y-axis, UV prefetching, bounds check elimination
int nv21_to_rgb_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                          Halide::Runtime::Buffer<uint8_t>& uv_plane,
                          Halide::Runtime::Buffer<uint8_t>& output);

// RGB -> NV21 optimized: compute_at for Cr/Cb reuse, tiled scheduling
int rgb_to_nv21_optimized(Halide::Runtime::Buffer<uint8_t>& input,
                          Halide::Runtime::Buffer<uint8_t>& y_output,
                          Halide::Runtime::Buffer<uint8_t>& uv_output);

// RGB <-> BGR optimized: wider vectorization (32px), multi-row tiling
int rgb_bgr_optimized(Halide::Runtime::Buffer<uint8_t>& input,
                      Halide::Runtime::Buffer<uint8_t>& output);

// --- Optimized RGB-Domain Resize (target-size) ---

// Bilinear: larger tiles, prefetching, bounds check elimination
int resize_bilinear_optimized(Halide::Runtime::Buffer<uint8_t>& input,
                              int target_w, int target_h,
                              Halide::Runtime::Buffer<uint8_t>& output);

// INTER_AREA: wider vectors (16 vs 8), larger tiles (64 vs 32)
int resize_area_optimized(Halide::Runtime::Buffer<uint8_t>& input,
                          int target_w, int target_h,
                          Halide::Runtime::Buffer<uint8_t>& output);

// Bicubic: a=-0.75 matching OpenCV (not Catmull-Rom a=-0.5)
int resize_bicubic_optimized(Halide::Runtime::Buffer<uint8_t>& input,
                             int target_w, int target_h,
                             Halide::Runtime::Buffer<uint8_t>& output);

// --- NV21-Domain Resize (output stays in NV21 format) ---
// Resizes Y plane at full resolution and UV plane at half resolution
// independently, avoiding the NV21->RGB->resize roundtrip.

int nv21_resize_nearest_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                  Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                  int target_w, int target_h,
                                  Halide::Runtime::Buffer<uint8_t>& y_output,
                                  Halide::Runtime::Buffer<uint8_t>& uv_output);

int nv21_resize_bilinear_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                   Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                   int target_w, int target_h,
                                   Halide::Runtime::Buffer<uint8_t>& y_output,
                                   Halide::Runtime::Buffer<uint8_t>& uv_output);

int nv21_resize_area_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                               Halide::Runtime::Buffer<uint8_t>& uv_plane,
                               int target_w, int target_h,
                               Halide::Runtime::Buffer<uint8_t>& y_output,
                               Halide::Runtime::Buffer<uint8_t>& uv_output);

int nv21_resize_bicubic_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                  Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                  int target_w, int target_h,
                                  Halide::Runtime::Buffer<uint8_t>& y_output,
                                  Halide::Runtime::Buffer<uint8_t>& uv_output);

// --- Fused NV21 Resize -> RGB Pipelines ---
// Resize in NV21 domain then convert to RGB in a single pass.

int nv21_resize_rgb_bilinear_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                       Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                       int target_w, int target_h,
                                       Halide::Runtime::Buffer<uint8_t>& output);

int nv21_resize_rgb_area_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                   Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                   int target_w, int target_h,
                                   Halide::Runtime::Buffer<uint8_t>& output);

int nv21_resize_rgb_bicubic_optimized(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                      Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                      int target_w, int target_h,
                                      Halide::Runtime::Buffer<uint8_t>& output);

// --- Fused NV21 Resize -> RGB (BT.709 full-range) ---
// Matches Samsung Camera2 HD+ colorspace. Coefficients 403/-48/-120/475 @ Q8
// agree with bt709::nv21_to_rgb_bt709_full_range_* (bt709_neon_ref.cpp).

int nv21_resize_rgb_bt709_nearest(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                  Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                  int target_w, int target_h,
                                  Halide::Runtime::Buffer<uint8_t>& output);

int nv21_resize_rgb_bt709_bilinear(Halide::Runtime::Buffer<uint8_t>& y_plane,
                                   Halide::Runtime::Buffer<uint8_t>& uv_plane,
                                   int target_w, int target_h,
                                   Halide::Runtime::Buffer<uint8_t>& output);

int nv21_resize_rgb_bt709_area(Halide::Runtime::Buffer<uint8_t>& y_plane,
                               Halide::Runtime::Buffer<uint8_t>& uv_plane,
                               int target_w, int target_h,
                               Halide::Runtime::Buffer<uint8_t>& output);

// ---------------------------------------------------------------------------
// Segmentation-Guided Pipelines
// ---------------------------------------------------------------------------

// Portrait mode: seg-guided selective disc blur with feathered alpha blending.
// Keeps foreground (fg_class) sharp, blurs background with disc bokeh kernel.
//
// input:        RGB interleaved (W × H × 3)
// seg_mask:     uint8 class indices, any resolution (bilinear-upsampled internally)
// fg_class:     which class is foreground (e.g., 15 = "person")
// blur_radius:  disc blur radius (0 = no blur, <= 12 default max)
// edge_softness: feathering control (1.0 = linear, 3-5 = natural, >10 = hard)
// output:       RGB interleaved (same size as input)
int seg_portrait_blur(Halide::Runtime::Buffer<uint8_t>& input,
                      Halide::Runtime::Buffer<uint8_t>& seg_mask,
                      int fg_class, int blur_radius, float edge_softness,
                      Halide::Runtime::Buffer<uint8_t>& output);

// Background replacement: composites foreground onto arbitrary background image.
// Uses seg_mask to determine foreground regions with feathered edges.
//
// fg_image:     Camera frame, RGB interleaved (W × H × 3)
// bg_image:     Replacement background, RGB interleaved, any resolution
//               (bilinear-resized to match fg internally)
// seg_mask:     uint8 class indices, any resolution
// fg_class:     foreground class index
// edge_softness: feathering control
// output:       RGB interleaved (same size as fg_image)
int seg_bg_replace(Halide::Runtime::Buffer<uint8_t>& fg_image,
                   Halide::Runtime::Buffer<uint8_t>& bg_image,
                   Halide::Runtime::Buffer<uint8_t>& seg_mask,
                   int fg_class, float edge_softness,
                   Halide::Runtime::Buffer<uint8_t>& output);

// Selective color grading: per-class LUT-based color transform.
// Each segmentation class gets its own linear color grade (gain, bias, blend).
//
// input:        RGB interleaved (W × H × 3)
// seg_mask:     uint8 class indices, any resolution (nearest-neighbor upsampled)
// color_lut:    float LUT, shape (num_classes, 7)
//               Per class: [R_gain, G_gain, B_gain, R_bias, G_bias, B_bias, blend_alpha]
//               Identity: gain=1, bias=0, alpha=1. Desaturate: gain=0.3, alpha=0.8.
// output:       RGB interleaved (same size as input)
int seg_color_style(Halide::Runtime::Buffer<uint8_t>& input,
                    Halide::Runtime::Buffer<uint8_t>& seg_mask,
                    Halide::Runtime::Buffer<float>& color_lut,
                    Halide::Runtime::Buffer<uint8_t>& output);

// Depth-map guided multi-kernel blur: continuous depth-of-field simulation.
// Applies different disc blur radii to different depth zones based on a
// continuous depth map. Supports up to 5 depth layers with distinct radii.
//
// input:         RGB interleaved (W x H x 3)
// depth_map:     uint8 continuous depth (0-255), any resolution (bilinear-upsampled)
// kernel_config: float buffer (num_kernels x 3): [min_depth_norm, max_depth_norm, radius]
//                Depth values normalized to [0,1]. Kernels sorted by min_depth.
// num_kernels:   number of active kernel entries (<= 5)
// output:        RGB interleaved (same size as input)
int seg_depth_blur(Halide::Runtime::Buffer<uint8_t>& input,
                   Halide::Runtime::Buffer<uint8_t>& depth_map,
                   Halide::Runtime::Buffer<float>& kernel_config,
                   int num_kernels,
                   Halide::Runtime::Buffer<uint8_t>& output);

} // namespace halide_ops
