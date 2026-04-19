package com.example.halidetest;

import android.graphics.Bitmap;

/**
 * JNI bridge to native Halide and OpenCV image processing functions.
 * Each method returns execution time in microseconds.
 */
public class NativeBridge {

    static {
        System.loadLibrary("halide_benchmark");
    }

    /**
     * RGB <-> BGR channel swap.
     * @return execution time in microseconds
     */
    public static native long rgbBgr(Bitmap inputBitmap, Bitmap outputBitmap,
                                      boolean useHalide);

    /**
     * NV21 to RGB conversion.
     * @param nv21Data raw NV21 byte array (Y plane + interleaved VU)
     * @return execution time in microseconds
     */
    public static native long nv21ToRgb(byte[] nv21Data, int width, int height,
                                         Bitmap outputBitmap, boolean useHalide);

    /**
     * Gaussian blur on 3-channel image.
     * @param kernelSize kernel size (must be odd, e.g. 5)
     * @return execution time in microseconds
     */
    public static native long gaussianBlur(Bitmap inputBitmap, Bitmap outputBitmap,
                                            int kernelSize, boolean useHalide);

    /**
     * Lens blur (disc/bokeh) on 3-channel image.
     * @param radius blur disc radius in pixels
     * @return execution time in microseconds
     */
    public static native long lensBlur(Bitmap inputBitmap, Bitmap outputBitmap,
                                        int radius, boolean useHalide);

    /**
     * Resize image.
     * @param useBicubic true for bicubic, false for bilinear
     * @return execution time in microseconds
     */
    public static native long resize(Bitmap inputBitmap, Bitmap outputBitmap,
                                      int newWidth, int newHeight,
                                      boolean useBicubic, boolean useHalide);

    /**
     * Rotate image.
     * @param angleDegrees rotation angle in degrees (90, 180, 270, or arbitrary)
     * @return execution time in microseconds
     */
    public static native long rotate(Bitmap inputBitmap, Bitmap outputBitmap,
                                      float angleDegrees, boolean useHalide);

    /**
     * RGB to NV21 conversion.
     * @param nv21Output pre-allocated byte array (w*h + w*h/2)
     * @return execution time in microseconds
     */
    public static native long rgbToNv21(Bitmap inputBitmap, byte[] nv21Output,
                                         boolean useHalide);

    /**
     * INTER_AREA resize (optimal box-filter downsampling).
     * @return execution time in microseconds
     */
    public static native long resizeArea(Bitmap inputBitmap, Bitmap outputBitmap,
                                          int newWidth, int newHeight,
                                          boolean useHalide);

    /**
     * Letterbox resize (aspect-ratio-preserving with black padding).
     * @return execution time in microseconds
     */
    public static native long resizeLetterbox(Bitmap inputBitmap, Bitmap outputBitmap,
                                               int targetWidth, int targetHeight,
                                               boolean useHalide);

    /**
     * Target-size bilinear resize (exact pixel dimensions).
     * @return execution time in microseconds
     */
    public static native long resizeBilinearTarget(Bitmap inputBitmap, Bitmap outputBitmap,
                                                    int targetWidth, int targetHeight,
                                                    boolean useHalide);

    /**
     * Target-size bicubic resize (exact pixel dimensions).
     * @return execution time in microseconds
     */
    public static native long resizeBicubicTarget(Bitmap inputBitmap, Bitmap outputBitmap,
                                                   int targetWidth, int targetHeight,
                                                   boolean useHalide);

    /**
     * Target-size INTER_AREA resize (exact pixel dimensions).
     * @return execution time in microseconds
     */
    public static native long resizeAreaTarget(Bitmap inputBitmap, Bitmap outputBitmap,
                                                int targetWidth, int targetHeight,
                                                boolean useHalide);

    /**
     * Flip image horizontally or vertically.
     * @param horizontal true for horizontal (mirror L-R), false for vertical (mirror T-B)
     * @return execution time in microseconds
     */
    public static native long flip(Bitmap inputBitmap, Bitmap outputBitmap,
                                    boolean horizontal, boolean useHalide);

    /**
     * Fused NV21 -> Rotate -> [Flip] -> Resize -> RGB pipeline.
     * @param rotationDegreesCW 0, 90, 180, or 270
     * @param flipCode 0=none, 1=horizontal, 2=vertical
     * @param useArea true for INTER_AREA, false for bilinear
     * @return execution time in microseconds
     */
    public static native long nv21RotateResizeRgb(byte[] nv21Data, int srcWidth, int srcHeight,
                                                   int rotationDegreesCW, int flipCode,
                                                   int targetWidth, int targetHeight,
                                                   boolean useArea, boolean useHalide);

    /**
     * NV21 to RGB with bilinear UV upsampling (YUV444 quality).
     * Higher quality than nv21ToRgb (which uses nearest-neighbor UV).
     * @return execution time in microseconds
     */
    public static native long nv21Yuv444Rgb(byte[] nv21Data, int width, int height,
                                             Bitmap outputBitmap, boolean useHalide);

    /**
     * NV21 to RGB using full-range BT.601 (Android Camera / JFIF).
     * @return execution time in microseconds
     */
    public static native long nv21ToRgbFullRange(byte[] nv21Data, int width, int height,
                                                  Bitmap outputBitmap, boolean useHalide);

    /**
     * Fused NV21 -> Resize -> Pad -> Rotate for ML preprocessing.
     * Produces a square RGB output (targetSize x targetSize).
     * @param rotationDegreesCW 0, 90, 180, or 270
     * @param targetSize side length of the square output
     * @return execution time in microseconds
     */
    public static native long nv21ResizePadRotate(byte[] nv21Data, int srcWidth, int srcHeight,
                                                   int rotationDegreesCW, int targetSize,
                                                   boolean useHalide);

    /**
     * Segmentation argmax on synthetic float data.
     * @param numClasses number of segmentation classes
     * @return execution time in microseconds
     */
    public static native long segArgmax(int width, int height, int numClasses,
                                         boolean useHalide);

    // ---- Optimized operations ----

    /**
     * RGB <-> BGR optimized (wider SIMD, multi-row tiles).
     * @return execution time in microseconds
     */
    public static native long rgbBgrOptimized(Bitmap inputBitmap, Bitmap outputBitmap,
                                               boolean useHalide);

    /**
     * NV21 to RGB optimized (tiled, prefetch, unsafe_promise_clamped).
     * @return execution time in microseconds
     */
    public static native long nv21ToRgbOptimized(byte[] nv21Data, int width, int height,
                                                  Bitmap outputBitmap, boolean useHalide);

    /**
     * RGB to NV21 optimized (tiled Y+UV, compute_at for chroma).
     * @return execution time in microseconds
     */
    public static native long rgbToNv21Optimized(Bitmap inputBitmap, byte[] nv21Output,
                                                  boolean useHalide);

    /**
     * RGB bilinear resize optimized (fixed-point weights, prefetch).
     * @return execution time in microseconds
     */
    public static native long resizeBilinearOptimized(Bitmap inputBitmap, Bitmap outputBitmap,
                                                       int targetWidth, int targetHeight,
                                                       boolean useHalide);

    /**
     * RGB area resize optimized (wider vectorization, better tiling).
     * @return execution time in microseconds
     */
    public static native long resizeAreaOptimized(Bitmap inputBitmap, Bitmap outputBitmap,
                                                   int targetWidth, int targetHeight,
                                                   boolean useHalide);

    /**
     * RGB bicubic resize optimized (a=-0.75 matching OpenCV).
     * @return execution time in microseconds
     */
    public static native long resizeBicubicOptimized(Bitmap inputBitmap, Bitmap outputBitmap,
                                                      int targetWidth, int targetHeight,
                                                      boolean useHalide);

    /**
     * NV21-domain bilinear resize optimized (output stays NV21).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeBilinearOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                           int targetWidth, int targetHeight,
                                                           boolean useHalide);

    /**
     * NV21-domain area resize optimized (output stays NV21).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeAreaOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                       int targetWidth, int targetHeight,
                                                       boolean useHalide);

    /**
     * NV21-domain bicubic resize optimized (output stays NV21).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeBicubicOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                          int targetWidth, int targetHeight,
                                                          boolean useHalide);

    /**
     * Fused NV21 -> bilinear resize -> RGB optimized (single pass).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbBilinearOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                              int targetWidth, int targetHeight,
                                                              boolean useHalide);

    /**
     * Fused NV21 -> area resize -> RGB optimized (single pass).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbAreaOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                          int targetWidth, int targetHeight,
                                                          boolean useHalide);

    /**
     * Fused NV21 -> bicubic resize -> RGB optimized (single pass).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbBicubicOptimized(byte[] nv21Data, int srcWidth, int srcHeight,
                                                             int targetWidth, int targetHeight,
                                                             boolean useHalide);

    /**
     * Fused NV21 -> nearest resize -> RGB (BT.709 full-range, Samsung Camera2 HD+).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbBt709Nearest(byte[] nv21Data, int srcWidth, int srcHeight,
                                                         int targetWidth, int targetHeight,
                                                         boolean useHalide);

    /**
     * Fused NV21 -> bilinear resize -> RGB (BT.709 full-range).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbBt709Bilinear(byte[] nv21Data, int srcWidth, int srcHeight,
                                                          int targetWidth, int targetHeight,
                                                          boolean useHalide);

    /**
     * Fused NV21 -> area resize -> RGB (BT.709 full-range).
     * @return execution time in microseconds
     */
    public static native long nv21ResizeRgbBt709Area(byte[] nv21Data, int srcWidth, int srcHeight,
                                                      int targetWidth, int targetHeight,
                                                      boolean useHalide);

    // ---- Segmentation-guided pipelines ----

    /**
     * Portrait mode: seg-guided disc blur with feathered alpha blending.
     * Keeps foreground sharp, blurs background with bokeh disc kernel.
     * Uses synthetic seg_mask (centered rectangle as foreground).
     * @param blurRadius disc blur radius in pixels
     * @return execution time in microseconds
     */
    public static native long segPortraitBlur(Bitmap inputBitmap, Bitmap outputBitmap,
                                               int blurRadius, boolean useHalide);

    /**
     * Background replacement: composites foreground onto arbitrary background.
     * Uses synthetic seg_mask (centered rectangle as foreground).
     * @return execution time in microseconds
     */
    public static native long segBgReplace(Bitmap inputBitmap, Bitmap bgBitmap,
                                            Bitmap outputBitmap, boolean useHalide);

    /**
     * Selective color grading: per-class LUT color transform using seg mask.
     * Uses synthetic seg_mask (striped pattern) and styled color LUT.
     * @return execution time in microseconds
     */
    public static native long segColorStyle(Bitmap inputBitmap, Bitmap outputBitmap,
                                             boolean useHalide);

    /**
     * Depth-map guided multi-kernel blur.
     * Simulates camera depth-of-field with continuous focus falloff.
     * Uses synthetic depth map (vertical gradient: top=near, bottom=far).
     * @param numKernels number of blur depth zones (max 5)
     * @return execution time in microseconds
     */
    public static native long segDepthBlur(Bitmap inputBitmap, Bitmap outputBitmap,
                                            int numKernels, boolean useHalide);

    /**
     * Append a CSV line to a file on device storage.
     */
    public static native void appendCsv(String filePath, String csvLine);

    /**
     * Native-only benchmark: allocates all buffers in native heap (malloc),
     * bypassing Java Bitmap heap limits. Supports 200MP+ resolutions.
     * @param opId operation index matching the operations string-array order
     * @param srcWidth source image width
     * @param srcHeight source image height
     * @param targetWidth target output width (for resize ops)
     * @param targetHeight target output height (for resize ops)
     * @param useHalide true for Halide, false for OpenCV
     * @return execution time in microseconds
     */
    public static native long nativeBenchmark(int opId, int srcWidth, int srcHeight,
                                               int targetWidth, int targetHeight,
                                               boolean useHalide);
}
