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

    /**
     * Append a CSV line to a file on device storage.
     */
    public static native void appendCsv(String filePath, String csvLine);
}
