LOCAL_PATH := $(call my-dir)

# ---------------------------------------------------------------
# Paths to external SDKs
# ---------------------------------------------------------------
HALIDE_GEN_DIR := $(LOCAL_PATH)/../../../../halide/generated/arm64-v8a
HALIDE_INCLUDE := $(LOCAL_PATH)/../../../../halide/Halide-21.0.0/include
OPENCV_VERSION ?= 3.4.16
OPENCV_ANDROID_SDK := $(LOCAL_PATH)/../../../../opencv/$(OPENCV_VERSION)/OpenCV-android-sdk

# ---------------------------------------------------------------
# Import OpenCV as STATIC library.
# Version selected via OPENCV_VERSION (default 3.4.16).
# OpenCV.mk calls CLEAR_VARS internally, so LOCAL_MODULE etc.
# must be set AFTER this include. Use += to append to OpenCV's variables.
# ---------------------------------------------------------------
OPENCV_LIB_TYPE := STATIC
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# GoogleTest test executable
# ---------------------------------------------------------------
LOCAL_MODULE := halide_tests

LOCAL_SRC_FILES := \
    ../test_nv21_to_rgb.cpp \
    ../test_rgb_to_nv21.cpp \
    ../test_rgb_bgr.cpp \
    ../test_gaussian_blur.cpp \
    ../test_lens_blur.cpp \
    ../test_resize.cpp \
    ../test_resize_target.cpp \
    ../test_rotate.cpp \
    ../test_flip.cpp \
    ../test_fused_pipeline.cpp \
    ../test_target_dispatch.cpp \
    ../test_nv21_full_range.cpp \
    ../test_nv21_yuv444.cpp \
    ../test_nv21_resize_pad_rotate.cpp \
    ../test_seg_argmax.cpp \
    ../test_rgb_resize_optimized.cpp \
    ../test_nv21_resize_optimized.cpp \
    ../test_nv21_rgb_optimized.cpp \
    ../test_rgb_bgr_optimized.cpp \
    ../test_nv21_resize_rgb_optimized.cpp \
    ../test_bench_optimized.cpp \
    ../../main/jni/halide_ops.cpp

LOCAL_C_INCLUDES += \
    $(HALIDE_INCLUDE) \
    $(HALIDE_GEN_DIR) \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../../main/jni

LOCAL_LDLIBS += -llog -lm

# Link all Halide AOT-generated static libraries.
# halide_runtime.a MUST come LAST: pipelines reference runtime symbols,
# and the linker only pulls .o from archives that resolve current undefs.
LOCAL_LDFLAGS += \
    $(HALIDE_GEN_DIR)/rgb_bgr_convert.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb.a \
    $(HALIDE_GEN_DIR)/rgb_to_nv21.a \
    $(HALIDE_GEN_DIR)/gaussian_blur_y.a \
    $(HALIDE_GEN_DIR)/gaussian_blur_rgb.a \
    $(HALIDE_GEN_DIR)/lens_blur.a \
    $(HALIDE_GEN_DIR)/resize_bilinear.a \
    $(HALIDE_GEN_DIR)/resize_bicubic.a \
    $(HALIDE_GEN_DIR)/resize_bilinear_target.a \
    $(HALIDE_GEN_DIR)/resize_bicubic_target.a \
    $(HALIDE_GEN_DIR)/resize_area.a \
    $(HALIDE_GEN_DIR)/resize_area_target.a \
    $(HALIDE_GEN_DIR)/resize_letterbox.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_90cw.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_180.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_270cw.a \
    $(HALIDE_GEN_DIR)/rotate_arbitrary.a \
    $(HALIDE_GEN_DIR)/flip_horizontal.a \
    $(HALIDE_GEN_DIR)/flip_vertical.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_none.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_90cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_180.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_270cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_none.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_90cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_180.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_270cw.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb_full_range.a \
    $(HALIDE_GEN_DIR)/nv21_yuv444_rgb.a \
    $(HALIDE_GEN_DIR)/nv21_resize_pad_rotate_none.a \
    $(HALIDE_GEN_DIR)/nv21_resize_pad_rotate_90cw.a \
    $(HALIDE_GEN_DIR)/nv21_resize_pad_rotate_180.a \
    $(HALIDE_GEN_DIR)/nv21_resize_pad_rotate_270cw.a \
    $(HALIDE_GEN_DIR)/seg_argmax.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb_optimized.a \
    $(HALIDE_GEN_DIR)/rgb_to_nv21_optimized.a \
    $(HALIDE_GEN_DIR)/rgb_bgr_optimized.a \
    $(HALIDE_GEN_DIR)/resize_bilinear_optimized.a \
    $(HALIDE_GEN_DIR)/resize_area_optimized.a \
    $(HALIDE_GEN_DIR)/resize_bicubic_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_bilinear_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_area_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_bicubic_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_bilinear_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_area_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_bicubic_optimized.a \
    $(HALIDE_GEN_DIR)/halide_runtime.a

# GoogleTest requires RTTI
LOCAL_CPP_FEATURES := rtti exceptions

LOCAL_STATIC_LIBRARIES += googletest_main

include $(BUILD_EXECUTABLE)

$(call import-module,third_party/googletest)
