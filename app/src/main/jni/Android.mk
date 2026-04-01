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
# Version selected via OPENCV_VERSION (default 4.9.0).
# OpenCV.mk calls CLEAR_VARS internally, so LOCAL_MODULE etc.
# must be set AFTER this include.
# ---------------------------------------------------------------
OPENCV_LIB_TYPE := STATIC
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# Main shared library: libhalide_benchmark.so
# Set LOCAL_MODULE after OpenCV.mk since it calls CLEAR_VARS.
# OpenCV.mk already populated LOCAL_STATIC_LIBRARIES and LOCAL_C_INCLUDES.
# ---------------------------------------------------------------
LOCAL_MODULE := halide_benchmark

LOCAL_SRC_FILES := \
    native_bridge.cpp \
    halide_ops.cpp \
    opencv_ops.cpp \
    benchmark_engine.cpp

LOCAL_C_INCLUDES += \
    $(HALIDE_INCLUDE) \
    $(HALIDE_GEN_DIR)

LOCAL_LDLIBS += -llog -lm -ljnigraphics -landroid

# Link all Halide AOT-generated static libraries.
# halide_runtime.a MUST come LAST: pipelines reference runtime symbols,
# and the linker only pulls .o from archives that resolve current undefs.
LOCAL_LDFLAGS += \
    $(HALIDE_GEN_DIR)/rgb_bgr_convert.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb.a \
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
    $(HALIDE_GEN_DIR)/rgb_to_nv21.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_none.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_90cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_180.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_bilinear_270cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_none.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_90cw.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_180.a \
    $(HALIDE_GEN_DIR)/nv21_pipeline_area_270cw.a \
    $(HALIDE_GEN_DIR)/halide_runtime.a

include $(BUILD_SHARED_LIBRARY)
