LOCAL_PATH := $(call my-dir)

# ---------------------------------------------------------------
# Paths to external SDKs
# ---------------------------------------------------------------
HALIDE_GEN_DIR := $(LOCAL_PATH)/../../../../halide/generated/arm64-v8a
HALIDE_INCLUDE := $(LOCAL_PATH)/../../../../halide/Halide-18.0.0/include

OPENCV_ANDROID_SDK := $(LOCAL_PATH)/../../../../opencv/OpenCV-android-sdk
OPENCV_LIB_TYPE := STATIC
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# GoogleTest test executable
# ---------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := halide_tests

LOCAL_SRC_FILES := \
    ../test_nv21_to_rgb.cpp \
    ../test_rgb_to_nv21.cpp \
    ../test_rgb_bgr.cpp \
    ../test_gaussian_blur.cpp \
    ../test_lens_blur.cpp \
    ../test_resize.cpp \
    ../test_rotate.cpp

LOCAL_C_INCLUDES := \
    $(HALIDE_INCLUDE) \
    $(HALIDE_GEN_DIR) \
    $(OPENCV_ANDROID_SDK)/sdk/native/jni/include \
    $(LOCAL_PATH)/..

LOCAL_LDLIBS := -llog -lm

# Link all Halide AOT-generated static libraries
LOCAL_LDFLAGS := \
    $(HALIDE_GEN_DIR)/rgb_bgr_convert.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb.a \
    $(HALIDE_GEN_DIR)/rgb_to_nv21.a \
    $(HALIDE_GEN_DIR)/gaussian_blur_y.a \
    $(HALIDE_GEN_DIR)/gaussian_blur_rgb.a \
    $(HALIDE_GEN_DIR)/lens_blur.a \
    $(HALIDE_GEN_DIR)/resize_bilinear.a \
    $(HALIDE_GEN_DIR)/resize_bicubic.a \
    $(HALIDE_GEN_DIR)/resize_area.a \
    $(HALIDE_GEN_DIR)/rotate_fixed.a \
    $(HALIDE_GEN_DIR)/rotate_arbitrary.a

# GoogleTest requires RTTI
LOCAL_CPP_FEATURES := rtti exceptions

LOCAL_STATIC_LIBRARIES := googletest_main

include $(BUILD_EXECUTABLE)

$(call import-module,third_party/googletest)
