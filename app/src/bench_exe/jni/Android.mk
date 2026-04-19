LOCAL_PATH := $(call my-dir)

# ---------------------------------------------------------------
# Paths to external SDKs
# ---------------------------------------------------------------
HALIDE_GEN_DIR := $(LOCAL_PATH)/../../../../halide/generated/arm64-v8a
HALIDE_INCLUDE := $(LOCAL_PATH)/../../../../halide/Halide-21.0.0/include
OPENCV_VERSION ?= 3.4.16
OPENCV_ANDROID_SDK := $(LOCAL_PATH)/../../../../opencv/$(OPENCV_VERSION)/OpenCV-android-sdk

# ---------------------------------------------------------------
# Import OpenCV (STATIC). OpenCV.mk calls CLEAR_VARS internally.
# ---------------------------------------------------------------
OPENCV_LIB_TYPE := STATIC
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# Standalone benchmark executable: bench
# ---------------------------------------------------------------
LOCAL_MODULE := bench

LOCAL_SRC_FILES := \
    ../bench_main.cpp \
    ../bench_stress.cpp \
    ../../main/jni/bt709_neon_ref.cpp

LOCAL_C_INCLUDES += \
    $(HALIDE_INCLUDE) \
    $(HALIDE_GEN_DIR) \
    $(LOCAL_PATH)/..

LOCAL_LDLIBS += -llog -lm

# Link every AOT Halide pipeline we dispatch from bench_main.cpp.
# halide_runtime.a MUST come LAST — the linker only pulls .o from archives
# that resolve an already-undefined symbol, and the runtime symbols are
# referenced by the pipeline objects.
LOCAL_LDFLAGS += \
    $(HALIDE_GEN_DIR)/rotate_fixed_90cw.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_180.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_270cw.a \
    $(HALIDE_GEN_DIR)/rotate_arbitrary.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_1c_90cw.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_1c_180.a \
    $(HALIDE_GEN_DIR)/rotate_fixed_1c_270cw.a \
    $(HALIDE_GEN_DIR)/nv21_to_rgb_bt709_full_range.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_bt709_nearest.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_bt709_bilinear.a \
    $(HALIDE_GEN_DIR)/nv21_resize_rgb_bt709_area.a \
    $(HALIDE_GEN_DIR)/nv21_resize_bilinear_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_area_optimized.a \
    $(HALIDE_GEN_DIR)/nv21_resize_nearest_optimized.a \
    $(HALIDE_GEN_DIR)/halide_runtime.a

# RTTI off, exceptions off (we don't throw anywhere).
LOCAL_CPP_FEATURES :=

include $(BUILD_EXECUTABLE)
