#!/bin/bash
# Compile Halide generators on host, then AOT cross-compile for Android arm64-v8a.
#
# Three-stage pipeline:
# 1. Host compile: generator source + libGenGen.a -> host executable
# 2. AOT cross-compile: host executable -> .a (static lib) + .h (header) for arm-64-android
#    Multi-target: high-feature (armv82a+dot_prod+fp16) with baseline fallback
# 3. Generate standalone Halide runtime for runtime feature dispatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HALIDE_DIR="${SCRIPT_DIR}/Halide-21.0.0"
GEN_SRC_DIR="${SCRIPT_DIR}/generators"
OUTPUT_DIR="${SCRIPT_DIR}/generated/arm64-v8a"
BIN_DIR="${SCRIPT_DIR}/bin"

HALIDE_INCLUDE="${HALIDE_DIR}/include"
HALIDE_LIB="${HALIDE_DIR}/lib"
GENGEN_LIB="${HALIDE_DIR}/lib/libHalide_GenGen.a"

# Multi-target: most-featured first (checked first at runtime via getauxval)
HL_TARGET_HIGH="arm-64-android-armv82a-arm_dot_prod-arm_fp16-no_runtime"
HL_TARGET_BASE="arm-64-android-no_runtime"
HL_MULTI_TARGET="${HL_TARGET_HIGH},${HL_TARGET_BASE}"
HL_SINGLE_TARGET="arm-64-android-no_runtime"

# Verify Halide SDK exists
if [ ! -f "${HALIDE_INCLUDE}/Halide.h" ]; then
    echo "ERROR: Halide SDK not found at ${HALIDE_DIR}"
    echo "Run scripts/setup_halide.sh first."
    exit 1
fi

if [ ! -f "${GENGEN_LIB}" ]; then
    echo "ERROR: libHalide_GenGen.a not found at ${GENGEN_LIB}"
    echo "Halide v21+ requires libHalide_GenGen.a. Run scripts/setup_halide.sh first."
    exit 1
fi

mkdir -p "${BIN_DIR}" "${OUTPUT_DIR}"

CXX="${CXX:-c++}"
CXXFLAGS="-std=c++17 -fno-rtti -Wall -O2"

# OS-specific whole-archive flag for libGenGen.a
OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Darwin)
        GENGEN_LINK="-Wl,-force_load,${GENGEN_LIB}"
        ;;
    *)
        GENGEN_LINK="-Wl,--whole-archive ${GENGEN_LIB} -Wl,--no-whole-archive"
        ;;
esac

# On macOS, we need to set DYLD_LIBRARY_PATH for the host executables
export DYLD_LIBRARY_PATH="${HALIDE_LIB}:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${HALIDE_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== Stage 1: Compiling generators to host executables ==="

for src in "${GEN_SRC_DIR}"/*_generator.cpp; do
    [ -f "$src" ] || continue
    name=$(basename "$src" .cpp)
    echo "  Compiling: ${name}"
    ${CXX} ${CXXFLAGS} \
        "${src}" \
        -I "${HALIDE_INCLUDE}" \
        -L "${HALIDE_LIB}" \
        -lHalide \
        ${GENGEN_LINK} \
        -lpthread -ldl -lz \
        -Wl,-rpath,"${HALIDE_LIB}" \
        -o "${BIN_DIR}/${name}"
done

echo ""
echo "=== Stage 2: AOT cross-compiling (multi-target: high + baseline) ==="

# Generators that benefit from arm_dot_prod / arm_fp16 (arithmetic-heavy)
MULTI_TARGET_RUNS=(
    "nv21_to_rgb_generator -g nv21_to_rgb -f nv21_to_rgb"
    "gaussian_blur_generator -g gaussian_blur_y -f gaussian_blur_y"
    "gaussian_blur_generator -g gaussian_blur_rgb -f gaussian_blur_rgb"
    "lens_blur_generator -g lens_blur -f lens_blur"
    "resize_generator -g resize_bilinear -f resize_bilinear"
    "resize_generator -g resize_bicubic -f resize_bicubic"
    # resize_bilinear_target and resize_bicubic_target removed:
    # replaced by resize_bilinear_optimized and resize_bicubic_optimized (faster, OpenCV-matching)
    "rotate_generator -g rotate_arbitrary -f rotate_arbitrary"
    "rgb_to_nv21_generator -g rgb_to_nv21 -f rgb_to_nv21"
    "resize_area_generator -g resize_area -f resize_area"
    # resize_area_target removed: replaced by resize_area_optimized
    "resize_letterbox_generator -g resize_letterbox -f resize_letterbox"
    # Fused NV21 pipeline — bilinear resize (4 rotation variants)
    "nv21_pipeline_generator -g nv21_pipeline_bilinear -f nv21_pipeline_bilinear_none rotation_code=0"
    "nv21_pipeline_generator -g nv21_pipeline_bilinear -f nv21_pipeline_bilinear_90cw rotation_code=1"
    "nv21_pipeline_generator -g nv21_pipeline_bilinear -f nv21_pipeline_bilinear_180 rotation_code=2"
    "nv21_pipeline_generator -g nv21_pipeline_bilinear -f nv21_pipeline_bilinear_270cw rotation_code=3"
    # Fused NV21 pipeline — INTER_AREA resize (4 rotation variants)
    "nv21_pipeline_generator -g nv21_pipeline_area -f nv21_pipeline_area_none rotation_code=0"
    "nv21_pipeline_generator -g nv21_pipeline_area -f nv21_pipeline_area_90cw rotation_code=1"
    "nv21_pipeline_generator -g nv21_pipeline_area -f nv21_pipeline_area_180 rotation_code=2"
    "nv21_pipeline_generator -g nv21_pipeline_area -f nv21_pipeline_area_270cw rotation_code=3"
    # Full-range BT.601 NV21 to RGB (Samsung/Android Camera)
    "nv21_to_rgb_full_range_generator -g nv21_to_rgb_full_range -f nv21_to_rgb_full_range"
    # NV21 -> YUV444 (bilinear UV upsample) -> RGB
    "nv21_yuv444_rgb_generator -g nv21_yuv444_rgb -f nv21_yuv444_rgb"
    # Fused NV21 -> Resize -> RGB -> Pad -> Rotate (4 rotation variants)
    "nv21_resize_pad_rotate_generator -g nv21_resize_pad_rotate -f nv21_resize_pad_rotate_none rotation_code=0"
    "nv21_resize_pad_rotate_generator -g nv21_resize_pad_rotate -f nv21_resize_pad_rotate_90cw rotation_code=1"
    "nv21_resize_pad_rotate_generator -g nv21_resize_pad_rotate -f nv21_resize_pad_rotate_180 rotation_code=2"
    "nv21_resize_pad_rotate_generator -g nv21_resize_pad_rotate -f nv21_resize_pad_rotate_270cw rotation_code=3"
    # Segmentation post-processing: argmax across class logits
    "seg_argmax_generator -g seg_argmax -f seg_argmax num_classes=8"
    # --- Optimized generators ---
    # RGB resize optimized (target-size, wider vectors, better tiling)
    "rgb_resize_optimized_generator -g resize_bilinear_optimized -f resize_bilinear_optimized"
    "rgb_resize_optimized_generator -g resize_area_optimized -f resize_area_optimized"
    # INTER_AREA integer ratio fast paths (2x, 3x, 4x block average)
    "rgb_resize_optimized_generator -g resize_area_nx -f resize_area_2x ratio=2"
    "rgb_resize_optimized_generator -g resize_area_nx -f resize_area_3x ratio=3"
    "rgb_resize_optimized_generator -g resize_area_nx -f resize_area_4x ratio=4"
    "rgb_resize_optimized_generator -g resize_bicubic_optimized -f resize_bicubic_optimized"
    # NV21 resize optimized (resize directly in NV21 domain)
    "nv21_resize_optimized_generator -g nv21_resize_bilinear_optimized -f nv21_resize_bilinear_optimized"
    "nv21_resize_optimized_generator -g nv21_resize_area_optimized -f nv21_resize_area_optimized"
    "nv21_resize_optimized_generator -g nv21_resize_bicubic_optimized -f nv21_resize_bicubic_optimized"
    # nv21_to_rgb_optimized and rgb_to_nv21_optimized removed:
    # baseline versions are faster on device (benchmark data: 624 vs 697 us, 1998 vs 2107 us)
    # Fused NV21 resize -> RGB optimized
    "nv21_resize_rgb_optimized_generator -g nv21_resize_rgb_bilinear_optimized -f nv21_resize_rgb_bilinear_optimized"
    "nv21_resize_rgb_optimized_generator -g nv21_resize_rgb_area_optimized -f nv21_resize_rgb_area_optimized"
    "nv21_resize_rgb_optimized_generator -g nv21_resize_rgb_bicubic_optimized -f nv21_resize_rgb_bicubic_optimized"
    # --- Segmentation-guided pipelines ---
    # Portrait mode: seg-guided selective disc blur with alpha blending
    "seg_portrait_blur_generator -g seg_portrait_blur -f seg_portrait_blur"
    # Background replacement: seg-guided foreground/background compositing
    "seg_bg_replace_generator -g seg_bg_replace -f seg_bg_replace"
    # Selective color grading: per-class LUT-based color transform
    "seg_color_style_generator -g seg_color_style -f seg_color_style num_classes=8"
    # Depth-map guided multi-kernel blur: continuous depth-of-field simulation
    "seg_depth_blur_generator -g seg_depth_blur -f seg_depth_blur"
)

# Generators with no arithmetic benefit (pure index/channel remapping)
SINGLE_TARGET_RUNS=(
    # rgb_bgr_convert removed: replaced by rgb_bgr_optimized (1.05x faster, identical output)
    # Fixed rotations — all 3 variants (90CW, 180, 270CW)
    "rotate_generator -g rotate_fixed -f rotate_fixed_90cw rotation_code=1"
    "rotate_generator -g rotate_fixed -f rotate_fixed_180 rotation_code=2"
    "rotate_generator -g rotate_fixed -f rotate_fixed_270cw rotation_code=3"
    # Flip — horizontal and vertical
    "flip_generator -g flip_fixed -f flip_horizontal flip_code=0"
    "flip_generator -g flip_fixed -f flip_vertical flip_code=1"
    # RGB <-> BGR optimized (wider vectors, multi-row tiling)
    "rgb_bgr_optimized_generator -g rgb_bgr_optimized -f rgb_bgr_optimized"
)

for run in "${MULTI_TARGET_RUNS[@]}"; do
    read -ra parts <<< "$run"
    gen_name="${parts[0]}"
    args=("${parts[@]:1}")
    echo "  [multi-target] ${args[*]}"
    "${BIN_DIR}/${gen_name}" "${args[@]}" \
        -e static_library,h,registration \
        -o "${OUTPUT_DIR}" \
        target="${HL_MULTI_TARGET}"
done

for run in "${SINGLE_TARGET_RUNS[@]}"; do
    read -ra parts <<< "$run"
    gen_name="${parts[0]}"
    args=("${parts[@]:1}")
    echo "  [single-target] ${args[*]}"
    "${BIN_DIR}/${gen_name}" "${args[@]}" \
        -e static_library,h,registration \
        -o "${OUTPUT_DIR}" \
        target="${HL_SINGLE_TARGET}"
done

echo ""
echo "=== Stage 3: Generating standalone Halide runtime ==="
"${BIN_DIR}/rgb_bgr_generator" \
    -r halide_runtime \
    -e static_library \
    -o "${OUTPUT_DIR}" \
    target=arm-64-android
echo "  Generated halide_runtime.a"

echo ""
echo "=== Done ==="
echo "Generated artifacts in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"/*.h "${OUTPUT_DIR}"/*.a 2>/dev/null || echo "(no files found)"
