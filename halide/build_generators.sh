#!/bin/bash
# Compile Halide generators on host, then AOT cross-compile for Android arm64-v8a.
#
# Two-stage pipeline:
# 1. Host compile: generator source + GenGen.cpp -> host executable
# 2. AOT cross-compile: host executable -> .a (static lib) + .h (header) for arm-64-android
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HALIDE_DIR="${SCRIPT_DIR}/Halide-18.0.0"
GENGEN="${SCRIPT_DIR}/tools/GenGen.cpp"
GEN_SRC_DIR="${SCRIPT_DIR}/generators"
OUTPUT_DIR="${SCRIPT_DIR}/generated/arm64-v8a"
BIN_DIR="${SCRIPT_DIR}/bin"

HALIDE_INCLUDE="${HALIDE_DIR}/include"
HALIDE_LIB="${HALIDE_DIR}/lib"

HL_TARGET="arm-64-android-arm_fp16"

# Verify Halide SDK exists
if [ ! -f "${HALIDE_INCLUDE}/Halide.h" ]; then
    echo "ERROR: Halide SDK not found at ${HALIDE_DIR}"
    echo "Run scripts/setup_halide.sh first."
    exit 1
fi

if [ ! -f "${GENGEN}" ]; then
    echo "ERROR: GenGen.cpp not found at ${GENGEN}"
    echo "Run scripts/setup_halide.sh first."
    exit 1
fi

mkdir -p "${BIN_DIR}" "${OUTPUT_DIR}"

CXX="${CXX:-c++}"
CXXFLAGS="-std=c++17 -fno-rtti -Wall -O2"

# On macOS, we need to set DYLD_LIBRARY_PATH for the host executables
export DYLD_LIBRARY_PATH="${HALIDE_LIB}:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${HALIDE_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== Stage 1: Compiling generators to host executables ==="

for src in "${GEN_SRC_DIR}"/*_generator.cpp; do
    [ -f "$src" ] || continue
    name=$(basename "$src" .cpp)
    echo "  Compiling: ${name}"
    ${CXX} ${CXXFLAGS} \
        "${src}" "${GENGEN}" \
        -I "${HALIDE_INCLUDE}" \
        -L "${HALIDE_LIB}" \
        -lHalide \
        -lpthread -ldl -lz \
        -Wl,-rpath,"${HALIDE_LIB}" \
        -o "${BIN_DIR}/${name}"
done

echo ""
echo "=== Stage 2: AOT cross-compiling for ${HL_TARGET} ==="

# Each entry: "generator_exe -g registered_name [-f func_name] [extra params]"
# The -g flag selects which registered generator to run
# Output: <registered_name>.a and <registered_name>.h in OUTPUT_DIR
RUNS=(
    "rgb_bgr_generator -g rgb_bgr_convert -f rgb_bgr_convert -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "nv21_to_rgb_generator -g nv21_to_rgb -f nv21_to_rgb -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "gaussian_blur_generator -g gaussian_blur_y -f gaussian_blur_y -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "gaussian_blur_generator -g gaussian_blur_rgb -f gaussian_blur_rgb -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "lens_blur_generator -g lens_blur -f lens_blur -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "resize_generator -g resize_bilinear -f resize_bilinear -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "resize_generator -g resize_bicubic -f resize_bicubic -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "rotate_generator -g rotate_fixed -f rotate_fixed -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "rotate_generator -g rotate_arbitrary -f rotate_arbitrary -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "rgb_to_nv21_generator -g rgb_to_nv21 -f rgb_to_nv21 -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "resize_area_generator -g resize_area -f resize_area -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
    "resize_letterbox_generator -g resize_letterbox -f resize_letterbox -e static_library,h,registration -o ${OUTPUT_DIR} target=${HL_TARGET}"
)

for run in "${RUNS[@]}"; do
    # Split into array: first element is the generator executable name, rest are args
    read -ra parts <<< "$run"
    gen_name="${parts[0]}"
    args=("${parts[@]:1}")
    echo "  Running: ${gen_name} ${args[*]}"
    "${BIN_DIR}/${gen_name}" "${args[@]}"
done

echo ""
echo "=== Done ==="
echo "Generated artifacts in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"/*.h "${OUTPUT_DIR}"/*.a 2>/dev/null || echo "(no files found)"
