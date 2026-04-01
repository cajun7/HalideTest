# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Android NDK C++ application benchmarking Halide AOT-compiled image processing pipelines against OpenCV 3.x. Targets arm64-v8a using ndk-build (Android.mk/Application.mk), follows TDD with GoogleTest.

## Build Commands

### Prerequisites (one-time setup)
```bash
# Download and extract Halide SDK (v21.0.0)
bash scripts/setup_halide.sh

# Download and extract OpenCV 3.4.16 Android SDK
bash scripts/setup_opencv.sh
```

### Compile Halide Generators (host -> AOT cross-compile for arm64-android)
```bash
bash halide/build_generators.sh
```
This produces `.a` (static libraries) and `.h` (headers) in `halide/generated/arm64-v8a/`.

### Build Android APK
```bash
./gradlew assembleDebug
```

### Run Native Tests on Device
```bash
bash app/src/test_native/run_tests.sh
```
Builds the GoogleTest executable with ndk-build, pushes to device via `adb`, and runs all tests.

## Architecture

### Directory Layout
- `halide/generators/` - Halide generator source files (one per operation)
- `halide/build_generators.sh` - Two-stage build: host compile + AOT cross-compile
- `halide/generated/arm64-v8a/` - Generated .a and .h files (gitignored)
- `app/src/main/jni/` - Native C++ code (JNI bridge, Halide/OpenCV wrappers)
- `app/src/main/java/` - Android UI (MainActivity, BenchmarkRunner, NativeBridge)
- `app/src/test_native/` - GoogleTest test files and build scripts
- `scripts/` - Setup and export scripts

### Image Processing Operations
1. **RGB <-> BGR** - Channel swap (`rgb_bgr_generator.cpp`)
2. **NV21 to RGB** - YUV conversion with BT.601 (`nv21_to_rgb_generator.cpp`)
3. **Gaussian Blur** - Separable 5x5, single-channel and RGB (`gaussian_blur_generator.cpp`)
4. **Lens Blur** - Disc kernel bokeh effect (`lens_blur_generator.cpp`)
5. **Resize** - Bilinear and Bicubic (`resize_generator.cpp`)
6. **Rotate** - Fixed 90/180/270 and arbitrary angle (`rotate_generator.cpp`)

### Critical Patterns
- **Boundary safety**: All blur/resize generators use `BoundaryConditions::repeat_edge` or `constant_exterior`
- **Odd resolution handling**: `TailStrategy::GuardWithIf` on final output, `TailStrategy::RoundUp` on intermediates
- **NEON vectorization**: `.vectorize(x, 16)` for uint8, `.vectorize(x, 8)` for float
- **Zero-copy JNI**: `AndroidBitmap_lockPixels` -> `halide_buffer_t` / `cv::Mat`

### TDD Workflow
1. Write test in `app/src/test_native/test_<operation>.cpp`
2. Test compares Halide output vs OpenCV reference (pixel-by-pixel with tolerance)
3. Tests parameterized by resolution including odd sizes (641x481, 1279x719)
4. Run: `bash app/src/test_native/run_tests.sh`

### Excel Report
```bash
python3 scripts/export_benchmark.py [output.xlsx]
```
Pulls CSV from device, generates 3-sheet Excel: Raw Data, Summary (with speedup), Charts.
