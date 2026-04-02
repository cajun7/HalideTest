# Build & Test Script

Unified script for building the HalideTest APK and running GoogleTest native tests on a connected Android device.

```
bash scripts/build_and_test.sh <command> [options]
```

## Commands

| Command | Description |
|---------|-------------|
| `apk` | Build the Android APK |
| `run` | Build APK, uninstall old, install new, and launch the app |
| `test` | Build and run GoogleTest native tests on device |
| `all` | Build APK, deploy, launch, then run all tests |
| `generators` | Rebuild Halide AOT generators (host compile + cross-compile) |

## Options

| Option | Description |
|--------|-------------|
| `--install` | Install APK on connected device after build |
| `--release` | Build release APK instead of debug |
| `--filter=<pattern>` | GoogleTest filter pattern (e.g. `FlipTest*`, `*Bilinear*`) |
| `--opencv=<version>` | Override OpenCV version (default: `3.4.16`) |
| `--gtest_*` | Pass-through any raw GoogleTest flags |
| `--help` | Show help message |

## Examples

### Build & Run APK

```bash
# Build APK, uninstall old, install new, launch app (the most common workflow)
bash scripts/build_and_test.sh run

# Build release variant and deploy
bash scripts/build_and_test.sh run --release

# Build debug APK only (no deploy)
bash scripts/build_and_test.sh apk

# Build and install on device (without uninstall/launch)
bash scripts/build_and_test.sh apk --install

# Build with a specific OpenCV version
bash scripts/build_and_test.sh run --opencv=3.4.16
```

### Run Tests

```bash
# Build and run ALL native tests
bash scripts/build_and_test.sh test

# Run only flip tests
bash scripts/build_and_test.sh test --filter='FlipTest*'

# Run only fused pipeline tests
bash scripts/build_and_test.sh test --filter='FusedBilinear*'

# Run all resize-related tests
bash scripts/build_and_test.sh test --filter='*Resize*'

# Run a single specific test case
bash scripts/build_and_test.sh test --filter='FlipTest.Horizontal_MatchesOpenCV'

# Run with verbose gtest output
bash scripts/build_and_test.sh test --gtest_print_time=0
```

### Build + Deploy + Test Together

```bash
# Build APK, deploy, launch, then run all native tests
bash scripts/build_and_test.sh all

# Run only target resize tests after deploying
bash scripts/build_and_test.sh all --filter='ResizeTarget*'
```

### Rebuild Generators

```bash
# Rebuild Halide AOT generators (needed after editing halide/generators/*.cpp)
bash scripts/build_and_test.sh generators
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENCV_VERSION` | Override OpenCV version (same as `--opencv=`) |
| `ANDROID_NDK_HOME` | Path to Android NDK root |
| `NDK_ROOT` | Alternative NDK path variable |

```bash
# Example: use env var for OpenCV version
OPENCV_VERSION=3.4.16 bash scripts/build_and_test.sh all --install
```

## Test Filter Patterns

The `--filter` option uses GoogleTest filter syntax:

| Pattern | Matches |
|---------|---------|
| `FlipTest*` | All tests in FlipTest suite |
| `*Bilinear*` | Any test with "Bilinear" in the name |
| `ResizeTarget*.HalfSize*` | HalfSize tests in ResizeTargetTest |
| `FlipTest.*:RotateTest.*` | All Flip and Rotate tests (`:` = OR) |
| `*-FlipTest.Vertical*` | All tests EXCEPT Vertical flip (`-` = exclude) |

## Available Test Suites

| Suite | Tests |
|-------|-------|
| `RgbBgrTest` | RGB/BGR channel swap |
| `Nv21ToRgbTest` | NV21 to RGB conversion |
| `RgbToNv21Test` | RGB to NV21 conversion |
| `GaussianBlurTest` | Gaussian blur (5x5) |
| `LensBlurTest` | Lens blur (disc kernel) |
| `ResizeTest` | Bilinear, bicubic, area resize (scale-factor) |
| `ResizeTargetTest` | Bilinear, bicubic, area resize (target-size) |
| `RotateTest` | Fixed 90/180/270 and arbitrary rotation |
| `FlipTest` | Horizontal and vertical flip |
| `FusedBilinearTest` | Fused NV21 pipeline (bilinear resize) |
| `FusedAreaTest` | Fused NV21 pipeline (area resize) |
| `TargetDispatchTest` | Multi-target runtime dispatch |

## Full Build Chain (First Time)

```bash
# 1. One-time setup
bash scripts/setup_halide.sh
OPENCV_VERSION=3.4.16 bash scripts/setup_opencv.sh
OPENCV_VERSION=3.4.16 bash scripts/build_opencv_source.sh

# 2. Build generators
bash scripts/build_and_test.sh generators

# 3. Build APK, deploy, launch, and run all tests
bash scripts/build_and_test.sh all
```
