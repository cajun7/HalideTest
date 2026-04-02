# HalideTest — Halide vs OpenCV Android Benchmark

Android NDK benchmark app comparing **Halide AOT-compiled** image processing pipelines against **OpenCV 3.4.16** on arm64 devices.

---

## Supported Operations (12 Halide Generators)

| # | Operation | Halide Generator | OpenCV Equivalent |
|---|-----------|-----------------|-------------------|
| 1 | RGB to BGR | `rgb_bgr_convert` | `cvtColor(COLOR_RGB2BGR)` |
| 2 | NV21 to RGB | `nv21_to_rgb` | `cvtColor(COLOR_YUV2RGB_NV21)` |
| 3 | RGB to NV21 | `rgb_to_nv21` | `cvtColor(COLOR_RGB2YUV_YV12)` + interleave |
| 4 | Gaussian Blur (Y) | `gaussian_blur_y` | `GaussianBlur` |
| 5 | Gaussian Blur (RGB) | `gaussian_blur_rgb` | `GaussianBlur` |
| 6 | Lens Blur (Bokeh) | `lens_blur` | `filter2D` with disc kernel |
| 7 | Resize Bilinear | `resize_bilinear` | `resize(INTER_LINEAR)` |
| 8 | Resize Bicubic | `resize_bicubic` | `resize(INTER_CUBIC)` |
| 9 | Resize Area | `resize_area` | `resize(INTER_AREA)` |
| 10 | Resize Letterbox | `resize_letterbox` | `resize` + `copyMakeBorder` |
| 11 | Rotate 90 CW | `rotate_fixed` | `rotate(ROTATE_90_CLOCKWISE)` |
| 12 | Rotate Arbitrary | `rotate_arbitrary` | `warpAffine` |

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| **macOS** | 12+ | Host build environment |
| **JDK** | 17 | Required by Gradle / AGP |
| **Android SDK** | Platform 34 | `~/Library/Android/sdk` |
| **Android NDK** | 27.x | Installed via SDK Manager |
| **Android Build Tools** | 34.x | Installed via SDK Manager |
| **Android Device** | arm64-v8a | Physical device with USB Debugging enabled |
| **adb** | latest | Part of Android SDK Platform Tools |
| **Python 3** | 3.8+ | Only for Excel report generation |

---

## Setup (One-Time)

### 1. Set JAVA_HOME

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home
```

> Adjust the path if your JDK 17 is installed elsewhere.
> Add this to `~/.zshrc` or `~/.bash_profile` to persist across sessions.

### 2. Install Android SDK Components

```bash
export ANDROID_HOME=~/Library/Android/sdk
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "ndk;27.1.12297006"
```

### 3. Create `local.properties`

```bash
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
echo "ndk.dir=$HOME/Library/Android/sdk/ndk/27.1.12297006" >> local.properties
```

### 4. Download Dependencies

```bash
# Halide 21.0.0 SDK (host compiler + includes)
bash scripts/setup_halide.sh

# OpenCV 3.4.16 Android SDK (static libraries)
bash scripts/setup_opencv.sh
```

### 5. Compile Halide Generators

```bash
bash halide/build_generators.sh
```

This runs a two-stage pipeline for each generator:
1. **Host compile**: generator source → macOS executable
2. **AOT cross-compile**: executable → `arm-64-android` static library (`.a`) + header (`.h`)

Output: `halide/generated/arm64-v8a/` (12 `.a` + 12 `.h` files)

---

## Build

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home
./gradlew assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`

---

## Install & Run

### Connect Device

1. Enable **Developer Options** on your Android device
   (Settings → About Phone → tap "Build Number" 7 times)
2. Enable **USB Debugging**
   (Settings → Developer Options → USB Debugging → ON)
3. Connect device via USB cable
4. Verify connection:
   ```bash
   adb devices
   ```

### Install APK

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Launch

```bash
adb shell am start -n com.example.halidetest/.MainActivity
```

Or tap **"HalideTest"** in your device's app drawer.

### Quick One-Liner (Build + Install + Launch)

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home && \
./gradlew assembleDebug && \
adb install -r app/build/outputs/apk/debug/app-debug.apk && \
adb shell am start -n com.example.halidetest/.MainActivity
```

---

## Using the App

The app provides a simple benchmark UI:

| Control | Description |
|---------|-------------|
| **Operation** (Spinner) | Select which image processing operation to benchmark |
| **Framework** (Radio) | Choose **Halide**, **OpenCV**, or **Both** |
| **Resolution** (Spinner) | Test image resolution (320x240 to 3840x2160, including odd sizes) |
| **Iterations** (EditText) | Number of iterations per benchmark run |
| **Run** (Button) | Benchmark the selected operation |
| **Run All** (Button) | Benchmark all operations sequentially |

### Available Resolutions

| Resolution | Notes |
|-----------|-------|
| 320x240 | QVGA |
| 640x480 | VGA |
| 641x481 | Odd-pixel edge case |
| 1280x720 | 720p HD |
| 1279x719 | Odd-pixel edge case |
| 1920x1080 | 1080p Full HD |
| 3840x2160 | 4K UHD |

### Benchmark Results

Results are displayed on screen showing:
- **Median** / **Mean** / **Min** / **Max** execution time in microseconds
- **Speedup ratio** (Halide vs OpenCV) when running both frameworks

> **Tip**: Use 50–100 iterations for stable, reliable measurements.

---

## Collecting & Exporting Results

### Pull CSV from Device

The app saves benchmark results as CSV automatically:

```bash
adb pull /sdcard/Android/data/com.example.halidetest/files/benchmark_results.csv .
```

CSV format:
```
operation,framework,resolution,median_us,mean_us,min_us,max_us,timestamp
```

### Generate Excel Report

```bash
pip3 install openpyxl    # one-time dependency
python3 scripts/export_benchmark.py benchmark_results.csv output.xlsx
```

The Excel report contains 3 sheets:

| Sheet | Contents |
|-------|----------|
| **Raw Data** | All benchmark rows from CSV |
| **Summary** | Pivot table: Operation × Resolution → Halide time, OpenCV time, Speedup |
| **Charts** | Grouped bar charts comparing Halide vs OpenCV per operation |

---

## Project Structure

```
Halide/
├── halide/
│   ├── generators/              # Halide generator source files
│   │   ├── rgb_bgr_generator.cpp
│   │   ├── nv21_to_rgb_generator.cpp
│   │   ├── rgb_to_nv21_generator.cpp
│   │   ├── gaussian_blur_generator.cpp
│   │   ├── lens_blur_generator.cpp
│   │   ├── resize_generator.cpp
│   │   ├── resize_area_generator.cpp
│   │   ├── resize_letterbox_generator.cpp
│   │   └── rotate_generator.cpp
│   ├── build_generators.sh      # Host compile + AOT cross-compile script
│   ├── Halide-21.0.0/           # (gitignored) Halide SDK
│   └── generated/arm64-v8a/     # (gitignored) .a + .h output
│
├── opencv/
│   └── OpenCV-android-sdk/      # (gitignored) OpenCV 3.4.16 SDK
│
├── app/src/main/
│   ├── jni/                     # Native C++ (JNI bridge, Halide/OpenCV wrappers)
│   ├── java/.../halidetest/     # Android UI (MainActivity, NativeBridge)
│   └── res/                     # Layouts, strings
│
├── scripts/
│   ├── setup_halide.sh          # Download Halide SDK
│   ├── setup_opencv.sh          # Download OpenCV Android SDK
│   └── export_benchmark.py      # CSV → Excel report
│
└── README.md                    # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `JAVA_HOME` not set | `export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home` |
| `ndk-build` not found | Install NDK via SDK Manager: `sdkmanager "ndk;27.1.12297006"` |
| `Halide.h` not found | Run `bash scripts/setup_halide.sh` |
| `OpenCV.mk` not found | Run `bash scripts/setup_opencv.sh` |
| Generator `.a` files missing | Run `bash halide/build_generators.sh` |
| `adb: no devices` | Enable USB Debugging on device, reconnect USB |
| App crashes on launch | Ensure device is arm64-v8a (not x86 emulator) |
| Build fails with `fcntl()` warning | This is harmless — check for actual errors below it |
