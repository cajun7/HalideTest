Here's the full step-by-step process to build and run the app:

Prerequisites
1. Java 17

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home
2. Android SDK & NDK
Ensure these are installed (should already be set up from previous work):

Android SDK Platform 34
Android NDK 27.1.12297006
Build Tools 34
Verify with:


ls ~/Library/Android/sdk/ndk/27.1.12297006/
3. Physical Device (arm64)
An Android device with arm64-v8a architecture (virtually all modern Android phones)
USB Debugging enabled: Settings → Developer Options → USB Debugging ON
Connected via USB cable
Verify device connection:


adb devices
You should see your device listed.

Build & Run
Step 1: Download dependencies (one-time)

# Download Halide 21.0.0 SDK
bash scripts/setup_halide.sh

# Download OpenCV 4.9.0 Android SDK
bash scripts/setup_opencv.sh
Step 2: Compile Halide generators (host → AOT cross-compile)

bash halide/build_generators.sh
This produces 12 .a + .h files in halide/generated/arm64-v8a/.

Step 3: Build the APK

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home
./gradlew assembleDebug
Output APK: app/build/outputs/apk/debug/app-debug.apk

Step 4: Install on device

adb install -r app/build/outputs/apk/debug/app-debug.apk
Step 5: Launch the app

adb shell am start -n com.example.halidetest/.MainActivity
Or simply tap "HalideTest" on your device's app drawer.

Using the App
The UI has these controls:

Control	Description
Operation Spinner	Select one of 11 operations (RGB to BGR, NV21 to RGB, Gaussian Blur, Lens Blur, Resize Bilinear/Bicubic/Area/Letterbox, Rotate 90/Arbitrary, RGB to NV21)
Framework Radio	Choose Halide, OpenCV, or Both
Resolution Spinner	Pick test resolution (320x240 up to 3840x2160, including odd sizes like 641x481)
Iterations	Number of benchmark iterations (default 10, recommend 50-100 for stable results)
Run button	Runs selected operation only
Run All button	Runs all 11 operations sequentially with both Halide and OpenCV, shows speedup
Results display: median, mean, min, max in microseconds.

Collecting Benchmark Data
On-device CSV
The app automatically writes results to:


/sdcard/Android/data/com.example.halidetest/files/benchmark_results.csv
Pull CSV to Mac

adb pull /sdcard/Android/data/com.example.halidetest/files/benchmark_results.csv .
Generate Excel Report

pip3 install openpyxl   # one-time
python3 scripts/export_benchmark.py benchmark_results.csv output.xlsx
This produces a 3-sheet Excel file:

Raw Data — all CSV rows
Summary — pivot table with Halide vs OpenCV times + speedup ratio
Charts — grouped bar charts
Quick One-Liner (build + install + launch)

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.0.3.1.jdk/Contents/Home && \
./gradlew assembleDebug && \
adb install -r app/build/outputs/apk/debug/app-debug.apk && \
adb shell am start -n com.example.halidetest/.MainActivity