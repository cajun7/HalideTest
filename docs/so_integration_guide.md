# Integrating Prebuilt .so Libraries into the HalideTest Project

This guide explains how to integrate prebuilt shared libraries (`.so` files) — both OpenCV and generic C++ libraries — into this Android NDK project that uses **ndk-build** (`Android.mk` / `Application.mk`).

---

## Table of Contents

1. [Prerequisites & Current State](#1-prerequisites--current-state)
2. [Inspect Your .so Files First](#2-inspect-your-so-files-first)
3. [Directory Layout](#3-directory-layout)
4. [OpenCV .so Integration (Main App)](#4-opencv-so-integration-main-app)
5. [Generic C++ .so Integration (Main App)](#5-generic-c-so-integration-main-app)
6. [APP_STL: c++_static vs c++_shared (Critical)](#6-app_stl-c_static-vs-c_shared-critical)
7. [Java Load Order](#7-java-load-order)
8. [APK Packaging](#8-apk-packaging)
9. [Test Integration](#9-test-integration)
10. [Debugging](#10-debugging)
11. [Quick Reference Cheat Sheet](#11-quick-reference-cheat-sheet)

---

## 1. Prerequisites & Current State

Before making any changes, understand what the project uses today:

| Setting | Current Value | File |
|---------|--------------|------|
| Build system | ndk-build | `app/build.gradle` |
| OpenCV linking | **STATIC** | `app/src/main/jni/Android.mk` line 17 |
| C++ STL | **c++_static** | `app/src/main/jni/Application.mk` line 3 |
| Target ABI | arm64-v8a only | `app/src/main/jni/Application.mk` line 1 |
| Min API | 24 | `app/src/main/jni/Application.mk` line 2 |
| NDK version | 27.1.12297006 | `app/build.gradle` line 6 |

**Key point:** Everything is currently statically linked — OpenCV modules are `.a` files embedded into `libhalide_benchmark.so`, and the C++ runtime is also static. Introducing a `.so` dependency changes the runtime linking model.

---

## 2. Inspect Your .so Files First

Before integrating **any** `.so` file, run these commands to check compatibility. Use `readelf` from the NDK toolchain or your host system.

### 2.1 Check architecture (must be AArch64)

```bash
file libfoo.so
# Expected: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), ...
```

### 2.2 Check shared library dependencies

```bash
readelf -d libfoo.so | grep NEEDED
# Look for:
#   libc++_shared.so  → means you MUST switch APP_STL to c++_shared (see Section 6)
#   libopencv_java3.so → transitive OpenCV dependency
#   liblog.so, libc.so, libm.so, libdl.so → standard Android, OK
```

### 2.3 Check the SONAME

```bash
readelf -d libfoo.so | grep SONAME
# The SONAME is what the linker records as the dependency name.
# Your file name should match the SONAME.
```

### 2.4 Check exported symbols

```bash
# List functions the .so provides
nm -D libfoo.so | grep ' T ' | head -20

# List functions the .so expects from others (undefined)
nm -D libfoo.so | grep ' U ' | head -20
```

### 2.5 Check target API level

```bash
readelf -n libfoo.so | grep -i api
# Or check the ELF note section. Libraries built for API > 24 may use
# symbols not available on your minSdkVersion.
```

---

## 3. Directory Layout

### 3.1 For the main app (APK packaging)

Place `.so` files where Gradle will automatically bundle them into the APK:

```
app/src/main/jniLibs/
└── arm64-v8a/
    ├── libopencv_java3.so      ← OpenCV shared lib (if switching from static)
    ├── libmylib.so             ← your custom C++ .so
    └── libc++_shared.so        ← only if APP_STL=c++_shared (see Section 6)
```

> **Note:** The `jniLibs/` directory does not exist yet — create it.

### 3.2 For headers and build-time reference

Place headers alongside the `.so` files in a structured directory:

```
prebuilt/
├── arm64-v8a/
│   ├── libmylib.so
│   └── libopencv_custom.so     ← if using a custom OpenCV build
└── include/
    └── mylib/
        ├── mylib.h
        └── mylib_types.h
```

### 3.3 For on-device tests

Test `.so` files are pushed manually — no `jniLibs` needed. See [Section 9](#9-test-integration).

---

## 4. OpenCV .so Integration (Main App)

### Option A: Switch the existing OpenCV SDK to shared linking (easiest)

The OpenCV Android SDK already ships with `libopencv_java3.so`. You just need to flip one variable.

**Step 1 — Change `app/src/main/jni/Android.mk`:**

```makefile
# Line 17: change STATIC → SHARED
OPENCV_LIB_TYPE := SHARED
```

That's it for the build file — `OpenCV.mk` handles the rest internally (it will use `PREBUILT_SHARED_LIBRARY` instead of `PREBUILT_STATIC_LIBRARY` and reference `libopencv_java3.so`).

**Step 2 — Copy the .so for APK packaging:**

```bash
mkdir -p app/src/main/jniLibs/arm64-v8a

cp opencv/3.4.16/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java3.so \
   app/src/main/jniLibs/arm64-v8a/
```

**Step 3 — Update Java load order** (see [Section 7](#7-java-load-order)).

**Step 4 — Check STL compatibility** (see [Section 6](#6-app_stl-c_static-vs-c_shared-critical)):

```bash
readelf -d opencv/3.4.16/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java3.so \
    | grep NEEDED
```

If `libc++_shared.so` appears in the output, you must also switch `APP_STL` to `c++_shared`.

### Option B: Manual integration of a custom OpenCV .so

If you have a custom-built OpenCV `.so` (not from the official Android SDK), declare it manually in `Android.mk`.

**Step 1 — Remove or comment out the OpenCV.mk include:**

```makefile
# OPENCV_LIB_TYPE := STATIC
# OPENCV_INSTALL_MODULES := on
# include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk
```

**Step 2 — Add a PREBUILT_SHARED_LIBRARY block BEFORE your main module:**

```makefile
# ---------------------------------------------------------------
# Custom OpenCV shared library
# ---------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := opencv_custom
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../prebuilt/arm64-v8a/libopencv_custom.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../prebuilt/include
include $(PREBUILT_SHARED_LIBRARY)
```

**Step 3 — In the main module, replace OpenCV static references:**

```makefile
LOCAL_MODULE := halide_benchmark

# ... (existing source files, includes, ldflags) ...

# Add the shared library dependency
LOCAL_SHARED_LIBRARIES += opencv_custom

include $(BUILD_SHARED_LIBRARY)
```

**Step 4 — Copy the .so to `jniLibs/arm64-v8a/`** for APK packaging.

---

## 5. Generic C++ .so Integration (Main App)

For any prebuilt C++ shared library (not OpenCV).

### 5.1 Declare the prebuilt module in Android.mk

Add this block **before** the `LOCAL_MODULE := halide_benchmark` section:

```makefile
# ---------------------------------------------------------------
# Prebuilt: libmylib.so
# ---------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := mylib_prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../prebuilt/arm64-v8a/libmylib.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../prebuilt/include
include $(PREBUILT_SHARED_LIBRARY)
```

Key fields:
- `LOCAL_MODULE` — any name you choose (used as a reference in `LOCAL_SHARED_LIBRARIES`)
- `LOCAL_SRC_FILES` — path to the `.so` file, relative to `LOCAL_PATH`
- `LOCAL_EXPORT_C_INCLUDES` — header directory; automatically added to include paths of any module that depends on this

### 5.2 Link your main module against it

In the `halide_benchmark` module section, add:

```makefile
LOCAL_SHARED_LIBRARIES += mylib_prebuilt
```

### 5.3 Full example (Android.mk with one prebuilt .so added)

```makefile
LOCAL_PATH := $(call my-dir)

HALIDE_GEN_DIR := $(LOCAL_PATH)/../../../../halide/generated/arm64-v8a
HALIDE_INCLUDE := $(LOCAL_PATH)/../../../../halide/Halide-21.0.0/include
OPENCV_VERSION ?= 3.4.16
OPENCV_ANDROID_SDK := $(LOCAL_PATH)/../../../../opencv/$(OPENCV_VERSION)/OpenCV-android-sdk
PREBUILT_DIR := $(LOCAL_PATH)/../../../../prebuilt

# ---------------------------------------------------------------
# Prebuilt: libmylib.so
# ---------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := mylib_prebuilt
LOCAL_SRC_FILES := $(PREBUILT_DIR)/arm64-v8a/libmylib.so
LOCAL_EXPORT_C_INCLUDES := $(PREBUILT_DIR)/include
include $(PREBUILT_SHARED_LIBRARY)

# ---------------------------------------------------------------
# OpenCV (still static in this example)
# ---------------------------------------------------------------
OPENCV_LIB_TYPE := STATIC
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# Main shared library
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

LOCAL_SHARED_LIBRARIES += mylib_prebuilt

LOCAL_LDFLAGS += \
    $(HALIDE_GEN_DIR)/rgb_bgr_convert.a \
    ... (all Halide .a files) ...
    $(HALIDE_GEN_DIR)/halide_runtime.a

include $(BUILD_SHARED_LIBRARY)
```

### 5.4 Multiple .so files

Repeat the `PREBUILT_SHARED_LIBRARY` block for each `.so` and add each to `LOCAL_SHARED_LIBRARIES`:

```makefile
LOCAL_SHARED_LIBRARIES += mylib_prebuilt anotherlib_prebuilt
```

---

## 6. APP_STL: c++_static vs c++_shared (Critical)

### The one-runtime-per-process rule

The Android NDK enforces: **there must be exactly one copy of the C++ runtime in a process.**

- `c++_static` — the C++ standard library is compiled into each `.so` separately
- `c++_shared` — all `.so` files share a single `libc++_shared.so` at runtime

### When you MUST switch to c++_shared

**If your prebuilt `.so` was built with `c++_shared`:**

```bash
readelf -d libmylib.so | grep NEEDED
# If you see "libc++_shared.so" in the output → MUST switch
```

If you see `libc++_shared.so` as a NEEDED dependency, you **must** change both Application.mk files:

**`app/src/main/jni/Application.mk`:**
```makefile
APP_ABI := arm64-v8a
APP_PLATFORM := android-24
APP_STL := c++_shared          # ← changed from c++_static
APP_CPPFLAGS := -std=c++17 -O2 -fno-rtti
APP_OPTIM := release
```

**`app/src/test_native/jni/Application.mk`:**
```makefile
APP_ABI := arm64-v8a
APP_PLATFORM := android-24
APP_STL := c++_shared          # ← changed from c++_static
APP_CPPFLAGS := -std=c++17 -O2
APP_OPTIM := release
```

### When c++_static is OK

If your `.so` does **not** list `libc++_shared.so` as NEEDED, it was likely built with `c++_static`. This is safe **only if** the `.so` does not pass C++ objects (like `std::string`, `std::vector`, exceptions) across the library boundary.

### After switching to c++_shared

1. The NDK build will produce `libc++_shared.so` alongside your app's `.so`.
2. For the **APK**, ndk-build + Gradle handles this automatically — `libc++_shared.so` is bundled.
3. For **on-device tests**, you must push `libc++_shared.so` manually (see [Section 9](#9-test-integration)).

### Where to find libc++_shared.so

```bash
# In your NDK installation:
find $ANDROID_NDK_HOME -name "libc++_shared.so" -path "*/aarch64*"
# Typically at:
# $NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
```

---

## 7. Java Load Order

When using shared libraries, **load order matters**. Dependencies must be loaded before the libraries that need them.

### Current code (`NativeBridge.java` line 11-12):

```java
static {
    System.loadLibrary("halide_benchmark");
}
```

### After adding shared OpenCV:

```java
static {
    System.loadLibrary("opencv_java3");       // dependency loaded first
    System.loadLibrary("halide_benchmark");   // depends on opencv_java3
}
```

### After adding a custom .so + shared OpenCV:

```java
static {
    System.loadLibrary("mylib");              // no dependencies (or only system libs)
    System.loadLibrary("opencv_java3");       // may depend on libc++_shared
    System.loadLibrary("halide_benchmark");   // depends on both above
}
```

### Rules

1. Load **leaf dependencies first** (libraries with no custom deps), then work up.
2. System libraries (`liblog.so`, `libm.so`, `libc.so`) are loaded by the OS automatically — don't call `loadLibrary` for them.
3. `libc++_shared.so` is loaded automatically by the linker on API 23+ — you don't need to load it explicitly in Java.
4. The library name in `loadLibrary()` is **without** the `lib` prefix and `.so` suffix: `libopencv_java3.so` → `"opencv_java3"`.

---

## 8. APK Packaging

### 8.1 Using jniLibs (recommended)

Any `.so` placed in `app/src/main/jniLibs/<abi>/` is automatically included in the APK by Gradle.

```bash
mkdir -p app/src/main/jniLibs/arm64-v8a
cp path/to/libmylib.so app/src/main/jniLibs/arm64-v8a/
```

### 8.2 Using Gradle sourceSets (alternative)

If your `.so` files live elsewhere, point Gradle to them in `app/build.gradle`:

```gradle
android {
    sourceSets {
        main {
            jniLibs.srcDirs = ['src/main/jniLibs', '../prebuilt']
            // prebuilt/ must contain arm64-v8a/ subdirectory with .so files
        }
    }
}
```

### 8.3 Verify .so files are in the APK

After building:

```bash
./gradlew assembleDebug

# List all .so files inside the APK
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep '\.so$'
# Expected output:
#   lib/arm64-v8a/libhalide_benchmark.so
#   lib/arm64-v8a/libopencv_java3.so      ← if using shared OpenCV
#   lib/arm64-v8a/libmylib.so             ← your prebuilt
#   lib/arm64-v8a/libc++_shared.so        ← if APP_STL=c++_shared
```

### 8.4 Verify on device after install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell "ls -la /data/app/~~*/com.example.halidetest*/lib/arm64/"
```

---

## 9. Test Integration

The native tests (`halide_tests`) run as a standalone executable pushed to the device — they don't go through the APK. Shared library dependencies must be handled differently.

### 9.1 Modify test Android.mk

Edit `app/src/test_native/jni/Android.mk`. Add prebuilt module blocks **before** the test executable definition:

```makefile
LOCAL_PATH := $(call my-dir)

HALIDE_GEN_DIR := $(LOCAL_PATH)/../../../../halide/generated/arm64-v8a
HALIDE_INCLUDE := $(LOCAL_PATH)/../../../../halide/Halide-21.0.0/include
OPENCV_VERSION ?= 3.4.16
OPENCV_ANDROID_SDK := $(LOCAL_PATH)/../../../../opencv/$(OPENCV_VERSION)/OpenCV-android-sdk
PREBUILT_DIR := $(LOCAL_PATH)/../../../../prebuilt

# ---------------------------------------------------------------
# Prebuilt: libmylib.so  (ADD THIS BLOCK)
# ---------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := mylib_prebuilt
LOCAL_SRC_FILES := $(PREBUILT_DIR)/arm64-v8a/libmylib.so
LOCAL_EXPORT_C_INCLUDES := $(PREBUILT_DIR)/include
include $(PREBUILT_SHARED_LIBRARY)

# ---------------------------------------------------------------
# OpenCV (switch to SHARED if needed)
# ---------------------------------------------------------------
OPENCV_LIB_TYPE := STATIC       # or SHARED
OPENCV_INSTALL_MODULES := on
include $(OPENCV_ANDROID_SDK)/sdk/native/jni/OpenCV.mk

# ---------------------------------------------------------------
# GoogleTest test executable
# ---------------------------------------------------------------
LOCAL_MODULE := halide_tests

LOCAL_SRC_FILES := \
    ../test_nv21_to_rgb.cpp \
    ... (all test files) ...

LOCAL_SHARED_LIBRARIES += mylib_prebuilt    # ADD THIS LINE

# ... rest of existing configuration ...

include $(BUILD_EXECUTABLE)

$(call import-module,third_party/googletest)
```

### 9.2 Modify run_tests.sh

Update `app/src/test_native/run_tests.sh` to push `.so` files and set `LD_LIBRARY_PATH`:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

# --- (existing NDK discovery code stays the same) ---

OPENCV_VERSION="${OPENCV_VERSION:-3.4.16}"
echo "Using OpenCV: ${OPENCV_VERSION}"
echo ""

echo "=== Building test executable ==="
"${NDK_ROOT}/ndk-build" \
    NDK_PROJECT_PATH="${SCRIPT_DIR}" \
    APP_BUILD_SCRIPT="${SCRIPT_DIR}/jni/Android.mk" \
    NDK_APPLICATION_MK="${SCRIPT_DIR}/jni/Application.mk" \
    NDK_OUT="${SCRIPT_DIR}/obj" \
    NDK_LIBS_OUT="${SCRIPT_DIR}/libs" \
    OPENCV_VERSION="${OPENCV_VERSION}" \
    -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo "=== Pushing test binary to device ==="
adb push "${SCRIPT_DIR}/libs/arm64-v8a/halide_tests" /data/local/tmp/
adb shell chmod 755 /data/local/tmp/halide_tests

# ---------------------------------------------------------------
# Push shared library dependencies (ADD THIS SECTION)
# ---------------------------------------------------------------
echo ""
echo "=== Pushing shared library dependencies ==="

# Push your prebuilt .so files
PREBUILT_SO_DIR="${PROJECT_ROOT}/prebuilt/arm64-v8a"
if [ -d "${PREBUILT_SO_DIR}" ]; then
    for so_file in "${PREBUILT_SO_DIR}"/*.so; do
        [ -f "$so_file" ] && adb push "$so_file" /data/local/tmp/
    done
fi

# Push shared OpenCV (only if using OPENCV_LIB_TYPE := SHARED)
# OPENCV_SO="${PROJECT_ROOT}/opencv/${OPENCV_VERSION}/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java3.so"
# [ -f "${OPENCV_SO}" ] && adb push "${OPENCV_SO}" /data/local/tmp/

# Push libc++_shared.so (only if APP_STL=c++_shared)
# Find it in the NDK:
# LIBCXX=$(find "${NDK_ROOT}" -name "libc++_shared.so" -path "*/aarch64*" | head -1)
# [ -n "${LIBCXX}" ] && adb push "${LIBCXX}" /data/local/tmp/
# ---------------------------------------------------------------

echo ""
echo "=== Running tests on device ==="
# Set LD_LIBRARY_PATH so the linker can find the .so files
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./halide_tests $*"
```

**Key changes:**
1. Push all `.so` files from `prebuilt/arm64-v8a/` to `/data/local/tmp/`
2. Optionally push `libopencv_java3.so` and `libc++_shared.so` (uncomment as needed)
3. Run the test with `LD_LIBRARY_PATH=/data/local/tmp` so the dynamic linker can find the `.so` files

> **Important:** Without `LD_LIBRARY_PATH`, the test will crash immediately with: `error while loading shared libraries: libmylib.so: cannot open shared object file`

---

## 10. Debugging

### 10.1 Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `UnsatisfiedLinkError: dlopen failed: library "libfoo.so" not found` | `.so` not in APK or not in `LD_LIBRARY_PATH` | Copy to `jniLibs/arm64-v8a/` or push to device |
| `UnsatisfiedLinkError: dlopen failed: cannot locate symbol "_ZNSt6..."` | STL mismatch | Check `readelf -d`, switch `APP_STL` to `c++_shared` |
| `SIGBUS` or `SIGABRT` in `__cxa_throw` or `__cxa_guard_acquire` | Two copies of C++ runtime | One `.so` uses `c++_static` and another also uses `c++_static` — switch to `c++_shared` |
| `cannot locate symbol "..."` | ABI mismatch or missing transitive dependency | Check `readelf -d` on all `.so` files, ensure all NEEDED libs are present |
| `dlopen failed: ... has text relocations` | `.so` built without `-fPIC` | Rebuild the `.so` with `-fPIC` flag (cannot be fixed post-build) |
| `has invalid shdr offset/size` | Wrong architecture | Verify with `file libfoo.so` — must be `ARM aarch64` |

### 10.2 Diagnostic commands

```bash
# Watch library loading in real-time
adb logcat | grep -E "(linker|dlopen|UnsatisfiedLink|halide_benchmark)"

# Check what's installed in the APK on device
adb shell "ls -la /data/app/~~*/com.example.halidetest*/lib/arm64/"

# Inspect a .so file on device
adb shell "readelf -d /data/local/tmp/libmylib.so"

# Check if symbols are resolved
adb shell "readelf -d /data/local/tmp/halide_tests | grep NEEDED"

# Verify library can be loaded
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. LD_DEBUG=libs ./halide_tests --gtest_list_tests 2>&1 | head -20"
```

### 10.3 Verifying symbol visibility

If your code calls a function from the `.so` but gets "undefined reference" at link time:

```bash
# Check if the function is exported
nm -D libmylib.so | grep "my_function_name"
# T = exported text (code) symbol ← what you want
# U = undefined (imported from elsewhere)
# If no output → the symbol is not exported; rebuild with __attribute__((visibility("default")))
```

---

## 11. Quick Reference Cheat Sheet

### PREBUILT_SHARED_LIBRARY template

```makefile
include $(CLEAR_VARS)
LOCAL_MODULE := <module_name>
LOCAL_SRC_FILES := <path/to/lib.so>
LOCAL_EXPORT_C_INCLUDES := <path/to/headers>
include $(PREBUILT_SHARED_LIBRARY)
```

### Integration checklist

- [ ] `.so` is built for `arm64-v8a` (verify with `file`)
- [ ] `.so` dependencies checked (verify with `readelf -d | grep NEEDED`)
- [ ] If NEEDED includes `libc++_shared.so` → switch `APP_STL` to `c++_shared` in **both** Application.mk files
- [ ] `PREBUILT_SHARED_LIBRARY` block added to `Android.mk` (before main module)
- [ ] `LOCAL_SHARED_LIBRARIES +=` added to main module
- [ ] `.so` copied to `app/src/main/jniLibs/arm64-v8a/`
- [ ] `System.loadLibrary()` call added in `NativeBridge.java` (before `halide_benchmark`)
- [ ] APK verified with `unzip -l | grep .so`
- [ ] Test `Android.mk` updated with same prebuilt block
- [ ] `run_tests.sh` updated to push `.so` files and set `LD_LIBRARY_PATH`

### APP_STL decision table

| Your .so NEEDED list | Action |
|---------------------|--------|
| No `libc++_shared.so` and no C++ objects cross boundary | Keep `c++_static` |
| No `libc++_shared.so` but C++ objects cross boundary | Switch to `c++_shared` |
| Contains `libc++_shared.so` | **Must** switch to `c++_shared` |

### Static vs Shared comparison

| | Static (`.a`) | Shared (`.so`) |
|---|---|---|
| Linked at | Build time | Runtime |
| In final APK | Embedded in `libhalide_benchmark.so` | Separate file in `lib/arm64-v8a/` |
| APK size | Larger single `.so` | Smaller main `.so` + separate `.so` files |
| Runtime overhead | None | Dynamic linker resolves symbols at load time |
| Must be on device | No (embedded) | Yes (APK or `LD_LIBRARY_PATH`) |
| Java `loadLibrary` | Not needed | Required (in dependency order) |

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `app/src/main/jni/Android.mk` | Main build — declare prebuilt modules and link flags here |
| `app/src/main/jni/Application.mk` | APP_STL and ABI settings for main app |
| `app/src/main/java/.../NativeBridge.java` | Java `System.loadLibrary()` calls |
| `app/build.gradle` | Gradle NDK integration, `jniLibs.srcDirs` |
| `app/src/main/jniLibs/arm64-v8a/` | Drop `.so` files here for APK packaging (create if missing) |
| `app/src/test_native/jni/Android.mk` | Test build — mirror prebuilt blocks from main |
| `app/src/test_native/jni/Application.mk` | APP_STL for tests — must match main |
| `app/src/test_native/run_tests.sh` | Push `.so` files + set `LD_LIBRARY_PATH` |
