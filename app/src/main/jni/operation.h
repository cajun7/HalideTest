#pragma once

#include "operation_context.h"
#include <chrono>
#include <string>

using Clock = std::chrono::high_resolution_clock;

// Abstract base for all image processing operations.
// Inspired by Google HDR+ pipeline's HalideOperation pattern and
// Android Camera HAL's block-based processing architecture.
//
// Each concrete operation:
//   1. Implements run_halide() and run_opencv() with the actual processing
//   2. Self-registers via static initializer (see operation_registry.h)
//   3. Lives in its own file (ops/op_<name>.cpp)
//
// Adding a new operation requires creating ONE file — no modification
// of existing code (OCP: Open for extension, Closed for modification).

enum class InputFormat {
    RGB_BITMAP,      // Standard Android RGBA bitmap (most operations)
    NV21_BYTE_ARRAY, // NV21 camera byte array
    FLOAT_PLANAR,    // ML model output (segmentation)
    MULTI_INPUT       // Multiple inputs (e.g., bg_replace: fg + bg + mask)
};

struct IOperation {
    virtual ~IOperation() = default;

    // Unique operation name (used for JNI dispatch and benchmark CSV)
    virtual const char* name() const = 0;

    // What kind of input this operation expects
    virtual InputFormat input_format() const = 0;

    // Buffer layout the Halide generator requires (default: interleaved)
    virtual BufferLayout halide_layout() const { return BufferLayout::INTERLEAVED; }

    // Whether output dimensions differ from input (e.g., resize, rotate 90)
    virtual bool changes_dimensions() const { return false; }

    // Execute with Halide. Returns elapsed microseconds.
    virtual long run_halide(OperationContext& ctx) = 0;

    // Execute with OpenCV. Returns elapsed microseconds.
    virtual long run_opencv(OperationContext& ctx) = 0;

protected:
    // Timing helper
    static long elapsed_us(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};
