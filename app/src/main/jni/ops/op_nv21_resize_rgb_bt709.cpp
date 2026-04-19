#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../bt709_neon_ref.h"
#include "nv21_resize_rgb_bt709_nearest.h"
#include "nv21_resize_rgb_bt709_bilinear.h"
#include "nv21_resize_rgb_bt709_area.h"
#include "nv21_resize_nearest_optimized.h"
#include "nv21_resize_bilinear_optimized.h"
#include "nv21_resize_area_optimized.h"

// Fused NV21 -> resize (Y+UV) -> BT.709 RGB, three interpolation variants.
// Registry entries are stub-only (NV21Context, not OperationContext); the
// real dispatch lives in the helpers below, consumed by bench_main.cpp.

class Nv21ResizeRgbBt709NearestOp : public IOperation {
public:
    const char* name() const override { return "nv21_resize_rgb_bt709_nearest"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }
    bool changes_dimensions() const override { return true; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Nv21ResizeRgbBt709NearestOp);

class Nv21ResizeRgbBt709BilinearOp : public IOperation {
public:
    const char* name() const override { return "nv21_resize_rgb_bt709_bilinear"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }
    bool changes_dimensions() const override { return true; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Nv21ResizeRgbBt709BilinearOp);

class Nv21ResizeRgbBt709AreaOp : public IOperation {
public:
    const char* name() const override { return "nv21_resize_rgb_bt709_area"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }
    bool changes_dimensions() const override { return true; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Nv21ResizeRgbBt709AreaOp);
