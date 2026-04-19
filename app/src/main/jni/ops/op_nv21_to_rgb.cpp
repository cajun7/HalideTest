#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class Nv21ToRgbOp : public IOperation {
public:
    const char* name() const override { return "nv21_to_rgb"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }

    long run_halide(OperationContext& ctx) override {
        // NV21 operations use NV21Context, not OperationContext
        // This is a placeholder for the registry pattern demo
        return -1;
    }

    long run_opencv(OperationContext& ctx) override {
        return -1;
    }
};

REGISTER_OP(Nv21ToRgbOp);

class Nv21Yuv444RgbOp : public IOperation {
public:
    const char* name() const override { return "nv21_yuv444_rgb"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }

    long run_halide(OperationContext& ctx) override { return -1; }
    long run_opencv(OperationContext& ctx) override { return -1; }
};

REGISTER_OP(Nv21Yuv444RgbOp);

class Nv21ToRgbFullRangeOp : public IOperation {
public:
    const char* name() const override { return "nv21_to_rgb_full_range"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }

    long run_halide(OperationContext& ctx) override { return -1; }
    long run_opencv(OperationContext& ctx) override { return -1; }
};

REGISTER_OP(Nv21ToRgbFullRangeOp);
