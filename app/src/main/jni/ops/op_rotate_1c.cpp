#include "../operation_registry.h"
#include "rotate_fixed_1c_90cw.h"
#include "rotate_fixed_1c_180.h"
#include "rotate_fixed_1c_270cw.h"

// 1-channel planar fixed rotate (90CW / 180 / 270CW) for masks / alpha / depth.
// Operates on uint8_t 2-D buffers; does not fit the RGBA OperationContext,
// so the registry run_* paths are stubs. Bench / tests call the AOT
// pipelines directly via the dispatch below.

class Rotate1C90Op : public IOperation {
public:
    const char* name() const override { return "rotate_1c_90cw"; }
    InputFormat input_format() const override { return InputFormat::FLOAT_PLANAR; }
    bool changes_dimensions() const override { return true; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Rotate1C90Op);

class Rotate1C180Op : public IOperation {
public:
    const char* name() const override { return "rotate_1c_180"; }
    InputFormat input_format() const override { return InputFormat::FLOAT_PLANAR; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Rotate1C180Op);

class Rotate1C270Op : public IOperation {
public:
    const char* name() const override { return "rotate_1c_270cw"; }
    InputFormat input_format() const override { return InputFormat::FLOAT_PLANAR; }
    bool changes_dimensions() const override { return true; }
    long run_halide(OperationContext&) override { return -1; }
    long run_opencv(OperationContext&) override { return -1; }
};
REGISTER_OP(Rotate1C270Op);
