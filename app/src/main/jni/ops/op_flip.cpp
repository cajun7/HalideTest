#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class FlipHorizontalOp : public IOperation {
public:
    const char* name() const override { return "flip_horizontal"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::flip_horizontal(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::flip_horizontal(ctx.cv_in, out);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(FlipHorizontalOp);

class FlipVerticalOp : public IOperation {
public:
    const char* name() const override { return "flip_vertical"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::flip_vertical(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::flip_vertical(ctx.cv_in, out);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(FlipVerticalOp);
