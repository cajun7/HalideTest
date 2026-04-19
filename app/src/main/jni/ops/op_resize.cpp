#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class ResizeBilinearOp : public IOperation {
public:
    const char* name() const override { return "resize_bilinear"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    bool changes_dimensions() const override { return true; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::resize_bilinear_optimized(ctx.h_in, ctx.dst_w, ctx.dst_h, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::resize_bilinear_optimized(ctx.cv_in, out, ctx.dst_w, ctx.dst_h);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(ResizeBilinearOp);

class ResizeBicubicOp : public IOperation {
public:
    const char* name() const override { return "resize_bicubic"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    bool changes_dimensions() const override { return true; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::resize_bicubic_optimized(ctx.h_in, ctx.dst_w, ctx.dst_h, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::resize_bicubic_optimized(ctx.cv_in, out, ctx.dst_w, ctx.dst_h);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(ResizeBicubicOp);

class ResizeAreaOp : public IOperation {
public:
    const char* name() const override { return "resize_area"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    bool changes_dimensions() const override { return true; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::resize_area_optimized(ctx.h_in, ctx.dst_w, ctx.dst_h, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::resize_area_optimized(ctx.cv_in, out, ctx.dst_w, ctx.dst_h);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(ResizeAreaOp);
