#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class Rotate90Op : public IOperation {
public:
    const char* name() const override { return "rotate_90cw"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    bool changes_dimensions() const override { return true; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::rotate_90cw(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::rotate_90(ctx.cv_in, out);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(Rotate90Op);

class Rotate180Op : public IOperation {
public:
    const char* name() const override { return "rotate_180"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::rotate_180(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::rotate_180(ctx.cv_in, out);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(Rotate180Op);

class Rotate270Op : public IOperation {
public:
    const char* name() const override { return "rotate_270cw"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    bool changes_dimensions() const override { return true; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::rotate_270cw(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::rotate_270(ctx.cv_in, out);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(Rotate270Op);

class RotateArbitraryOp : public IOperation {
public:
    const char* name() const override { return "rotate_arbitrary"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }
    BufferLayout halide_layout() const override { return BufferLayout::PLANAR; }

    void set_angle_degrees(float deg) { angle_deg_ = deg; }

    long run_halide(OperationContext& ctx) override {
        float angle_rad = angle_deg_ * 3.14159265358979f / 180.0f;
        auto start = Clock::now();
        halide_ops::rotate_angle(ctx.h_in, angle_rad, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::rotate_angle(ctx.cv_in, out, angle_deg_);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

private:
    float angle_deg_ = 45.0f;
};

REGISTER_OP(RotateArbitraryOp);
