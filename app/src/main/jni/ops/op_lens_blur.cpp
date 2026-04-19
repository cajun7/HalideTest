#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class LensBlurOp : public IOperation {
public:
    const char* name() const override { return "lens_blur"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }

    void set_radius(int r) { radius_ = r; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::lens_blur(ctx.h_in, radius_, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out;
        opencv_ops::lens_blur(ctx.cv_in, out, radius_);
        out.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

private:
    int radius_ = 4;
};

REGISTER_OP(LensBlurOp);
