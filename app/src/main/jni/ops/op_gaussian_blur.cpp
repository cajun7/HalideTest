#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../opencv_ops.h"

class GaussianBlurRgbOp : public IOperation {
public:
    const char* name() const override { return "gaussian_blur_rgb"; }
    InputFormat input_format() const override { return InputFormat::RGB_BITMAP; }

    long run_halide(OperationContext& ctx) override {
        auto start = Clock::now();
        halide_ops::gaussian_blur_rgb(ctx.h_in, ctx.h_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }

    long run_opencv(OperationContext& ctx) override {
        auto start = Clock::now();
        cv::Mat out_bgr;
        opencv_ops::gaussian_blur_rgb(ctx.cv_in, out_bgr, 5);
        out_bgr.copyTo(ctx.cv_out);
        auto end = Clock::now();
        return elapsed_us(start, end);
    }
};

REGISTER_OP(GaussianBlurRgbOp);
