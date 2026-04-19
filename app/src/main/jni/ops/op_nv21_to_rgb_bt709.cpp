#include "../operation_registry.h"
#include "../halide_ops.h"
#include "../bt709_neon_ref.h"
#include "nv21_to_rgb_bt709_full_range.h"

// NV21 -> BT.709 full-range RGB.
// Uses NV21Context at the native_bridge / bench layer; the OperationContext
// path here is a stub (NV21 ops don't fit OperationContext's RGBA-bitmap shape),
// matching the existing convention in op_nv21_to_rgb.cpp.

class Nv21ToRgbBt709FullRangeOp : public IOperation {
public:
    const char* name() const override { return "nv21_to_rgb_bt709_full_range"; }
    InputFormat input_format() const override { return InputFormat::NV21_BYTE_ARRAY; }

    long run_halide(OperationContext& ctx) override { return -1; }
    long run_opencv(OperationContext& ctx) override { return -1; }
};

REGISTER_OP(Nv21ToRgbBt709FullRangeOp);

namespace bench_bt709 {

// Direct entry points used by the standalone benchmark executable and
// any other caller that already owns NV21 planes + RGB output buffer.
// These do NOT time themselves — the caller owns the clock.
inline int halide_nv21_to_rgb_bt709(Halide::Runtime::Buffer<uint8_t>& y,
                                    Halide::Runtime::Buffer<uint8_t>& uv,
                                    Halide::Runtime::Buffer<uint8_t>& rgb) {
    return ::nv21_to_rgb_bt709_full_range(y, uv, rgb);
}

inline void neon_nv21_to_rgb_bt709(const uint8_t* y, int y_stride,
                                   const uint8_t* uv, int uv_stride,
                                   uint8_t* rgb, int rgb_stride,
                                   int w, int h) {
    bt709::nv21_to_rgb_bt709_full_range_neon(y, y_stride, uv, uv_stride,
                                             rgb, rgb_stride, w, h);
}

}  // namespace bench_bt709
