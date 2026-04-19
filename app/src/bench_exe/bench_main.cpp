// =============================================================================
// Standalone CLI benchmark — Halide AOT vs hand-rolled OpenCV/NEON reference.
//
// Designed to run on an Android device via adb push. No JNI, no gradle —
// just a static-linked aarch64 executable that prints one CSV row per run.
//
// Contention model: --stress=N spawns N background threads that thrash DRAM
// and the scalar pipeline. No affinity calls on anyone: we want the OS
// scheduler doing what it does in a real phone.
// =============================================================================

#include "bench_clock.h"
#include "bench_stress.h"

#include "HalideBuffer.h"
#include "../main/jni/bt709_neon_ref.h"

// Halide AOT headers we directly dispatch to from here.
#include "rotate_fixed_90cw.h"
#include "rotate_fixed_180.h"
#include "rotate_fixed_270cw.h"
#include "rotate_arbitrary.h"
#include "rotate_fixed_1c_90cw.h"
#include "rotate_fixed_1c_180.h"
#include "rotate_fixed_1c_270cw.h"
#include "nv21_to_rgb_bt709_full_range.h"
#include "nv21_resize_rgb_bt709_nearest.h"
#include "nv21_resize_rgb_bt709_bilinear.h"
#include "nv21_resize_rgb_bt709_area.h"
#include "nv21_resize_bilinear_optimized.h"
#include "nv21_resize_area_optimized.h"
#include "nv21_resize_nearest_optimized.h"


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using Halide::Runtime::Buffer;

namespace {

// -----------------------------------------------------------------------------
// CLI parsing
// -----------------------------------------------------------------------------
struct Args {
    std::string backend = "halide";     // halide | opencv_neon
    std::string op = "";
    std::string interp = "linear";       // nearest | linear | area
    int iters = 200;
    int warmup = 20;
    int width = 1920, height = 1080;
    int dst_w = 0, dst_h = 0;            // 0 = same as src (or rotated swap)
    int stress = 0;
    float angle_deg = 45.0f;
    int rotation_cw = 90;                // 90 / 180 / 270 for rotate_*
    uint64_t seed = 42;
    std::string input_file = "";
    bool csv_header = false;
};

bool starts_with(const std::string& s, const char* p) {
    size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}

bool parse_wxh(const std::string& s, int& w, int& h) {
    size_t pos = s.find('x');
    if (pos == std::string::npos) return false;
    w = std::atoi(s.substr(0, pos).c_str());
    h = std::atoi(s.substr(pos + 1).c_str());
    return w > 0 && h > 0;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (starts_with(s, "--backend="))     a.backend     = s.substr(10);
        else if (starts_with(s, "--op="))     a.op          = s.substr(5);
        else if (starts_with(s, "--interp=")) a.interp      = s.substr(9);
        else if (starts_with(s, "--iters="))  a.iters       = std::atoi(s.c_str() + 8);
        else if (starts_with(s, "--warmup=")) a.warmup      = std::atoi(s.c_str() + 9);
        else if (starts_with(s, "--stress=")) a.stress      = std::atoi(s.c_str() + 9);
        else if (starts_with(s, "--angle="))  a.angle_deg   = (float)std::atof(s.c_str() + 8);
        else if (starts_with(s, "--rot="))    a.rotation_cw = std::atoi(s.c_str() + 6);
        else if (starts_with(s, "--seed="))   a.seed        = (uint64_t)std::strtoull(s.c_str() + 7, nullptr, 10);
        else if (starts_with(s, "--input-file=")) a.input_file = s.substr(13);
        else if (starts_with(s, "--resolution=")) parse_wxh(s.substr(13), a.width, a.height);
        else if (starts_with(s, "--dst="))    parse_wxh(s.substr(6), a.dst_w, a.dst_h);
        else if (s == "--csv-header")         a.csv_header = true;
        else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: bench [options]\n"
                "  --backend=halide|opencv_neon\n"
                "  --op=rotate|rotate_1c|nv21_to_rgb|nv21_resize_rgb\n"
                "  --interp=nearest|linear|area   (nv21_resize_rgb only)\n"
                "  --resolution=WxH              (source, default 1920x1080)\n"
                "  --dst=WxH                     (destination, default = source or rotated)\n"
                "  --rot=90|180|270              (rotate / rotate_1c fixed angle)\n"
                "  --angle=DEG                    (rotate arbitrary only; rot=0 selects it)\n"
                "  --iters=N --warmup=M\n"
                "  --stress=0..8                  (# background stressor threads)\n"
                "  --seed=K\n"
                "  --input-file=PATH              (raw NV21 or RGB bytes; optional)\n"
                "  --csv-header                   (also print header line before the row)\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            std::exit(2);
        }
    }
    return a;
}

// -----------------------------------------------------------------------------
// Input data
// -----------------------------------------------------------------------------
std::vector<uint8_t> make_random(size_t n, uint64_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t)(rng() & 0xFF);
    return v;
}

bool load_file(const std::string& path, std::vector<uint8_t>& out, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.resize(expected);
    f.read((char*)out.data(), (std::streamsize)expected);
    return (size_t)f.gcount() == expected;
}

// -----------------------------------------------------------------------------
// Runners (per-iteration closures)
// -----------------------------------------------------------------------------
using RunFn = void();

struct Runner {
    void (*fn)(void*);
    void* user;
    void operator()() const { fn(user); }
};

// -------- rotate (3-channel interleaved RGB) --------
struct RotateCtx {
    Buffer<uint8_t> in, out;       // interleaved
    cv::Mat in_mat, out_mat;       // CV_8UC3 view
    int rot_cw;
    float angle_rad;
    bool is_halide;
    bool arbitrary;
};

void run_rotate(void* p) {
    auto* c = (RotateCtx*)p;
    if (c->is_halide) {
        if (c->arbitrary) {
            ::rotate_arbitrary(c->in, c->angle_rad, c->out);
        } else {
            switch (c->rot_cw) {
                case 90:  ::rotate_fixed_90cw (c->in, c->out); break;
                case 180: ::rotate_fixed_180  (c->in, c->out); break;
                case 270: ::rotate_fixed_270cw(c->in, c->out); break;
            }
        }
    } else {
        if (c->arbitrary) {
            cv::Point2f center(c->in_mat.cols * 0.5f, c->in_mat.rows * 0.5f);
            cv::Mat M = cv::getRotationMatrix2D(center, (double)(c->angle_rad * 57.2957795131), 1.0);
            cv::warpAffine(c->in_mat, c->out_mat, M, c->in_mat.size(),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else {
            int code;
            switch (c->rot_cw) {
                case 90:  code = cv::ROTATE_90_CLOCKWISE; break;
                case 180: code = cv::ROTATE_180; break;
                case 270: code = cv::ROTATE_90_COUNTERCLOCKWISE; break;
                default:  code = cv::ROTATE_90_CLOCKWISE; break;
            }
            cv::rotate(c->in_mat, c->out_mat, code);
        }
    }
}

// -------- rotate_1c (1-channel planar) --------
struct Rotate1CCtx {
    Buffer<uint8_t> in, out;       // planar 2D
    cv::Mat in_mat, out_mat;       // CV_8UC1
    int rot_cw;
    bool is_halide;
};

void run_rotate_1c(void* p) {
    auto* c = (Rotate1CCtx*)p;
    if (c->is_halide) {
        switch (c->rot_cw) {
            case 90:  ::rotate_fixed_1c_90cw (c->in, c->out); break;
            case 180: ::rotate_fixed_1c_180  (c->in, c->out); break;
            case 270: ::rotate_fixed_1c_270cw(c->in, c->out); break;
        }
    } else {
        int code;
        switch (c->rot_cw) {
            case 90:  code = cv::ROTATE_90_CLOCKWISE; break;
            case 180: code = cv::ROTATE_180; break;
            case 270: code = cv::ROTATE_90_COUNTERCLOCKWISE; break;
            default:  code = cv::ROTATE_90_CLOCKWISE; break;
        }
        cv::rotate(c->in_mat, c->out_mat, code);
    }
}

// -------- nv21 -> BT.709 RGB (size unchanged) --------
struct Nv21RgbCtx {
    Buffer<uint8_t> y_buf, uv_buf, rgb_buf;
    const uint8_t* y_ptr;  int y_stride;
    const uint8_t* uv_ptr; int uv_stride;
    uint8_t*       rgb_ptr; int rgb_stride;
    int w, h;
    bool is_halide;
};

void run_nv21_rgb(void* p) {
    auto* c = (Nv21RgbCtx*)p;
    if (c->is_halide) {
        ::nv21_to_rgb_bt709_full_range(c->y_buf, c->uv_buf, c->rgb_buf);
    } else {
        bt709::nv21_to_rgb_bt709_full_range_neon(
            c->y_ptr, c->y_stride, c->uv_ptr, c->uv_stride,
            c->rgb_ptr, c->rgb_stride, c->w, c->h);
    }
}

// -------- nv21 -> resize + BT.709 RGB (fused halide vs opencv_neon 2-step) --------
struct Nv21ResizeRgbCtx {
    // Halide side
    Buffer<uint8_t> y_buf, uv_buf, rgb_buf;
    // OpenCV-neon side intermediate NV21 (resized) + scratch RGB
    Buffer<uint8_t> y_resized, uv_resized;
    const uint8_t* y_ptr;  int y_stride;
    const uint8_t* uv_ptr; int uv_stride;
    uint8_t*       rgb_ptr; int rgb_stride;
    std::vector<uint8_t>   y_r_storage, uv_r_storage;
    int src_w, src_h;
    int dst_w, dst_h;
    int interp;   // 0=nearest 1=linear 2=area
    bool is_halide;
};

void run_nv21_resize_rgb(void* p) {
    auto* c = (Nv21ResizeRgbCtx*)p;
    if (c->is_halide) {
        switch (c->interp) {
            case 0: ::nv21_resize_rgb_bt709_nearest (c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->rgb_buf); break;
            case 1: ::nv21_resize_rgb_bt709_bilinear(c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->rgb_buf); break;
            case 2: ::nv21_resize_rgb_bt709_area    (c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->rgb_buf); break;
        }
    } else {
        // Non-fused: Halide NV21 resize (bit-identical to fused resize step)
        // -> NEON BT.709 conversion. This is the "two-pass OpenCV-style" path
        // whose extra memory traffic is exactly what fusion saves.
        switch (c->interp) {
            case 0: ::nv21_resize_nearest_optimized (c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->y_resized, c->uv_resized); break;
            case 1: ::nv21_resize_bilinear_optimized(c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->y_resized, c->uv_resized); break;
            case 2: ::nv21_resize_area_optimized    (c->y_buf, c->uv_buf, c->dst_w, c->dst_h, c->y_resized, c->uv_resized); break;
        }
        bt709::nv21_to_rgb_bt709_full_range_neon(
            c->y_r_storage.data(),  c->dst_w,
            c->uv_r_storage.data(), c->dst_w,
            c->rgb_ptr, c->rgb_stride, c->dst_w, c->dst_h);
    }
}

// -----------------------------------------------------------------------------
// Main driver loop
// -----------------------------------------------------------------------------
bench::Stats time_it(int warmup, int iters, void (*fn)(void*), void* user) {
    for (int i = 0; i < warmup; ++i) fn(user);
    std::vector<uint64_t> samples;
    samples.reserve((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = bench::Clock::now_ns();
        fn(user);
        uint64_t t1 = bench::Clock::now_ns();
        samples.push_back((t1 - t0 + 500) / 1000);   // ns->us
    }
    return bench::Stats::from(samples);
}

void print_header() {
    std::printf("backend,op,interp,src_w,src_h,dst_w,dst_h,rot,stress,iters,"
                "min_us,mean_us,trimmed_mean_us,stddev_us,cv,"
                "p50_us,p95_us,p99_us,max_us\n");
}

void print_row(const Args& a, int dst_w, int dst_h, const bench::Stats& s) {
    std::printf("%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
                "%llu,%.2f,%.2f,%.2f,%.4f,"
                "%llu,%llu,%llu,%llu\n",
                a.backend.c_str(), a.op.c_str(), a.interp.c_str(),
                a.width, a.height, dst_w, dst_h, a.rotation_cw,
                a.stress, a.iters,
                (unsigned long long)s.min_us, s.mean_us, s.trimmed_mean_us,
                s.stddev_us, s.cv,
                (unsigned long long)s.p50_us, (unsigned long long)s.p95_us,
                (unsigned long long)s.p99_us, (unsigned long long)s.max_us);
    std::fflush(stdout);
}

int interp_code(const std::string& s) {
    if (s == "nearest") return 0;
    if (s == "linear" || s == "bilinear") return 1;
    if (s == "area")    return 2;
    std::fprintf(stderr, "unknown interp: %s\n", s.c_str());
    std::exit(2);
}

}  // namespace

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.csv_header) print_header();
    if (a.op.empty()) {
        std::fprintf(stderr, "error: --op is required (see --help)\n");
        return 2;
    }
    const bool is_halide = (a.backend == "halide");
    if (!is_halide && a.backend != "opencv_neon") {
        std::fprintf(stderr, "error: backend must be halide or opencv_neon\n");
        return 2;
    }

    bench::Stressor stressor(a.stress);

    if (a.op == "rotate") {
        const int W = a.width, H = a.height;
        std::vector<uint8_t> src = make_random((size_t)W * H * 3, a.seed);
        if (!a.input_file.empty()) load_file(a.input_file, src, (size_t)W * H * 3);

        bool arbitrary = (a.rotation_cw == 0);
        int ow = (arbitrary || a.rotation_cw == 180) ? W : H;
        int oh = (arbitrary || a.rotation_cw == 180) ? H : W;
        std::vector<uint8_t> dst((size_t)ow * oh * 3);

        RotateCtx c{};
        c.in  = Buffer<uint8_t>::make_interleaved(src.data(), W, H, 3);
        c.out = Buffer<uint8_t>::make_interleaved(dst.data(), ow, oh, 3);
        c.in_mat  = cv::Mat(H,  W,  CV_8UC3, src.data());
        c.out_mat = cv::Mat(oh, ow, CV_8UC3, dst.data());
        c.rot_cw = a.rotation_cw;
        c.angle_rad = a.angle_deg * 3.14159265358979f / 180.0f;
        c.is_halide = is_halide;
        c.arbitrary = arbitrary;

        auto s = time_it(a.warmup, a.iters, run_rotate, &c);
        print_row(a, ow, oh, s);

    } else if (a.op == "rotate_1c") {
        const int W = a.width, H = a.height;
        std::vector<uint8_t> src = make_random((size_t)W * H, a.seed);
        if (!a.input_file.empty()) load_file(a.input_file, src, (size_t)W * H);

        int ow = (a.rotation_cw == 180) ? W : H;
        int oh = (a.rotation_cw == 180) ? H : W;
        std::vector<uint8_t> dst((size_t)ow * oh);

        Rotate1CCtx c{};
        c.in  = Buffer<uint8_t>(src.data(), W, H);
        c.out = Buffer<uint8_t>(dst.data(), ow, oh);
        c.in_mat  = cv::Mat(H,  W,  CV_8UC1, src.data());
        c.out_mat = cv::Mat(oh, ow, CV_8UC1, dst.data());
        c.rot_cw = a.rotation_cw;
        c.is_halide = is_halide;

        auto s = time_it(a.warmup, a.iters, run_rotate_1c, &c);
        print_row(a, ow, oh, s);

    } else if (a.op == "nv21_to_rgb") {
        const int W = a.width, H = a.height;
        size_t ybytes = (size_t)W * H;
        size_t uvbytes = (size_t)W * (H / 2);
        std::vector<uint8_t> y_src  = make_random(ybytes,  a.seed);
        std::vector<uint8_t> uv_src = make_random(uvbytes, a.seed ^ 0xABCDEF);
        if (!a.input_file.empty()) {
            std::vector<uint8_t> full;
            if (load_file(a.input_file, full, ybytes + uvbytes)) {
                std::memcpy(y_src.data(),  full.data(),          ybytes);
                std::memcpy(uv_src.data(), full.data() + ybytes, uvbytes);
            }
        }
        std::vector<uint8_t> rgb((size_t)W * H * 3);

        Nv21RgbCtx c{};
        c.y_buf  = Buffer<uint8_t>(y_src.data(),  W, H);
        c.uv_buf = Buffer<uint8_t>(uv_src.data(), W, H / 2);
        c.rgb_buf = Buffer<uint8_t>::make_interleaved(rgb.data(), W, H, 3);
        c.y_ptr = y_src.data();   c.y_stride  = W;
        c.uv_ptr = uv_src.data(); c.uv_stride = W;
        c.rgb_ptr = rgb.data();   c.rgb_stride = W * 3;
        c.w = W; c.h = H;
        c.is_halide = is_halide;

        auto s = time_it(a.warmup, a.iters, run_nv21_rgb, &c);
        print_row(a, W, H, s);

    } else if (a.op == "nv21_resize_rgb") {
        const int W = a.width, H = a.height;
        int dw = a.dst_w > 0 ? a.dst_w : W / 2;
        int dh = a.dst_h > 0 ? a.dst_h : H / 2;
        size_t ybytes = (size_t)W * H;
        size_t uvbytes = (size_t)W * (H / 2);
        std::vector<uint8_t> y_src  = make_random(ybytes,  a.seed);
        std::vector<uint8_t> uv_src = make_random(uvbytes, a.seed ^ 0xABCDEF);
        if (!a.input_file.empty()) {
            std::vector<uint8_t> full;
            if (load_file(a.input_file, full, ybytes + uvbytes)) {
                std::memcpy(y_src.data(),  full.data(),          ybytes);
                std::memcpy(uv_src.data(), full.data() + ybytes, uvbytes);
            }
        }
        std::vector<uint8_t> rgb((size_t)dw * dh * 3);
        std::vector<uint8_t> y_r((size_t)dw * dh);
        std::vector<uint8_t> uv_r((size_t)dw * (dh / 2));

        Nv21ResizeRgbCtx c{};
        c.y_buf  = Buffer<uint8_t>(y_src.data(),  W, H);
        c.uv_buf = Buffer<uint8_t>(uv_src.data(), W, H / 2);
        c.rgb_buf = Buffer<uint8_t>::make_interleaved(rgb.data(), dw, dh, 3);
        c.y_resized  = Buffer<uint8_t>(y_r.data(),  dw, dh);
        c.uv_resized = Buffer<uint8_t>(uv_r.data(), dw, dh / 2);
        c.y_ptr = y_src.data();   c.y_stride  = W;
        c.uv_ptr = uv_src.data(); c.uv_stride = W;
        c.rgb_ptr = rgb.data();   c.rgb_stride = dw * 3;
        c.y_r_storage = std::move(y_r);
        c.uv_r_storage = std::move(uv_r);
        c.src_w = W; c.src_h = H;
        c.dst_w = dw; c.dst_h = dh;
        c.interp = interp_code(a.interp);
        c.is_halide = is_halide;

        auto s = time_it(a.warmup, a.iters, run_nv21_resize_rgb, &c);
        print_row(a, dw, dh, s);

    } else {
        std::fprintf(stderr, "error: unknown --op=%s\n", a.op.c_str());
        return 2;
    }

    stressor.stop();
    return 0;
}
