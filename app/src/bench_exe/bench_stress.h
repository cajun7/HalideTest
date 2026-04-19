#pragma once
// Background stressor threads for realistic benchmark measurement.
//
// Motivation: benchmarking on an idle CPU produces numbers that never happen
// in production. The foreground camera app lives in an ecosystem with other
// threads (encoder, preview compositor, ML inference) competing for:
//   - DRAM bandwidth
//   - Scalar/FPU issue slots
//   - L2 cache
//
// Each stressor thread per iteration:
//   1. memcpy a ~4 MB buffer (DRAM bandwidth contention)
//   2. 8192-iter integer MAC spin on a 4 KB array (scalar issue + L1 reuse)
//
// Stressors deliberately do NOT pin to cores — per project direction, we want
// the OS scheduler doing its normal work so the result reflects real contention.

#include <atomic>
#include <thread>
#include <vector>
#include <cstddef>

namespace bench {

class Stressor {
public:
    // num_threads = 0 → no-op.
    explicit Stressor(int num_threads);
    ~Stressor();
    void stop();

    Stressor(const Stressor&) = delete;
    Stressor& operator=(const Stressor&) = delete;

private:
    std::atomic<bool> stop_{false};
    std::vector<std::thread> workers_;
};

}  // namespace bench
