#include "bench_stress.h"

#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

namespace bench {

namespace {

// Each worker loops until stop.
// - 4 MB memcpy per outer iter (DRAM BW)
// - 8192-iter int MAC over 4 KB array (scalar + L1)
// Buffers are per-thread to avoid synchronized cache line bouncing.
void stress_loop(std::atomic<bool>* stop, uint64_t seed) {
    constexpr size_t BW_BYTES = 4 * 1024 * 1024;
    constexpr size_t MAC_WORDS = 1024;          // 4 KB
    constexpr int    MAC_ITERS = 8192;

    std::vector<uint8_t> src(BW_BYTES), dst(BW_BYTES);
    std::vector<int32_t> mac(MAC_WORDS);
    std::mt19937_64 rng(seed);
    for (auto& b : src) b = (uint8_t)(rng() & 0xFF);
    for (auto& w : mac) w = (int32_t)(rng() & 0x0000FFFF);

    int32_t acc = 1;
    while (!stop->load(std::memory_order_relaxed)) {
        std::memcpy(dst.data(), src.data(), BW_BYTES);
        for (int k = 0; k < MAC_ITERS; ++k) {
            for (size_t i = 0; i < MAC_WORDS; ++i) {
                acc = acc * 1103515245 + mac[i];
            }
        }
        // Prevent the MAC loop from being optimized out: fold acc into dst[0].
        dst[0] = (uint8_t)acc;
    }
    // Sink so the compiler can't DCE the whole loop.
    volatile uint8_t sink = dst[0];
    (void)sink;
}

}  // namespace

Stressor::Stressor(int num_threads) {
    if (num_threads <= 0) return;
    workers_.reserve((size_t)num_threads);
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(stress_loop, &stop_, 0xA5A5A5A5ull + (uint64_t)i);
    }
}

Stressor::~Stressor() { stop(); }

void Stressor::stop() {
    if (stop_.exchange(true)) return;
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

}  // namespace bench
