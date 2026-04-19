#pragma once
// CLOCK_MONOTONIC_RAW: hardware-monotonic, not affected by NTP/settimeofday.
// On Android this is the clock the Camera2 framework uses for frame timestamps,
// which is the closest thing to "what the user actually experiences."

#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace bench {

struct Clock {
    static uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    }
};

struct Stats {
    uint64_t min_us, max_us;
    double mean_us, stddev_us;
    double trimmed_mean_us;   // mean after dropping top/bottom 5%
    double cv;                // coefficient of variation = stddev / mean
    uint64_t p50_us, p95_us, p99_us;

    static Stats from(std::vector<uint64_t>& us) {
        Stats s{};
        if (us.empty()) return s;
        std::vector<uint64_t> sorted = us;
        std::sort(sorted.begin(), sorted.end());
        s.min_us = sorted.front();
        s.max_us = sorted.back();
        double sum = 0;
        for (uint64_t v : sorted) sum += (double)v;
        s.mean_us = sum / (double)sorted.size();
        double sq = 0;
        for (uint64_t v : sorted) {
            double d = (double)v - s.mean_us;
            sq += d * d;
        }
        s.stddev_us = std::sqrt(sq / (double)sorted.size());
        s.cv = (s.mean_us > 0.0) ? (s.stddev_us / s.mean_us) : 0.0;
        // Trimmed mean: drop top 5% and bottom 5%, average the middle. For N<20
        // the trim collapses to the plain mean (one sample each side already
        // exceeds 5%), which is fine — small samples don't give useful tails.
        size_t n = sorted.size();
        size_t trim = n / 20;   // floor(5%)
        if (trim * 2 >= n) {
            s.trimmed_mean_us = s.mean_us;
        } else {
            double tsum = 0;
            for (size_t i = trim; i < n - trim; ++i) tsum += (double)sorted[i];
            s.trimmed_mean_us = tsum / (double)(n - 2 * trim);
        }
        // Nearest-rank percentile (matches Go/Python numpy default "lower").
        auto pct = [&](double p) -> uint64_t {
            if (sorted.empty()) return 0;
            size_t idx = (size_t)std::ceil(p * (double)sorted.size()) - 1;
            if (idx >= sorted.size()) idx = sorted.size() - 1;
            return sorted[idx];
        };
        s.p50_us = pct(0.50);
        s.p95_us = pct(0.95);
        s.p99_us = pct(0.99);
        return s;
    }
};

}  // namespace bench
