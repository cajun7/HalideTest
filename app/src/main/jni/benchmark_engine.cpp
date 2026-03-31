#include "benchmark_engine.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <ctime>

void compute_stats(BenchmarkResult& r) {
    if (r.timings_us.empty()) {
        r.median_us = r.mean_us = r.min_us = r.max_us = 0;
        return;
    }

    std::vector<long> sorted = r.timings_us;
    std::sort(sorted.begin(), sorted.end());

    r.min_us = sorted.front();
    r.max_us = sorted.back();
    r.median_us = sorted[sorted.size() / 2];
    r.mean_us = std::accumulate(sorted.begin(), sorted.end(), 0L) / (long)sorted.size();
}

std::string result_to_csv(const BenchmarkResult& r) {
    // Get current timestamp
    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    std::ostringstream oss;
    oss << r.operation << ","
        << r.framework << ","
        << r.width << "x" << r.height << ","
        << r.median_us << ","
        << r.mean_us << ","
        << r.min_us << ","
        << r.max_us << ","
        << timestamp;
    return oss.str();
}

std::string result_to_string(const BenchmarkResult& r) {
    std::ostringstream oss;
    oss << r.operation << " [" << r.framework << "] "
        << r.width << "x" << r.height << "\n"
        << "  median: " << r.median_us << " us\n"
        << "  mean:   " << r.mean_us << " us\n"
        << "  min:    " << r.min_us << " us\n"
        << "  max:    " << r.max_us << " us\n"
        << "  iterations: " << r.timings_us.size();
    return oss.str();
}
