#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct BenchmarkResult {
    std::string operation;
    std::string framework;    // "Halide" or "OpenCV"
    int width;
    int height;
    std::vector<long> timings_us;  // per-iteration timings in microseconds
    long median_us;
    long mean_us;
    long min_us;
    long max_us;
};

// Format a BenchmarkResult as a CSV line
std::string result_to_csv(const BenchmarkResult& r);

// Format a BenchmarkResult as a human-readable string
std::string result_to_string(const BenchmarkResult& r);

// Compute statistics from timing array and populate median/mean/min/max
void compute_stats(BenchmarkResult& r);
