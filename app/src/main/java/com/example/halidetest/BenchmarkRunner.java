package com.example.halidetest;

import android.graphics.Bitmap;
import java.util.Arrays;

/**
 * Runs benchmark iterations for a given operation and collects timing statistics.
 */
public class BenchmarkRunner {

    public static class Result {
        public String operation;
        public String framework;
        public int width;
        public int height;
        public long[] timings;
        public long median;
        public long mean;
        public long min;
        public long max;

        public String toCsvLine() {
            return String.format("%s,%s,%dx%d,%d,%d,%d,%d",
                    operation, framework,
                    width, height,
                    median, mean, min, max);
        }

        public String toDisplayString() {
            return String.format(
                    "%s [%s] %dx%d\n" +
                    "  median: %,d us  (%.2f ms)\n" +
                    "  mean:   %,d us  (%.2f ms)\n" +
                    "  min:    %,d us  (%.2f ms)\n" +
                    "  max:    %,d us  (%.2f ms)\n" +
                    "  iterations: %d",
                    operation, framework, width, height,
                    median, median / 1000.0,
                    mean, mean / 1000.0,
                    min, min / 1000.0,
                    max, max / 1000.0,
                    timings.length);
        }

        private void computeStats() {
            if (timings == null || timings.length == 0) {
                median = mean = min = max = 0;
                return;
            }
            long[] sorted = timings.clone();
            Arrays.sort(sorted);
            min = sorted[0];
            max = sorted[sorted.length - 1];
            median = sorted[sorted.length / 2];
            long sum = 0;
            for (long t : sorted) sum += t;
            mean = sum / sorted.length;
        }
    }

    public interface Operation {
        long execute(boolean useHalide);
    }

    /**
     * Run a benchmark operation for the specified number of iterations.
     * @param opName display name of the operation
     * @param useHalide true for Halide, false for OpenCV
     * @param width image width
     * @param height image height
     * @param iterations number of iterations to run
     * @param op the operation to benchmark (lambda calling NativeBridge)
     * @return Result with timing statistics
     */
    public static Result run(String opName, boolean useHalide,
                             int width, int height, int iterations,
                             Operation op) {
        Result r = new Result();
        r.operation = opName;
        r.framework = useHalide ? "Halide" : "OpenCV";
        r.width = width;
        r.height = height;
        r.timings = new long[iterations];

        // Warm-up run (not counted)
        op.execute(useHalide);

        for (int i = 0; i < iterations; i++) {
            r.timings[i] = op.execute(useHalide);
        }

        r.computeStats();
        return r;
    }
}
