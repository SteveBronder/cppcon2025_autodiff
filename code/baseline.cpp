// Type your code here, or load an example.
#include <stdint.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <memory_resource>
#include <ranges> // For std::views::reverse
#include <functional>
#include <benchmark/benchmark.h>

static void baseline_bench(benchmark::State& state) {
    double x(2.0);
    double y(4.0);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    for (auto _ : state) {
      double z_fwd = x * std::log(y) + std::log(x * y) * y;
      double x_rev = y / x + std::log(y);
      double y_rev = x / y + std::log(x * y) + 1;
      benchmark::DoNotOptimize(z_fwd);
      benchmark::DoNotOptimize(x_rev);
      benchmark::DoNotOptimize(y_rev);
    }
}


BENCHMARK(baseline_bench);
