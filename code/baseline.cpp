/**
 * \file
 * \brief Baseline benchmark evaluating the analytic expression without
 * automatic differentiation.
 *
 * The implementation in this file computes the primal and analytic derivatives
 * directly using doubles, providing a point of comparison for the automatic
 * differentiation implementations in the other benchmarks.
 */
#include <stdint.h>

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <ranges> // For std::views::reverse
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * \brief Benchmark that evaluates the expression without automatic
 * differentiation.
 *
 * The benchmark computes the primal expression and the manually derived
 * sensitivities to quantify the baseline computational cost absent any AD data
 * structures.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void baseline_bench(benchmark::State &state) {
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
