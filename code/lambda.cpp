/**
 * \file
 * \brief Benchmark for the lambda-based dynamic reverse-mode automatic differentiation tape.
 *
 * This benchmark exercises the custom AD system implemented in
 * `ad_ex/lambda.hpp`, where each operation creates a lambda that captures the
 * local gradient propagation. Nodes are stored in a monotonic buffer, yielding a
 * lightweight dynamic tape.
 */
#include <ad_ex/lambda.hpp>
#include <benchmark/benchmark.h>

/**
 * \brief Benchmark that evaluates the lambda-based reverse-mode tape.
 *
 * Each iteration constructs a computation graph using lambda callbacks for the
 * reverse pass, runs the gradient sweep, and clears the arena-backed tape.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void lambda_bench(benchmark::State &state) {
  for (auto _ : state) {
    ad::var x(2.0);
    ad::var y(4.0);
    auto z = x * log(y) + log(x * y) * y;
    ad::grad(z);
    benchmark::DoNotOptimize(z);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_bench);
