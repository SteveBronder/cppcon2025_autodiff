#include <ad_ex/lambda.hpp>
#include <benchmark/benchmark.h>
static void lambda_bench(benchmark::State& state) {
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
