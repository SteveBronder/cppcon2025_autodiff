#include <stdint.h>

#include <benchmark/benchmark.h>
#include <ad_ex/meta/Eigen.hpp>
#include <ad_ex/lambda.hpp>
#include <ad_ex/eigen_numtraits.hpp>
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
static void lambda_eigen_bench(benchmark::State& state) {

  using matv = Eigen::Matrix<ad::var, -1, -1>;
  using matd = Eigen::Matrix<double, -1, -1>;
  const auto N = state.range(0);
  const auto X1_d = matd::Random(N, N);
  const auto X2_d = matd::Random(N, N);
  for (auto _ : state) {
    matv X1(X1_d);
    matv X2(X2_d);
    ad::var ret = (X1 * X2).sum();
    ad::grad(ret);
    benchmark::DoNotOptimize(ret);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_eigen_bench)-> RangeMultiplier(2) -> Range(1, 512);
