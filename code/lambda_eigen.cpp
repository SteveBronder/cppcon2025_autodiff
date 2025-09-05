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
  for (auto _ : state) {
    matv X1 = matv::Random(4, 4);
    matv X2 = matv::Random(4, 4);
    ad::var ret = (X1 * X2).sum();
    ad::grad(ret);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_eigen_bench);
