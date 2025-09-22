/**
 * \file
 * \brief Benchmark wiring the lambda-based dynamic tape into Eigen matrix operations.
 *
 * The benchmark demonstrates how the lambda-backed reverse-mode nodes from
 * `ad_ex/lambda.hpp` interact with Eigen matrices via the specialized
 * numtraits, enabling element-wise AD through Eigen expressions while storing
 * the tape in a monotonic buffer.
 */
#include <stdint.h>

#include <ad_ex/eigen_numtraits.hpp>
#include <ad_ex/lambda.hpp>
#include <ad_ex/meta/Eigen.hpp>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <ranges> // For std::views::reverse
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * \brief Benchmark that multiplies Eigen matrices using the lambda-based tape.
 *
 * The benchmark repeatedly instantiates Eigen matrices of AD variables, forms a
 * matrix product, and triggers reverse mode to evaluate the cost of coupling the
 * lambda tape with Eigen expressions.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void lambda_eigen_bench(benchmark::State &state) {

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
BENCHMARK(lambda_eigen_bench)->RangeMultiplier(2)->Range(1, 512);
