/**
 * \file
 * \brief Benchmark combining arena-backed Eigen matrices with lambda-based reverse-mode nodes.
 *
 * This benchmark exercises a hybrid AD pattern that materializes Eigen matrix
 * results into arena-managed temporaries while still storing reverse callbacks
 * as lambdas. It highlights how the tape integrates with Eigen's lazy
 * evaluation while ensuring matrix temporaries live long enough for the reverse
 * pass.
 */

#include <stdint.h>

#include <ad_ex/arena_matrix.hpp>
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

namespace ad {

// Trait detecting Eigen matrices whose scalar type is the lambda-based AD variable.
template <typename T>
inline constexpr bool is_matrix_var =
    is_eigen_v<std::decay_t<T>> && is_var_v<typename std::decay_t<T>::Scalar>;

// Concept describing Eigen matrices whose scalar is an automatic differentiation variable.
template <typename T>
concept MatrixVar =
    EigenMatrix<T> && is_var_v<typename std::decay_t<T>::Scalar>;

// Concept enforcing that all template parameters satisfy ::MatrixVar.
template <typename... Types>
concept AllMatrixVar = (MatrixVar<Types> && ...);

// Concept representing plain Eigen matrices with arithmetic scalar types.
template <typename T>
concept PlainMatrix =
    EigenMatrix<T> && std::is_arithmetic_v<typename std::decay_t<T>::Scalar>;

template <PlainMatrix T> inline decltype(auto) value(T &&x) { return x; }

/**
 * \brief Matrix multiplication for AD-aware Eigen matrices backed by arena storage.
 *
 * The multiplication materializes its result in an arena matrix and registers a
 * lambda that propagates reverse-mode sensitivities back into the operands.
 *
 * @tparam T1 Left-hand matrix type.
 * @tparam T2 Right-hand matrix type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Arena-backed matrix storing the primal result.
 */
template <typename T1, typename T2>
  requires AllMatrixVar<T1, T2>
inline auto operator*(const T1 &lhs, const T2 &rhs) {
  arena_t<T1> lhs_arena = lhs;
  arena_t<T2> rhs_arena = rhs;
  arena_matrix<Eigen::Matrix<var, -1, -1>> ret = value(lhs) * value(rhs);
  make_var(0.0, [lhs_arena, rhs_arena, ret](auto &&toss) mutable {
    lhs_arena.adj().array() +=
        (ret.adj_op() * rhs_arena.val_op().transpose()).array();
    rhs_arena.adj().array() +=
        (ret.adj_op() * lhs_arena.val_op().transpose()).array();
  });
  return ret;
}

/**
 * \brief Reduces an Eigen matrix of AD variables to a scalar by summation.
 *
 * The function captures the arena-backed argument to keep it alive until the
 * reverse sweep and accumulates sensitivities element-wise.
 *
 * @tparam T Matrix expression whose scalar type is an AD variable.
 * @param x Matrix to sum.
 * @return Automatic differentiation variable containing the sum.
 */
template <MatrixVar T> inline auto sum(T &&x) {
  arena_t<T> x_arena = x;
  return make_var(x_arena.val().sum().eval(), [x_arena](auto &&ret) mutable {
    x_arena.adj().array() += ret.adj();
  });
}

} // namespace ad

/**
 * \brief Benchmark combining arena-backed Eigen matrices with lambda callbacks.
 *
 * Each iteration constructs arena-managed Eigen matrices of AD variables,
 * multiplies them, and computes the gradient to evaluate the performance of the
 * arena-assisted lambda tape.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void lambda_eigen_special_bench(benchmark::State &state) {

  using matv = Eigen::Matrix<ad::var, -1, -1>;
  using matd = Eigen::Matrix<double, -1, -1>;
  const auto N = state.range(0);
  const auto X1_d = matd::Random(N, N);
  const auto X2_d = matd::Random(N, N);
  for (auto _ : state) {
    ad::arena_matrix<matv> X1(X1_d);
    ad::arena_matrix<matv> X2(X2_d);
    ad::var ret = (X1 * X2).sum();
    ad::grad(ret);
    benchmark::DoNotOptimize(ret);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_eigen_special_bench)->RangeMultiplier(2)->Range(1, 4096);
