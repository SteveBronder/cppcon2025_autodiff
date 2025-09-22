/**
 * \file
 * \brief Benchmark specializing the lambda-based tape for matrix-valued variables.
 *
 * The implementation tailors the dynamic lambda tape to handle Eigen matrices as
 * primary variable values. Each matrix variable stores its value and adjoint in
 * arena-backed matrices, while scalar operations still reuse the lambda-based
 * callback mechanism.
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

/**
 * \brief Specialization of the lambda variable base for Eigen matrix values.
 *
 * Each matrix variable stores value and adjoint data in arena-managed matrices
 * to avoid per-iteration allocations during gradient computation.
 *
 * @tparam T Eigen matrix type satisfying ::EigenMatrix.
 */
template <typename T>
  requires EigenMatrix<T>
struct var_base<T> : public var_base_chain {
  arena_matrix<T> value_;
  arena_matrix<T> adjoint_;
  var_base(const T &x)
      : var_base_chain(), value_(x), adjoint_(value_.rows(), value_.cols()) {
    adjoint_.setZero();
  }
  inline auto &val() { return value_; }
  inline auto &adj() { return adjoint_; }
};

template <typename T>
inline constexpr bool is_matrix_var =
    is_eigen_v<std::decay_t<T>> && is_var_v<typename std::decay_t<T>::Scalar>;

template <typename T>
inline constexpr bool is_var_matrix =
    is_eigen_v<typename std::decay_t<T>::value_type>;

template <typename T>
concept VarMatrix = EigenMatrix<typename std::decay_t<T>::value_type> &&
                    is_var_v<std::decay_t<T>>;

template <typename... Types>
concept AllVarMatrix = (VarMatrix<Types> && ...);

template <typename T>
concept MatrixVar =
    EigenMatrix<T> && is_var_v<typename std::decay_t<T>::Scalar>;

template <typename... Types>
concept AllMatrixVar = (MatrixVar<Types> && ...);

template <typename T>
concept PlainMatrix =
    EigenMatrix<T> && std::is_arithmetic_v<typename std::decay_t<T>::Scalar>;

template <typename T>
concept RevMatrix = MatrixVar<T> || VarMatrix<T>;

template <PlainMatrix T> inline decltype(auto) value(T &&x) { return x; }


/**
 * \brief Performs matrix multiplication between reverse-mode matrices.
 *
 * The function allocates a new lambda variable representing the product and
 * registers a lambda that propagates adjoints into both operands.
 *
 * @tparam T1 Left operand type satisfying ::RevMatrix.
 * @tparam T2 Right operand type satisfying ::RevMatrix.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Automatic differentiation variable holding the product.
 */
template <typename T1, typename T2> inline auto multiply(T1 &&lhs, T2 &&rhs) {
  return make_var(
      (value(lhs) * value(rhs)).eval(), [lhs, rhs](auto &&ret) mutable {
        lhs.adj().array() += (ret.adj() * rhs.val().transpose()).array();
        rhs.adj().array() += (ret.adj() * lhs.val().transpose()).array();
      });
}

/**
 * \brief Reduces a matrix variable to a scalar by summing its entries.
 *
 * @tparam T Matrix expression holding reverse-mode variables.
 * @param x Matrix expression to sum.
 * @return Automatic differentiation variable holding the sum.
 */
template <typename T> inline auto sum(T &&x) {
  return make_var(x.val().sum(),
                  [x](auto &&ret) mutable { x.adj().array() += ret.adj(); });
}

} // namespace ad

/**
 * \brief Benchmark specializing lambda variables for matrix storage.
 *
 * Each iteration constructs lambda variables whose values are Eigen matrices,
 * executes a matrix product, and performs a gradient sweep to measure the
 * overhead of storing matrices directly inside the lambda tape.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void lambda_var_eigen(benchmark::State &state) {

  using mat_d = Eigen::Matrix<double, -1, -1>;
  using v_mat = ad::var_impl<mat_d>;
  const auto N = state.range(0);
  auto X1_d = mat_d::Random(N, N);
  auto X2_d = mat_d::Random(N, N);
  for (auto _ : state) {
    v_mat X1(X1_d);
    v_mat X2(X2_d);
    ad::var ret = ad::sum(ad::multiply(X1, X2));
    ad::grad(ret);
    benchmark::DoNotOptimize(ret);
    benchmark::DoNotOptimize(X1);
    benchmark::DoNotOptimize(X2);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_var_eigen)->RangeMultiplier(2)->Range(1, 4096);
