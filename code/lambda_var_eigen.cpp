#include <stdint.h>

#include <benchmark/benchmark.h>
#include <ad_ex/meta/Eigen.hpp>
#include <ad_ex/lambda.hpp>
#include <ad_ex/eigen_numtraits.hpp>
#include <ad_ex/arena_matrix.hpp>
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

namespace ad {

template <typename T>
requires Matrix<T>
struct var_base<T>  : public var_base_chain {
  arena_matrix<T> value_;
  arena_matrix<T> adjoint_;
  var_base(const T& x)
      : var_base_chain(),
        value_(x),
        adjoint_(value_.rows(), value_.cols()) {
    adjoint_.setZero();
  }
};

template <typename T>
inline constexpr bool is_matrix_var = is_eigen_v<std::decay_t<T>> && is_var_v<typename std::decay_t<T>::Scalar>;
template <typename T>
inline constexpr bool is_var_matrix = is_eigen_v<typename std::decay_t<T>::value_type> && is_var_v<std::decay_t<T>>;
template <typename T>
concept VarMatrix = Matrix<typename std::decay_t<T>::value_type> && is_var_v<std::decay_t<T>>;
template <typename... Types>
concept AllVarMatrix = (VarMatrix<Types> && ...);


template <typename T>
concept MatrixVar = Matrix<T> && is_var_v<typename std::decay_t<T>::Scalar>;
template <typename... Types>
concept AllMatrixVar = (MatrixVar<Types> && ...);

template <typename T>
concept PlainMatrix = Matrix<T> && std::is_arithmetic_v<typename std::decay_t<T>::Scalar>;

template <typename T>
concept RevMatrix = MatrixVar<T> || VarMatrix<T>;
template <PlainMatrix T>
inline decltype(auto) value(T&& x) {
  return x;
}
template <typename T1, typename T2>
inline auto multiply(const T1& lhs, const T2& rhs) {
  return make_var((value(lhs) * value(rhs)).eval(), [lhs, rhs](auto&& ret) mutable {
    if constexpr (is_var_matrix<T1>) {
      lhs.adj().array() += (ret.adj_op() * rhs.val_op().transpose()).array();
    }
    if constexpr (is_var_matrix<T2>) {
      rhs.adj().array() += (ret.adj_op() * lhs.val_op().transpose()).array();
    }
  });
}
template <typename T>
inline auto sum(T&& x) {
  return make_var(x.val().sum(), [x](auto&& ret) mutable {
    x.adj().array() += ret.adj();
  });
}

}
static void lambda_var_eigen(benchmark::State& state) {

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
    ad::clear_mem();
  }
}
BENCHMARK(lambda_var_eigen)-> RangeMultiplier(2) -> Range(1, 4096);

