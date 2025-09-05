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
inline constexpr bool is_matrix_var = is_eigen_v<std::decay_t<T>> && is_var_v<typename std::decay_t<T>::Scalar>;
template <typename T>
concept MatrixVar = Matrix<T> && is_var_v<typename std::decay_t<T>::Scalar>;

template <typename... Types>
concept AllMatrixVar = (MatrixVar<Types> && ...);
template <typename T>
concept PlainMatrix = Matrix<T> && std::is_arithmetic_v<typename std::decay_t<T>::Scalar>;

template <PlainMatrix T>
inline decltype(auto) value(T&& x) {
  return x;
}
template <typename T1, typename T2>
requires AllMatrixVar<T1, T2>
inline auto operator*(const T1& lhs, const T2& rhs) {
  arena_t<T1> lhs_arena = lhs;
  arena_t<T2> rhs_arena = rhs;
  arena_matrix<Eigen::Matrix<var, -1, -1>> ret = value(lhs) * value(rhs);
  make_var(0.0, [lhs_arena, rhs_arena, ret](auto&& toss) mutable {
    if constexpr (is_matrix_var<T1>) {
      lhs_arena.adj().array() += (ret.adj_op() * rhs_arena.val_op().transpose()).array();
    }
    if constexpr (is_matrix_var<T2>) {
      rhs_arena.adj().array() += (ret.adj_op() * lhs_arena.val_op().transpose()).array();
    }
  });
  return ret;
}
template <MatrixVar T>
inline auto sum(T&& x) {
  arena_t<T> x_arena = x;
  return make_var(x_arena.val().sum().eval(), [x_arena](auto&& ret) mutable {
    x_arena.adj().array() += ret.adj();
  });
}

}
static void lambda_eigen_special_bench(benchmark::State& state) {

  using matv = Eigen::Matrix<ad::var, -1, -1>;
  for (auto _ : state) {
    ad::arena_matrix<matv> X1 = matv::Random(4, 4);
    ad::arena_matrix<matv> X2 = matv::Random(4, 4);
    ad::var ret = (X1 * X2).sum();
    ad::grad(ret);
    ad::clear_mem();
  }
}
BENCHMARK(lambda_eigen_special_bench);
