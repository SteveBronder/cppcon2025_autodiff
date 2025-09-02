// Type your code here, or load an example.
#include <stdint.h>

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
#include <functional>
#include <benchmark/benchmark.h>

static std::pmr::monotonic_buffer_resource mbr{1<<16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;

static alloc_t pa{&mbr};
struct var {
    double* values_;
    var(double x) : values_((double*)pa.allocate_bytes(sizeof(double) * 2)) {
      values_[0] = x;
      values_[1] = 0;
    }
    auto& adj() {
      return values_[0];
    }
    auto val() {
      return values_[1];
    }
};

template <typename F, typename... Exprs>
struct ad_expr {
   var ret_;
   std::tuple<Exprs...> exprs_;
   F f_;
   ad_expr(double x, F&& f, const Exprs&... exprs) :
     ret_(x),
     f_(std::forward<F>(f)),
     exprs_(exprs...) {}
   auto val() {
     return ret_.val();
   }
   auto& adj() {
     return ret_.adj();
   }
};
/**
 * Helper functions
 */
auto& adjoint(var x) { return x.adj(); }
template <typename F, typename... Exprs>
auto& adjoint(ad_expr<F, Exprs...>& x) { return x.adj(); }

auto value(var x) { return x.val(); }
template <typename F, typename... Exprs>
auto value(ad_expr<F, Exprs...>& x) { return x.val(); }

namespace detail {
  template <typename T>
  struct is_expr : std::false_type {};
  template <typename F, typename... Exprs>
  struct is_expr<ad_expr<F, Exprs...>> : std::true_type {};
}
template <typename T>
struct is_expr : detail::is_expr<std::decay_t<T>> {};
template <typename T>
constexpr bool is_expr_v = is_expr<T>::value;
#ifndef DEBUG_AD
void print_var(const char* name, var& ret, var x) {
    std::cout << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
              << std::endl;
    std::cout << name << " Op: (" << value(x) << ", " << adjoint(x) << ")"
              << std::endl;
}

void print_var(const char* name, var& ret, var x, var y) {
    std::cout << "\t\t" << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
              << std::endl;
    std::cout << "\t\t" << name << " OpL: (" << value(x) << ", " << adjoint(x) << ")"
              << std::endl;
    std::cout << "\t\t" << name << " OpR: (" << value(y) << ", " << adjoint(y) << ")"
              << std::endl;
}
#else
constexpr void print_var(const char* name, var& ret, var x) {}

constexpr void print_var(const char* name, var& ret, var x, var y) {}
#endif
template <typename T1, typename T2>
inline auto operator+(T1 lhs, T2 rhs) {
  return ad_expr{value(lhs) + value(rhs), [](auto&& ret, auto&& lhs, auto&& rhs) {
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret);
    }
  }, lhs, rhs};
}

template <typename T1, typename T2>
inline auto operator*(T1 lhs, T2 rhs) {
  return ad_expr{value(lhs) * value(rhs), [](auto&& ret, auto&& lhs, auto&& rhs) {
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret) * value(rhs);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret) * value(lhs);
    }
  }, lhs, rhs};
}

template <typename Expr>
inline auto log(Expr&& x) {
  return ad_expr{std::log(x.val()), [](auto&& ret, auto&& x) {
    x.adj() += ret.adj() / x.val();
  }, x};
}

template <typename Expr, typename... Exprs>
inline auto compute_f(Expr&& z, Exprs&&... exprs) {
    z.f_(z.ret_, exprs...);
}
template <typename... Exprs>
inline constexpr auto compute_f(var& z, Exprs&&... exprs) {}
template <typename... Exprs>
inline constexpr auto compute_f(const var& z, Exprs&&... exprs) {}

template <typename Expr>
constexpr void grad_inner(const var& z) {}
template <typename Expr>
constexpr void grad_inner(var& z) {}
template <typename Expr>
void grad_inner(Expr&& z) {
  if constexpr (is_expr_v<Expr>) {
    std::apply([&z](auto&&... args) {
      compute_f(z, args...);
    }, z.exprs_);
    std::apply([&z](auto&&... args) {
      (grad_inner(args),...);
    }, z.exprs_);
  }
}

template <typename Expr>
inline void grad_init(Expr z) {
  adjoint(z) = 1.0;
  std::apply([&z](auto&&... args) {
    compute_f(z, args...);
  }, z.exprs_);
  std::apply([&z](auto&&... args) {
    (grad_inner(args),...);
  }, z.exprs_);

}

template <typename Expr>
void grad(Expr&& z) {
  grad_init(z);
}
inline void clear_mem() {
    mbr.release();
}
static void sct_bench(benchmark::State& state) {
    for (auto _ : state) {
      var x(2.0);
      var y(4.0);
      auto z = x * log(y) + log(x * y) * y;
      grad(z);
      benchmark::DoNotOptimize(z);
      clear_mem();
    }
}


BENCHMARK(sct_bench);
