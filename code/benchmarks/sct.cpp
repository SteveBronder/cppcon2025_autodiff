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

// Wrap an object in a tuple if it's an ad_expr, otherwise an empty tuple.
template <typename T>
constexpr auto make_expr_tuple(T& x) {
  if constexpr (is_expr_v<std::decay_t<T>>) {
    return std::tuple<std::reference_wrapper<std::decay_t<T>>>{std::ref(x)};
  } else {
    return std::tuple<>();
  }
}

// Given an ad_expr node, return a tuple of its children that are ad_exprs.
template <typename E>
constexpr auto child_exprs(E& e) {
  static_assert(is_expr_v<std::decay_t<E>>,
                "child_exprs expects an ad_expr node");
  return std::apply(
      [](auto&... args) {
        return std::tuple_cat(make_expr_tuple(args)...);
      },
      e.exprs_);
}

// Flatten the graph in BFS order: level, then its children, etc.
template <typename Tuple>
constexpr auto bfs_flatten(Tuple&& level) {
  if constexpr (std::tuple_size_v<std::decay_t<Tuple>> == 0) {
    return std::tuple<>();
  } else {
    auto next = std::apply(
        [](auto&... nodes) { return std::tuple_cat(child_exprs(nodes.get())...); },
        level);
    return std::tuple_cat(level, bfs_flatten(next));
  }
}

// Entry point: collect nodes (ad_exprs) reachable from z in BFS order.
template <typename Expr>
constexpr auto collect_bfs(Expr& z) {
  if constexpr (is_expr_v<std::decay_t<Expr>>) {
    return bfs_flatten(std::tuple{std::ref(z)});
  } else {
    return std::tuple<>();
  }
}

// Evaluate reverse-pass functors breadthwise using the collected tuple.
template <typename... NodeRefs>
inline void eval_breadthwise(std::tuple<NodeRefs...>& nodes) {
  std::apply(
      [](auto&... node_wrappers) {
        // For each node in BFS order, apply its local reverse functor.
        (std::apply(
             [&](auto&... args) {
               auto& node = node_wrappers.get();
               compute_f(node, args...);  // propagates adjoint to children
             },
             node_wrappers.get().exprs_),
         ...);
      },
      nodes);
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

  // Collect all expression nodes (ad_expr) in breadth-first order.
  auto nodes = collect_bfs(z);

  // Evaluate the reverse pass breadthwise across the whole graph.
  eval_breadthwise(nodes);
}

template <typename Expr>
inline void grad(Expr&& z) {
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
