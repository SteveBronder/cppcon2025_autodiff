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

#include <type_traits>
#include <utility>
#include <benchmark/benchmark.h>


static std::pmr::monotonic_buffer_resource mbr{1<<16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;

static alloc_t pa{&mbr};
struct var {
    double values_;
    double adjoints_;
    var(double x) : values_(x), adjoints_(0) {}
    auto val() const {
      return values_;
    }
    auto& adj() {
      return adjoints_;
    }
};
// If you already have is_var_v, keep that and skip this block.
template <typename T>
inline constexpr bool is_var_v =
    std::is_same_v<std::remove_cvref_t<T>, var>;

// Concept aliases your trait and handles cv/ref.
template <typename T>
concept Var = is_var_v<std::remove_cvref_t<T>>;

template <typename A, typename B>
concept any_var = Var<A> || Var<B>;

template <typename T>
struct deduce_ownership {
  static constexpr bool value = std::is_rvalue_reference_v<T>;
  using type = std::conditional_t<value,
    std::remove_reference_t<T>, std::reference_wrapper<std::decay_t<T>>>;
};
template <typename T>
using deduce_ownership_t = typename deduce_ownership<T&&>::type;
template <typename F, typename... Exprs>
struct ad_expr {
   var ret_;
   std::tuple<deduce_ownership_t<Exprs>...> exprs_;
   std::decay_t<F> f_;
   template <typename FF, typename... Args>
   ad_expr(double x, FF&& f, Args&&... args) :
     ret_(x), f_(std::forward<F>(f)),
     exprs_(std::forward<Args>(args)...) {}
   auto val() { return ret_.val();}
   auto& adj() { return ret_.adj();}
};
template <typename T, typename F, typename... Args>
inline auto make_expr(T&& x, F&& f, Args&&... args) {
  return ad_expr<F, Args&&...>{std::forward<T>(x),
                                           std::forward<F>(f),
                                           std::forward<Args>(args)...};
}
namespace detail {
  template <typename T>
  struct is_expr : std::false_type {};
  template <typename F, typename... Exprs>
  struct is_expr<ad_expr<F, Exprs...>> : std::true_type {};
  template <typename T>
  struct is_ref_wrap : std::false_type {};
  template <typename T>
  struct is_ref_wrap<std::reference_wrapper<T>> : std::true_type {};
  template <typename T>
  struct is_ref_wrap_expr : std::false_type {};
  template <typename F, typename... Exprs>
  struct is_ref_wrap_expr<std::reference_wrapper<ad_expr<F, Exprs...>>> : std::true_type {};
}
template <typename T>
struct is_expr : detail::is_expr<std::decay_t<T>> {};
template <typename T>
constexpr bool is_expr_v = is_expr<T>::value;

template <typename T>
struct is_ref_wrap : detail::is_ref_wrap<std::decay_t<T>> {};
template <typename T>
constexpr bool is_ref_wrap_v = is_ref_wrap<T>::value;
template <typename T>
concept RefWrap = is_ref_wrap_v<T>;

template <typename T>
struct is_ref_wrap_expr : detail::is_ref_wrap_expr<std::decay_t<T>> {};
template <typename T>
constexpr bool is_ref_wrap_expr_v = is_ref_wrap_expr<T>::value;

template <typename T>
concept Expr = is_expr_v<T>;

/**
 * Helper functions
 */
template <Var T>
auto& adjoint(T&& x) { return x.adj(); }
template <Expr T>
auto& adjoint(T&& x) { return x.adj(); }
template <RefWrap T>
auto& adjoint(T&& x) { return adjoint(x.get()); }

template <Var T>
auto value(T&& x) { return x.val(); }
template <Expr T>
auto value(T&& x) { return x.val(); }
template <RefWrap T>
auto value(T&& x) { return value(x.get()); }

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

template <typename... Types>
concept any_var_or_expr = ((Var<Types> || Expr<Types>) || ... || (false));

template <typename T1, typename T2>
requires any_var_or_expr<T1, T2>
inline auto operator+(T1&& lhs, T2&& rhs) {
  return make_expr(value(lhs) + value(rhs), [](auto&& ret, auto&& lhs, auto&& rhs) {
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret);
    }
  }, std::forward<T1>(lhs), std::forward<T2>(rhs));
}

template <typename T1, typename T2>
requires any_var_or_expr<T1, T2>
inline auto operator*(T1&& lhs, T2&& rhs) {
  return make_expr(value(lhs) * value(rhs), [](auto&& ret, auto&& lhs, auto&& rhs) {
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret) * value(rhs);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret) * value(lhs);
    }
  }, std::forward<T1>(lhs), std::forward<T2>(rhs));
}

template <typename Expr>
inline auto log(Expr&& x) {
  return make_expr(std::log(x.val()), [](auto&& ret, auto&& x) {
    adjoint(x) += ret.adj() / value(x);
  }, std::forward<Expr>(x));
}

// Wrap an object in a tuple if it's an ad_expr, otherwise an empty tuple.
template <typename T>
constexpr auto make_expr_tuple(T&& x) {
  if constexpr (is_ref_wrap_expr_v<std::decay_t<T>>) {
    return std::tuple{x};
  } else if constexpr (is_expr_v<std::decay_t<T>>) {
    return std::tuple<std::reference_wrapper<std::decay_t<T>>>{std::ref(x)};
  } else {
    return std::tuple<>();
  }
}

// Given an ad_expr node, return a tuple of its children that are ad_exprs.
template <typename E>
constexpr auto child_exprs(E&& e) {
  static_assert(is_expr_v<std::decay_t<E>> || is_ref_wrap_v<std::decay_t<E>>,
                "child_exprs expects an ad_expr node");
  return std::apply(
      [](auto&... args) {
        return std::tuple_cat(make_expr_tuple(args)...);
      },
      std::forward<E>(e).exprs_);
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
    return std::tuple_cat(std::forward<Tuple>(level), bfs_flatten(std::move(next)));
  }
}

// Entry point: collect nodes (ad_exprs) reachable from z in BFS order.
template <typename Expr>
constexpr auto collect_bfs(Expr&& z) {
  if constexpr (is_expr_v<std::decay_t<Expr>>) {
    return bfs_flatten(std::tuple{std::ref(std::forward<Expr>(z))});
  } else {
    return std::tuple<>();
  }
}

// Evaluate reverse-pass functors breadthwise using the collected tuple.
template <typename Tuple>
inline void eval_breadthwise(Tuple&& nodes) {
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
      std::forward<Tuple>(nodes));
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
inline void grad(Expr&& z) {
  adjoint(z) = 1.0;

  // Collect all expression nodes (ad_expr) in breadth-first order.
  auto nodes = collect_bfs(z);

  // Evaluate the reverse pass breadthwise across the whole graph.
  eval_breadthwise(nodes);
}

static void sct_bench(benchmark::State& state) {
    for (auto _ : state) {
      var x(2.0);
      var y(4.0);
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      auto z = x * log(y) + log(x * y) * y;
      grad(z);
      benchmark::DoNotOptimize(z);
      benchmark::DoNotOptimize(x.adj());
      benchmark::DoNotOptimize(y.adj());
    }
}


BENCHMARK(sct_bench);
