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
#include <benchmark/benchmark.h>

static std::pmr::monotonic_buffer_resource mbr{1<<16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
static alloc_t pa{&mbr};
struct var_impl {
    double value_;
    double adjoint_{0};
    virtual void chain() {};
    var_impl(double x) : value_(x), adjoint_(0) {}
    inline auto val() const {
      return value_;
    }
    inline auto& adj() {
      return adjoint_;
    }

};
static std::vector<var_impl*> var_vec;
template <typename T, typename... Args>
auto* make_inbuffer(Args&&... args) {
  auto* ret = var_vec.emplace_back(new (pa.allocate_bytes(sizeof(T))) T{std::forward<Args>(args)...});
  return ret;
}
struct var {
    var_impl* vi_; // ptr to impl
    var(var_impl* x) : vi_(x) {}
    // Put new var_impl on arena
    var(double x) : vi_(make_inbuffer<var_impl>(x)) {}
    var& operator+=(var x);
    auto& adj() {
      return vi_->adjoint_;
    }
    auto val() const {
      return vi_->value_;
    }
    void chain() {
      if (vi_) {
        vi_->chain();
      }
    }
};

/**
 * Helper functions
 */
inline auto& adjoint(var x) { return x.vi_->adjoint_; }
inline auto& adjoint(var_impl& x) { return x.adjoint_; }

inline auto value(var x) { return x.vi_->value_; }
#ifdef DEBUG_AD
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
template <typename Lambda>
struct lambda_var_impl final : public var_impl {
    Lambda lambda_;
    lambda_var_impl(double val, Lambda&& lambda)
        : var_impl(val), lambda_(std::move(lambda)) {}
    void chain() {
      lambda_(*this);
    }
};
template <typename Lambda>
inline auto make_var(double ret_val, Lambda&& lambda) {
    return var(make_inbuffer<lambda_var_impl<Lambda>>(ret_val, std::move(lambda)));
}
namespace detail {
  template <typename T>
  struct is_var : std::false_type {};
  template <>
  struct is_var<var> : std::true_type {};
}
template <typename T>
struct is_var : detail::is_var<std::decay_t<T>> {};
template <typename T>
inline constexpr bool is_var_v = is_var<T>::value;
template <typename T1, typename T2>
inline auto operator+(T1 lhs, T2 rhs) {
  return make_var(value(lhs) + value(rhs), [lhs, rhs](auto&& ret) mutable {
    if constexpr (is_var_v<T1>) {
      adjoint(lhs) += adjoint(ret);
    }
    if constexpr (is_var_v<T2>) {
      adjoint(rhs) += adjoint(ret);
    }
  });
}
var& var::operator+=(var x) {
    this->vi_ = ((*this) + x).vi_;
    return *this;
}
template <typename T1, typename T2>
inline auto operator*(T1 lhs, T2 rhs) {
  return make_var(value(lhs) + value(rhs), [lhs, rhs](auto&& ret) mutable {
    if constexpr (is_var_v<T1>) {
      adjoint(lhs) += adjoint(ret) * value(rhs);
    }
    if constexpr (is_var_v<T2>) {
      adjoint(rhs) += adjoint(ret) * value(lhs);
    }
  });
}

inline auto log(var x) {
    return make_var(std::log(x.val()), [x](auto&& ret) mutable {
      x.adj() += ret.adj() / x.val();
    });
}

inline void grad(var z) {
    adjoint(z) = 1;
    for (auto&& x : var_vec | std::views::reverse) {
      x->chain();
    }
}

inline void clear_mem() {
    var_vec.clear();
    mbr.release();
}
static void lambda_bench(benchmark::State& state) {
    for (auto _ : state) {
      var x(2.0);
      var y(4.0);
      auto z = x * log(y) + log(x * y) * y;
      grad(z);
      benchmark::DoNotOptimize(z);
      clear_mem();
    }
}


BENCHMARK(lambda_bench);
