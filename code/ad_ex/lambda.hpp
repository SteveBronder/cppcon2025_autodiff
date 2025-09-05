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
#include <concepts>
#include <benchmark/benchmark.h>

namespace ad {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<std::decay_t<T>>;

static std::pmr::monotonic_buffer_resource mbr{1<<16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
static alloc_t pa{&mbr};

struct var_base_chain {
  virtual void chain() {};
};
template <typename T>
struct var_base;

template <typename T>
requires Arithmetic<T>
struct var_base<T> : public var_base_chain {
    T value_;
    T adjoint_{0};
    var_base(T x) : var_base_chain(), value_(x), adjoint_(0) {}
    inline auto val() const {
      return value_;
    }
    inline auto& adj() {
      return adjoint_;
    }

};
static std::vector<var_base_chain*> var_vec;
namespace detail {
template <typename T>
struct is_var_base : std::false_type {};
template <typename T>
struct is_var_base<var_base<T>> : std::true_type {};
}
template <typename T>
struct is_var_base : detail::is_var_base<std::decay_t<T>> {};


template <typename T>
inline constexpr bool is_var_base_v = is_var_base<T>::value;
template <typename T, typename... Args>
auto* make_inbuffer(Args&&... args) {
  if constexpr (!is_var_base_v<T>) {
    auto* ret = pa.new_object<T>(std::forward<Args>(args)...);
    var_vec.push_back(ret);
    return ret;
  } else {
    auto* ret = pa.new_object<T>(std::forward<Args>(args)...);
    return ret;
  }
}
template <typename T>
struct var_impl {
  using value_type = std::decay_t<T>;
  var_base<T>* vi_;  // ptr to impl

  var_impl() : vi_(nullptr) {}
  explicit var_impl(var_base<T>* x) : vi_(x) {}

  // For scalar-like inputs: e.g., double -> var_impl<double>.
  template <Arithmetic TT>
  explicit var_impl(TT&& x)
      : vi_(make_inbuffer<var_base<T>>(static_cast<double>(x))) {}

  // For exactly T (e.g., Eigen matrices), not var_impl.
  var_impl(const T& x)
      : vi_(make_inbuffer<var_base<T>>(x)) {}

  var_impl(T&& x)
      : vi_(make_inbuffer<var_base<T>>(std::move(x))) {}

  // Copy ctor stays as a real copy ctor, not hijacked by a template.
  var_impl(const var_impl& x) : vi_(x.vi_) {}

  var_impl& operator+=(var_impl x);
  auto& adj() { return vi_->adjoint_; }
  const auto& val() const { return vi_->value_; }
  void chain() {
    if (vi_) vi_->chain();
  }
};
using var = var_impl<double>;
template <typename T>
inline constexpr bool is_var_v =
    std::is_same_v<std::remove_cvref_t<T>, var>;

// Concept aliases your trait and handles cv/ref.
template <typename T>
concept Var = is_var_v<std::remove_cvref_t<T>>;

template <typename A, typename B>
concept any_var = Var<A> || Var<B>;

template <typename T>
concept var_or_scalar = Var<T> || std::is_arithmetic_v<T>;

template <typename A, typename B>
concept any_var_all_scalar = (var_or_scalar<A> && var_or_scalar<B>);

/**
 * Helper functions
 */
template <typename T>
inline auto& adjoint(var_base<T>& x) { return x.adjoint_; }
template <typename T>
concept HasAdj = requires(T x) { { x.adj() }; };
template <typename T>
concept HasVal = requires(T x) { { x.val() }; };
template <HasAdj T>
inline auto& adjoint(T&& x) {
  return x.adj();
}
template <HasVal T>
inline auto value(T&& x) {
  return x.val();
}
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
template <typename T, typename Lambda>
struct lambda_var_base final : public var_base<T> {
    Lambda lambda_;
    template <typename TT>
    lambda_var_base(TT val, Lambda&& lambda)
        : var_base<T>(val), lambda_(std::move(lambda)) {}
    void chain() {
      lambda_(*this);
    }
};
template <typename T, typename Lambda>
inline auto make_var(T&& ret_val, Lambda&& lambda) {
    return var_impl<T>(make_inbuffer<lambda_var_base<T, Lambda>>(ret_val, std::move(lambda)));
}

template <typename T1, typename T2>
requires any_var_all_scalar<T1, T2>
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
template <>
var& var::operator+=(var x) {
    this->vi_ = ((*this) + x).vi_;
    return *this;
}
template <typename T1, typename T2>
requires any_var_all_scalar<T1, T2>
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

}
