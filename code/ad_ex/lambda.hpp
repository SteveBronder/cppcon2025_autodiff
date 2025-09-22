/**
 * \file
 * \brief Lambda-based dynamic reverse-mode automatic differentiation system.
 *
 * Each arithmetic operation constructs a lightweight node that stores a lambda
 * capturing the reverse propagation logic. Nodes are allocated from a monotonic
 * buffer to minimize allocation overhead while still forming a dynamic tape.
 */
#pragma once
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
#include <ranges> 
#include <concepts>
#include <benchmark/benchmark.h>

namespace ad {
template <typename T>
// Concept identifying arithmetic types supported by the scalar interface.
concept Arithmetic = std::is_arithmetic_v<std::decay_t<T>>;
/// Static buffer used to back the monotonic buffer resource.
inline std::array<std::byte, 8192> buffer{};

// Monotonic buffer resource providing fast allocations for tape nodes.
inline std::pmr::monotonic_buffer_resource mbr{buffer.data(), buffer.size()};

using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
// Polymorphic allocator bound to the monotonic buffer.
inline alloc_t pa{&mbr};

struct var_base_chain {
  // Virtual hook that allows derived nodes to implement reverse propagation.
  virtual void chain() {};
};
// Forward declaration of the templated base implementation.
template <typename T>
struct var_base;

/**
 * \brief Primary storage for scalar values and adjoints.
 *
 * @tparam T Arithmetic type for values and adjoints.
 */
template <typename T>
requires Arithmetic<T>
struct var_base<T> : public var_base_chain {
    T value_;
    T adjoint_{0};
    /**
     * \brief Constructs a node storing the supplied value.
     *
     * @param x Initial value.
     */
    var_base(T x) : var_base_chain(), value_(x), adjoint_(0) {}
    inline auto val() const {
      return value_;
    }
    inline auto& adj() {
      return adjoint_;
    }

};
// Vector storing pointers to nodes in allocation order.
inline std::vector<var_base_chain*> var_vec;

namespace detail {
// Detecting whether a type is a ::var_base specialization.
template <typename T>
struct is_var_base : std::false_type {};
template <typename T>
struct is_var_base<var_base<T>> : std::true_type {};
}
template <typename T>
struct is_var_base : detail::is_var_base<std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_var_base_v = is_var_base<T>::value;

/**
 * \brief Allocates an object inside the monotonic buffer.
 *
 * Objects derived from ::var_base_chain are recorded on the tape for the reverse pass.
 *
 * @tparam T Object type.
 * @tparam Args Constructor argument types.
 * @param args Constructor arguments.
 * @return Pointer to the allocated object.
 */
template <typename T, typename... Args>
inline auto* make_inbuffer(Args&&... args) {
  if constexpr (!is_var_base_v<T>) {
    auto* ret = pa.new_object<T>(std::forward<Args>(args)...);
    var_vec.push_back(ret);
    return ret;
  } else {
    auto* ret = pa.new_object<T>(std::forward<Args>(args)...);
    return ret;
  }
}
/**
 * \brief PIMPL that points to a ::var_base instance.
 *
 * @tparam T Stored value type.
 */
template <typename T>
struct var_impl {
  using value_type = std::decay_t<T>;
  var_base<T>* vi_;  // ptr to impl

  var_impl() : vi_(nullptr) {}
  explicit var_impl(var_base<T>* x) : vi_(x) {}

  /**
   * \brief Constructs an input node
   *
   * @tparam TT Arithmetic type.
   * @param x Scalar value.
   */
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

  /**
   * \brief Adds another variable to the current one in-place.
   *
   * @param x Variable to add.
   * @return Reference to `*this` after the addition.
   */
  var_impl& operator+=(var_impl x);
  auto& adj() { return vi_->adjoint_; }
  const auto& val() const { return vi_->value_; }
  // Invokes the reverse propagation for the underlying node.
  void chain() {
    if (vi_) {
      vi_->chain();
    } 
  }
};

using var = var_impl<double>;

template <typename T>
inline constexpr bool is_var_v =
    std::is_same_v<std::remove_cvref_t<T>, var>;

// Concept for a `var` type
template <typename T>
concept Var = is_var_v<std::remove_cvref_t<T>>;

template <typename A, typename B>
concept any_var = Var<A> || Var<B>;

// Concept satisfied by either ::var or arithmetic types.
template <typename T>
concept var_or_scalar = Var<T> || std::is_arithmetic_v<T>;

// Concept ensuring both operands are ::var or arithmetic.
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

/**
 * \brief Node storing a lambda that executes reverse propagation.
 *
 * @tparam T Value type.
 * @tparam Lambda Lambda type capturing the reverse rule.
 */
template <typename T, typename Lambda>
struct lambda_var_base final : public var_base<T> {
    Lambda lambda_;
    /**
     * \brief Constructs a lambda-backed node.
     *
     * @tparam TT Value type.
     * @param val Primal value.
     * @param lambda Reverse propagation lambda.
     */
    template <typename TT>
    lambda_var_base(TT&& val, Lambda&& lambda)
        : var_base<T>(std::forward<TT>(val)), lambda_(std::move(lambda)) {}
    void chain() {
      lambda_(*this);
    }
};

/**
 * \brief Creates a lambda-backed variable.
 *
 * @tparam T Value type.
 * @tparam Lambda Lambda type.
 * @param ret_val Primal value.
 * @param lambda Reverse propagation lambda.
 * @return Variable implementation pointing to the allocated node.
 */
template <typename T, typename Lambda>
inline auto make_var(T&& ret_val, Lambda&& lambda) {
    return var_impl<T>(make_inbuffer<lambda_var_base<T, Lambda>>(ret_val, std::move(lambda)));
}

/**
 * \brief Addition between variables and/or scalars.
 *
 * @tparam T1 Left operand type.
 * @tparam T2 Right operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Variable representing the sum.
 */
template <typename T1, typename T2>
requires any_var_all_scalar<T1, T2>
inline auto operator+(T1 lhs, T2 rhs) {
  return make_var(value(lhs) + value(rhs), 
  [lhs, rhs](auto&& ret) mutable {
    if constexpr (is_var_v<T1>) {
      adjoint(lhs) += adjoint(ret);
    }
    if constexpr (is_var_v<T2>) {
      adjoint(rhs) += adjoint(ret);
    }
  });
}
// In-place addition assignment for scalar variables.
template <>
inline var& var::operator+=(var x) {
    this->vi_ = ((*this) + x).vi_;
    return *this;
}

/**
 * \brief Multiplication between variables and/or scalars.
 *
 * @tparam T1 Left operand type.
 * @tparam T2 Right operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Variable representing the product.
 */
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

/**
 * \brief Natural logarithm for scalar variables.
 *
 * @param x Input variable.
 * @return Variable representing \f$\log(x)\f$.
 */
inline auto log(var x) {
    return make_var(std::log(x.val()), [x](auto&& ret) mutable {
      x.adj() += ret.adj() / x.val();
    });
}

/**
 * \brief Runs the reverse sweep starting from a scalar output.
 *
 * @param z Output variable of the expression node.
 */
inline void grad(var z) {
    adjoint(z) = 1;
    for (auto&& x : var_vec | std::views::reverse) {
      x->chain();
    }
}
/**
 * \brief Clears the tape and releases the monotonic buffer.
 */
inline void clear_mem() {
    var_vec.clear();
    mbr.release();
}

}
