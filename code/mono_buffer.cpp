/**
 * \file
 * \brief Benchmark implementing a hand-written expression graph using a monotonic buffer.
 *
 * This variant constructs explicit node types for each operation and stores them
 * in a monotonic buffer resource to avoid per-node heap allocations. The graph
 * is traversed manually during the reverse sweep, providing a canonical example
 * of a dynamic tape with custom memory management.
 */
#include <stdint.h>
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

inline std::pmr::monotonic_buffer_resource mbr{1 << 16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
inline alloc_t pa{&mbr};

/**
 * \brief Base node storing scalar value and adjoint information.
 *
 * Concrete operation nodes derive from this type and override ::chain to
 * propagate adjoints to their operands.
 */
struct var_impl {
  double value_;
  double adjoint_{0};
  virtual void chain() {};
  var_impl(double x) : value_(x), adjoint_(0) {}
};

inline std::vector<var_impl *> var_vec;

/**
 * \brief Allocates an object inside the monotonic buffer.
 *
 * Non-leaf nodes are pushed onto the tape so that the reverse sweep visits them
 * in LIFO order.
 *
 * @tparam T Node type to allocate.
 * @tparam Args Constructor argument types.
 * @param args Arguments forwarded to the node constructor.
 * @return Pointer to the allocated node.
 */
template <typename T, typename... Args> auto *make_inbuffer(Args &&...args) {
  if constexpr (!std::is_same_v<T, var_impl>) {
    auto *ret = var_vec.emplace_back(new (pa.allocate_bytes(sizeof(T)))
                                         T{std::forward<Args>(args)...});
    return ret;
  } else {
    auto *ret =
        new (pa.allocate_bytes(sizeof(T))) T{std::forward<Args>(args)...};
    return ret;
  }
}

/**
 * \brief Public facing variable wrapper owning a pointer to a node implementation.
 */
struct var {
  var_impl *vi_; // ptr to impl
  var(var_impl *x) : vi_(x) {}
  // Put new var_impl on arena
  var(double x) : vi_(make_inbuffer<var_impl>(x)) {}
  var &operator+=(var x);
  auto &adj() { return vi_->adjoint_; }
  auto val() { return vi_->value_; }
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

inline auto value(var x) { return x.vi_->value_; }

#ifdef DEBUG_AD
void print_var(const char *name, var &ret, var x) {
  std::cout << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
            << std::endl;
  std::cout << name << " Op: (" << value(x) << ", " << adjoint(x) << ")"
            << std::endl;
}

void print_var(const char *name, var &ret, var x, var y) {
  std::cout << "\t\t" << name << ": (" << value(ret) << ", " << adjoint(ret)
            << ")" << std::endl;
  std::cout << "\t\t" << name << " OpL: (" << value(x) << ", " << adjoint(x)
            << ")" << std::endl;
  std::cout << "\t\t" << name << " OpR: (" << value(y) << ", " << adjoint(y)
            << ")" << std::endl;
}
#else
constexpr void print_var(const char *name, var &ret, var x) {}

constexpr void print_var(const char *name, var &ret, var x, var y) {}
#endif

/**
 * \brief Node representing the sum of two variables.
 */
struct add_vv final : public var_impl {
  var lhs_;
  var rhs_;
  add_vv(double val, var lhs, var rhs) : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += this->adjoint_;
    rhs_.adj() += this->adjoint_;
  }
};
struct add_dv final : public var_impl {
  double lhs_;
  var rhs_;
  add_dv(double val, double lhs, var rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() { rhs_.adj() += this->adjoint_; }
};
struct add_vd final : public var_impl {
  var lhs_;
  double rhs_;
  add_vd(double val, var lhs, double rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() { lhs_.adj() += this->adjoint_; }
};

/**
 * \brief Overload implementing addition between variables and/or scalars.
 *
 * @tparam T1 Left operand type.
 * @tparam T2 Right operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Variable representing the sum.
 */
template <typename T1, typename T2> auto operator+(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<add_vv>(lhs.val() + rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{make_inbuffer<add_vd>(lhs.val() + rhs), lhs, rhs};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<add_dv>(lhs + rhs.val()), lhs, rhs};
  }
}
var &var::operator+=(var x) {
  this->vi_ = ((*this) + x).vi_;
  return *this;
}

/**
 * \brief Node representing the product of two variables.
 */
struct mul_vv final : public var_impl {
  var lhs_;
  var rhs_;
  mul_vv(double val, var lhs, var rhs) : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += rhs_.val() * this->adjoint_;
    rhs_.adj() += lhs_.val() * this->adjoint_;
  }
};
struct mul_dv final : public var_impl {
  double lhs_;
  var rhs_;
  mul_dv(double val, double lhs, var rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}

  void chain() { rhs_.adj() += lhs_ * this->adjoint_; }
};

struct mul_vd final : public var_impl {
  var lhs_;
  double rhs_;
  mul_vd(double val, var lhs, double rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() { lhs_.adj() += rhs_ * this->adjoint_; }
};

/**
 * \brief Overload implementing multiplication between variables and/or scalars.
 *
 * @tparam T1 Left operand type.
 * @tparam T2 Right operand type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Variable representing the product.
 */
template <typename T1, typename T2> auto operator*(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<mul_vv>(lhs.val() * rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{make_inbuffer<mul_vd>(lhs.val() * rhs, lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<mul_dv>(lhs * rhs.val(), lhs, rhs)};
  }
}

/**
 * \brief Node representing the natural logarithm of a variable.
 */
struct log_var final : public var_impl {
  var in_;
  log_var(double x, var in) : var_impl(x), in_(in) {}
  void chain() { in_.adj() += this->adjoint_ / in_.val(); }
};

/**
 * \brief Computes the natural logarithm of a variable.
 *
 * @param x Input variable.
 * @return Variable representing \f$\log(x)\f$.
 */
auto log(var x) { return var{make_inbuffer<log_var>(std::log(x.val()), x)}; }

/**
 * \brief Runs the reverse sweep starting from a scalar output.
 *
 * @param z Output variable of the expression graph
 */
void grad(var z) noexcept {
  adjoint(z) = 1;
  for (auto &&x : var_vec | std::views::reverse) {
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

/**
 * \brief Benchmark exercising the monotonic-buffer-backed tape implementation.
 *
 * The benchmark builds a computation graph, executes the reverse sweep, and
 * clears the tape on each iteration.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void monobuff_bench(benchmark::State &state) {
  for (auto _ : state) {
    var x(2.0);
    var y(4.0);
    auto z = x * log(y) + log(x * y) * y;
    grad(z);
    benchmark::DoNotOptimize(z);
    clear_mem();
  }
}
BENCHMARK(monobuff_bench);
