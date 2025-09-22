/**
 * \file
 * \brief Benchmark using expression templates with explicit storage binding.
 *
 * The implementation encodes the computation graph in the type system and binds
 * contiguous value and adjoint buffers before evaluation. This resembles a
 * static reverse-mode system with manual memory management for composite
 * expressions.
 */
#include <Eigen/Dense>
#include <ad_ex/meta/is_eigen.hpp>
#include <benchmark/benchmark.h>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>
#define STRONG_INLINE __attribute__((always_inline, hot)) inline

namespace ad {
/**
 * \brief Helper selecting ownership semantics for expression nodes.
 *
 * Lvalues are wrapped in `std::reference_wrapper` while rvalues are stored by
 * value, mirroring the logic used in the static expression system.
 *
 * @tparam T Expression type.
 */
template <typename T> struct deduce_ownership {
  static constexpr bool value = std::is_rvalue_reference_v<T>;
  using type = std::conditional_t<value, std::remove_reference_t<T>,
                                  std::reference_wrapper<std::decay_t<T>>>;
};
template <typename T>
using deduce_ownership_t = typename deduce_ownership<T &&>::type;

namespace detail {
template <typename T> struct is_ref_wrap : std::false_type {};
template <typename T>
struct is_ref_wrap<std::reference_wrapper<T>> : std::true_type {};
} // namespace detail
template <typename T>
struct is_ref_wrap : detail::is_ref_wrap<std::decay_t<T>> {};
template <typename T> constexpr bool is_ref_wrap_v = is_ref_wrap<T>::value;
template <typename T>
concept RefWrap = is_ref_wrap_v<T>;

/**
 * \brief Returns the underlying object from a reference wrapper.
 *
 * @tparam T Reference wrapper type.
 * @param x Reference wrapper instance.
 * @return Underlying object.
 */
template <RefWrap T> constexpr STRONG_INLINE auto &&get(T &&x) {
  return x.get();
}

/**
 * \brief Returns the provided value unchanged.
 *
 * This overload handles non-wrapper arguments transparently.
 *
 * @tparam T Value type.
 * @param x Value to forward.
 * @return Forwarded value.
 */
template <typename T> constexpr STRONG_INLINE auto &&get(T &&x) { return x; }

template <typename T> struct Var;
template <typename T> struct VarView;

/**
 * \brief View providing access to value and adjoint matrices.
 *
 * @tparam T Eigen matrix type.
 */
template <typename T>
  requires EigenMatrix<T>
struct VarView<T> {
  using mat_t = std::decay_t<T>;
  VarView(T &val, T &adj)
      : vals_(val.data(), val.rows(), val.cols()),
        adjs_(adj.data(), adj.rows(), adj.cols()) {}
  Eigen::Map<T> vals_;
  Eigen::Map<T> adjs_;
};

/**
 * \brief Leaf node storing references to contiguous buffers for value and
 * adjoint.
 *
 * @tparam T Eigen matrix type.
 */
template <typename T>
  requires EigenMatrix<T>
struct Var<T> {
  static constexpr std::size_t ops = 1;
  using mat_type = std::decay_t<T>;
  template <typename T1>
    requires EigenMatrix<T1>
  Var(T1 &&init_value)
      : rows_(init_value.rows()), cols_(init_value.cols()),
        value_ptr_(init_value.data()), adj_ptr_(nullptr) {}
  /**
   * \brief Constructs a variable with both value and adjoint buffers.
   *
   * @tparam T1 Value matrix type.
   * @tparam T2 Adjoint matrix type.
   * @param init_value Initial value matrix.
   * @param init_adjoint Initial adjoint matrix.
   */
  template <typename T1, typename T2>
  Var(T1 &&init_value, T2 &&init_adjoint)
      : rows_(init_value.rows()), cols_(init_value.cols()),
        value_ptr_(init_value.data()), adj_ptr_(init_adjoint.data()) {}
  /**
   * \brief Constructs an unbound variable with the specified dimensions.
   *
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  Var(Eigen::Index rows, Eigen::Index cols)
      : rows_(rows), cols_(cols), value_ptr_(nullptr), adj_ptr_(nullptr) {}

  // Memory sizing (in number of doubles).
  constexpr STRONG_INLINE std::pair<std::size_t, std::size_t>
  cache_bind_size() const {
    return {0, rows_ * cols_};
  }

  // bind this node to segments within the provided contiguous buffers.
  constexpr STRONG_INLINE void bind(double *__restrict values_base,
                                    double *__restrict adjs_base,
                                    std::size_t &v_off, std::size_t &a_off) {
    adj_ptr_ = adjs_base + a_off;
    a_off += static_cast<std::size_t>(rows_) * cols_;
  }

  /**
   * \brief Evaluates the forward pass by returning a mapped value view.
   *
   * @return Eigen map referencing the value buffer.
   */
  STRONG_INLINE constexpr auto f_eval() { return this->value_map(); }

  // b_eval: leaf (no children).
  template <typename TT> STRONG_INLINE constexpr void b_eval(TT &&seed) {
    this->adjoint_map().noalias() += std::forward<TT>(seed);
  }

  // Access mapped views (created on demand).
  STRONG_INLINE auto value_map() {
    return Eigen::Map<const Eigen::MatrixXd>(value_ptr_, rows_, cols_);
  }
  STRONG_INLINE auto adjoint_map() {
    return Eigen::Map<Eigen::MatrixXd>(adj_ptr_, rows_, cols_);
  }

  STRONG_INLINE Eigen::Index rows() const { return rows_; }
  STRONG_INLINE Eigen::Index cols() const { return cols_; }

  Eigen::Index rows_;
  Eigen::Index cols_;
  double *__restrict value_ptr_; // bound at bind()
  double *__restrict adj_ptr_;   // bound at bind()
};

/**
 * \brief Returns a slice of a parameter pack as a tuple.
 *
 * @tparam start Starting index.
 * @tparam slice_size Number of elements to select.
 * @tparam Args Parameter pack types.
 * @param args Parameter pack values.
 * @return Tuple containing the requested slice.
 */
template <std::size_t start, std::size_t slice_size, typename... Args>
[[nodiscard]] STRONG_INLINE constexpr auto slice_param_pack(Args &&...args) {
  // Materialize decayed copies/moves to avoid dangling references.
  auto values = std::forward_as_tuple(std::forward<Args>(args)...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::forward_as_tuple(std::get<start + I>(values)...);
  }(std::make_index_sequence<slice_size>{});
}

/**
 * \brief Binary expression node implementing matrix multiplication.
 *
 * @tparam Left Left operand type.
 * @tparam Right Right operand type.
 */
template <typename Left, typename Right> struct MatMul {
  using Left_ = std::decay_t<Left>;
  using Right_ = std::decay_t<Right>;
  static constexpr std::size_t ops = 2;
  /**
   * \brief Constructs a matrix multiplication node from its operands.
   *
   * @tparam L Left operand type.
   * @tparam R Right operand type.
   * @param left Left operand.
   * @param right Right operand.
   */
  template <typename L, typename R>
  MatMul(L &&left, R &&right)
      : left_(std::forward<L>(left)), right_(std::forward<R>(right)),
        rows_(left_.rows()), cols_(right_.cols()), value_ptr_(nullptr),
        adj_ptr_(nullptr) {}

  STRONG_INLINE std::pair<std::size_t, std::size_t> cache_bind_size() const {
    // Own storage + children.
    const std::size_t n = static_cast<std::size_t>(rows_) * cols_;
    auto [lv, la] = left_.cache_bind_size();
    auto [rv, ra] = right_.cache_bind_size();
    return {n + lv + rv, n + la + ra};
  }

  STRONG_INLINE void bind(double *__restrict values_base,
                          double *__restrict adjs_base, std::size_t &v_off,
                          std::size_t &a_off) {
    // bind children first (any order is fine) then self.
    left_.bind(values_base, adjs_base, v_off, a_off);
    right_.bind(values_base, adjs_base, v_off, a_off);
    value_ptr_ = values_base + v_off;
    adj_ptr_ = adjs_base + a_off;
    v_off += static_cast<std::size_t>(rows_) * cols_;
    a_off += static_cast<std::size_t>(rows_) * cols_;
  }
  STRONG_INLINE auto f_eval() {
    // value = left.value * right.value
    return Eigen::Map<Eigen::MatrixXd>(value_ptr_, rows_, cols_).noalias() =
               left_.f_eval() * right_.f_eval();
  }

  template <typename TT> STRONG_INLINE void b_eval(TT &&seed) {
    // Propagate: dL += dC * R^T ; dR += L^T * dC
    this->adjoint_map().array() += seed;
    auto l_adj = this->adjoint_map() * right_.value_map().transpose();
    left_.b_eval(std::move(l_adj));
    auto r_adj = left_.value_map().transpose() * this->adjoint_map();
    right_.b_eval(std::move(r_adj));
  }

  STRONG_INLINE auto value_map() {
    return Eigen::Map<Eigen::MatrixXd>(value_ptr_, rows_, cols_);
  }
  STRONG_INLINE auto adjoint_map() {
    return Eigen::Map<Eigen::MatrixXd>(adj_ptr_, rows_, cols_);
  }

  STRONG_INLINE Eigen::Index rows() const { return rows_; }
  STRONG_INLINE Eigen::Index cols() const { return cols_; }

  std::decay_t<Left> left_;
  std::decay_t<Right> right_;
  Eigen::Index rows_;
  Eigen::Index cols_;
  double *__restrict value_ptr_;
  double *__restrict adj_ptr_;
};

/**
 * \brief Unary expression node that sums all elements of its child.
 *
 * @tparam Child Child expression type.
 */
template <typename Child> struct Sum {
  static constexpr std::size_t ops = 1;
  /**
   * \brief Constructs a summation node from its child.
   *
   * @tparam T Child expression type.
   * @param child Child expression.
   */
  template <typename T>
  explicit Sum(T &&child) : child_(std::forward<T>(child)), val_(0), adj_(0) {}

  // Total storage: 1 for value + child; 1 for adjoint + child.
  STRONG_INLINE std::pair<std::size_t, std::size_t> cache_bind_size() const {
    return child_.cache_bind_size();
  }

  STRONG_INLINE void bind(double *__restrict values_base,
                          double *__restrict adjs_base, std::size_t &v_off,
                          std::size_t &a_off) {
    child_.bind(values_base, adjs_base, v_off, a_off);
  }
  STRONG_INLINE auto f_eval() { return val_ = child_.f_eval().sum(); }
  template <typename TT> STRONG_INLINE void b_eval(TT &&seed) {
    adj_ += seed;
    child_.b_eval(seed);
  }

  // We really need a reset function, but for this toy example this is fine
  STRONG_INLINE void reset_adjoint() { adj_ = 0.0; }

  std::decay_t<Child> child_;
  double val_;
  double adj_;
};

/**
 * \brief Utility wrapper delegating to an expression's CacheBindSize method.
 *
 * @tparam Expr Expression type.
 * @param expr Expression whose storage size is requested.
 * @return Pair of counts for values and adjoints respectively.
 */
template <typename Expr>
STRONG_INLINE std::pair<std::size_t, std::size_t> cache_bind_size(Expr &&expr) {
  return expr.cache_bind_size();
}

/**
 * \brief Binds an expression tree to contiguous value and adjoint buffers.
 *
 * @tparam Expr Expression type.
 * @param expr Expression to bind.
 * @param values_base Pointer to the value buffer.
 * @param adjs_base Pointer to the adjoint buffer.
 */
template <typename Expr>
STRONG_INLINE void bind(Expr &&expr, double *__restrict values_base,
                        double *__restrict adjs_base) {
  std::size_t v_off = 0, a_off = 0;
  expr.bind(values_base, adjs_base, v_off, a_off);
}

/**
 * \brief Runs the forward and reverse pass for an expression tree.
 *
 * @tparam Expr Expression type.
 * @param expr Expression to evaluate.
 */
template <typename Expr> STRONG_INLINE void AutoDiff(Expr &&expr) {
  expr.f_eval();
  expr.reset_adjoint();
  expr.b_eval(1.0);
}

/**
 * \brief Convenience overload constructing a MatMul node.
 *
 * @tparam Op1 Left operand type.
 * @tparam Op2 Right operand type.
 * @param left Left operand.
 * @param right Right operand.
 * @return Matrix multiplication node.
 */
template <typename Op1, typename Op2>
STRONG_INLINE auto operator*(Op1 &&left, Op2 &&right) {
  return MatMul<Op1, Op2>(std::forward<Op1>(left), std::forward<Op2>(right));
}

/**
 * \brief Convenience overload constructing a Sum node.
 *
 * @tparam Op Child expression type.
 * @param child Child expression.
 * @return Summation node.
 */
template <typename Op> STRONG_INLINE auto sum(Op &&child) {
  return Sum<Op>(std::forward<Op>(child));
}

} // namespace ad

/**
 * \brief Benchmark exercising the expression-template automatic differentiation
 * system.
 *
 * Each iteration binds contiguous storage for the expression graph, performs
 * the forward and reverse passes, and resets the adjoint buffer.
 *
 * @param state Google Benchmark harness state controlling repetitions.
 */
static void expr_template(benchmark::State &state) {
  // Dynamic inputs (2x2 here; any MxK and KxN will work).
  Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::MatrixXd B0 = Eigen::MatrixXd::Random(state.range(0), state.range(0));

  // 1) Build expression graph: f = sum(A * B). No storage yet.
  ad::Var<Eigen::MatrixXd> A(A0);
  ad::Var<Eigen::MatrixXd> B(B0);
  auto f = ad::sum(A * B);

  // 2) Ask graph how much contiguous storage it needs.
  auto [vsize, asize] = ad::cache_bind_size(f);

  // 3) Provide contiguous buffers and bind them to the graph.
  Eigen::VectorXd values(vsize);
  Eigen::VectorXd adjs(asize);
  values.setZero();
  adjs.setZero();
  ad::bind(f, values.data(), adjs.data());
  for (auto _ : state) {
    // 4) Autodiff (forward + reverse).
    ad::AutoDiff(f);
    adjs.setZero();
  }
}
BENCHMARK(expr_template)->RangeMultiplier(2)->Range(1, 4096);
