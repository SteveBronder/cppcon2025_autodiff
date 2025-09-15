#include <ad_ex/meta/is_eigen.hpp>
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>
#define STRONG_INLINE __attribute__((always_inline, hot)) inline

namespace ad{
  template <typename T>
struct deduce_ownership {
  static constexpr bool value = std::is_rvalue_reference_v<T>;
  using type = std::conditional_t<value,
    std::remove_reference_t<T>, std::reference_wrapper<std::decay_t<T>>>;
};
template <typename T>
using deduce_ownership_t = typename deduce_ownership<T&&>::type;

namespace detail {
  template <typename T>
  struct is_ref_wrap : std::false_type {};
  template <typename T>
  struct is_ref_wrap<std::reference_wrapper<T>> : std::true_type {};
}
template <typename T>
struct is_ref_wrap : detail::is_ref_wrap<std::decay_t<T>> {};
template <typename T>
constexpr bool is_ref_wrap_v = is_ref_wrap<T>::value;
template <typename T>
concept RefWrap = is_ref_wrap_v<T>;

template <RefWrap T>
constexpr STRONG_INLINE auto&& get(T&& x) {
  return x.get();
}
template <typename T>
constexpr STRONG_INLINE auto&& get(T&& x) {
  return x;
}

template <typename T>
struct Var;
template <typename T>
struct VarView;

template <typename T>
requires EigenMatrix<T>
struct VarView<T> {
  using mat_t = std::decay_t<T>;
  VarView(T& val, T& adj) : 
  vals_(val.data(), val.rows(), val.cols()), adjs_(adj.data(), adj.rows(), adj.cols()) {}
  Eigen::Map<T> vals_;
  Eigen::Map<T> adjs_;
};
// ------------------------------ Var ---------------------------------
// Leaf node holding dynamic (rows x cols) matrices for value/adjoint.
// Does not allocate at construction; it binds to external contiguous
// buffers later (lazy allocation). It keeps an initial value to copy
// into the bound value buffer during f_eval().
template <typename T>
requires EigenMatrix<T>
struct Var<T> {
 static constexpr std::size_t ops = 1;
 using mat_type = std::decay_t<T>;
  template <typename T1>
  requires EigenMatrix<T1>
  Var(T1&& init_value)
      : rows_(init_value.rows()), cols_(init_value.cols()), 
        value_ptr_(init_value.data()), adj_ptr_(nullptr) {
  }
  template <typename T1, typename T2>
  Var(T1&& init_value, T2&& init_adjoint)
      : rows_(init_value.rows()), cols_(init_value.cols()), 
        value_ptr_(init_value.data()), adj_ptr_(init_adjoint.data()) {
  }
  Var(Eigen::Index rows, Eigen::Index cols)
      : rows_(rows), cols_(cols),
        value_ptr_(nullptr), adj_ptr_(nullptr) {
  }

  // Memory sizing (in number of doubles).
  constexpr STRONG_INLINE std::pair<std::size_t, std::size_t> CacheBindSize() const {
    return {0, rows_ * cols_};
  }

  // Bind this node to segments within the provided contiguous buffers.
  constexpr STRONG_INLINE void Bind(double __restrict* values_base, double __restrict* adjs_base,
            std::size_t& v_off, std::size_t& a_off) {
    adj_ptr_ = adjs_base + a_off;
    a_off += static_cast<std::size_t>(rows_) * cols_;
  }

  STRONG_INLINE constexpr auto f_eval() {
    return this->value_map();
  }

  // b_eval: leaf (no children).
  template <typename TT>
  STRONG_INLINE constexpr void b_eval(TT&& seed) {
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
  double __restrict* value_ptr_;     // bound at Bind()
  double __restrict* adj_ptr_;       // bound at Bind()
};

template <std::size_t start, std::size_t slice_size, typename... Args>
[[nodiscard]] STRONG_INLINE constexpr auto slice_param_pack(Args&&... args) {
  // Materialize decayed copies/moves to avoid dangling references.
  auto values = std::forward_as_tuple(std::forward<Args>(args)...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::forward_as_tuple(std::get<start + I>(values)...);
  }(std::make_index_sequence<slice_size>{});
}


// ---------------------------- MatMul -------------------------------
// Static binary node: C = Left * Right (matrix × matrix).
template <typename Left, typename Right>
struct MatMul {
  using Left_ = std::decay_t<Left>;
  using Right_ = std::decay_t<Right>;
  static constexpr std::size_t ops = 2;
  template <typename L, typename R>
  MatMul(L&& left, R&& right)
      : left_(std::forward<L>(left)), right_(std::forward<R>(right)),
        rows_(left_.rows()), cols_(right_.cols()),
        value_ptr_(nullptr), adj_ptr_(nullptr) {
  }

  STRONG_INLINE std::pair<std::size_t, std::size_t> CacheBindSize() const {
    // Own storage + children.
    const std::size_t n = static_cast<std::size_t>(rows_) * cols_;
    auto [lv, la] = left_.CacheBindSize();
    auto [rv, ra] = right_.CacheBindSize();
    return {n + lv + rv, n + la + ra};
  }

  STRONG_INLINE void Bind(double __restrict* values_base, double __restrict* adjs_base,
            std::size_t& v_off, std::size_t& a_off) {
    // Bind children first (any order is fine) then self.
    value_ptr_ = values_base + v_off;
    adj_ptr_ = adjs_base + a_off;
    v_off += static_cast<std::size_t>(rows_) * cols_;
    a_off += static_cast<std::size_t>(rows_) * cols_;
    left_.Bind(values_base, adjs_base, v_off, a_off);
    right_.Bind(values_base, adjs_base, v_off, a_off);
  }
  STRONG_INLINE auto f_eval() {
    // value = left.value * right.value
    return Eigen::Map<Eigen::MatrixXd>(value_ptr_, rows_, cols_).noalias() = left_.f_eval() * right_.f_eval();
  }

  template <typename TT>
  STRONG_INLINE void b_eval(TT&& seed) {
    // Propagate: dL += dC * R^T ; dR += L^T * dC
    this->adjoint_map().array() += seed;
    auto l_adj = this->adjoint_map() * right_.value_map().transpose();
    left_.b_eval(std::move(l_adj));
    auto r_adj = left_.value_map().transpose() * this->adjoint_map();
    right_.b_eval(std::move(r_adj));
  }

  STRONG_INLINE auto value_map() { return Eigen::Map<Eigen::MatrixXd>(value_ptr_, rows_, cols_); }
  STRONG_INLINE auto adjoint_map() { return Eigen::Map<Eigen::MatrixXd>(adj_ptr_, rows_, cols_); }

  STRONG_INLINE Eigen::Index rows() const { return rows_; }
  STRONG_INLINE Eigen::Index cols() const { return cols_; }

  std::decay_t<Left> left_;
  std::decay_t<Right> right_;
  Eigen::Index rows_;
  Eigen::Index cols_;
  double __restrict* value_ptr_;
  double __restrict* adj_ptr_;
};

// ------------------------------ Sum --------------------------------
// Reduces all elements of a matrix to a scalar (1x1).
template <typename Child>
struct Sum {
 static constexpr std::size_t ops = 1;
  template <typename T>
  explicit Sum(T&& child)
      : child_(std::forward<T>(child)), val_(0), adj_(0) {}

  // Total storage: 1 for value + child; 1 for adjoint + child.
  STRONG_INLINE std::pair<std::size_t, std::size_t> CacheBindSize() const {
    return child_.CacheBindSize();
  }

  STRONG_INLINE void Bind(double __restrict* values_base, double __restrict* adjs_base,
            std::size_t& v_off, std::size_t& a_off) {
    child_.Bind(values_base, adjs_base, v_off, a_off);
  }
  STRONG_INLINE auto f_eval() {
    return val_ = child_.f_eval().sum();
  }
  template <typename TT>
  STRONG_INLINE void b_eval(TT&& seed) {
    adj_ += seed;
    child_.b_eval(seed);
  }

  // Convenience: seed output gradient to 1 (∂f/∂f).
  STRONG_INLINE void SeedOutputAdjoint() {
    adj_ = 1.0;
  }

  std::decay_t<Child> child_;
  double val_;
  double adj_;
};

// ---------------------------- Utility -------------------------------

template <typename Expr>
STRONG_INLINE std::pair<std::size_t, std::size_t> CacheBindSize(Expr&& expr) {
  return expr.CacheBindSize();
}

template <typename Expr>
STRONG_INLINE void Bind(Expr&& expr, double __restrict* values_base, double __restrict* adjs_base) {
  std::size_t v_off = 0, a_off = 0;
  expr.Bind(values_base, adjs_base, v_off, a_off);
}

template <typename Expr>
STRONG_INLINE void AutoDiff(Expr&& expr) {
  expr.f_eval();
  expr.b_eval(1.0);
}

template <typename Op1, typename Op2>
STRONG_INLINE auto operator*(Op1&& left, Op2&& right) {
  return MatMul<Op1, Op2>(std::forward<Op1>(left), std::forward<Op2>(right));
}
template <typename Op>
STRONG_INLINE auto sum(Op&& child) {
  return Sum<Op>(std::forward<Op>(child));
}


}

static void expr_template(benchmark::State& state) {
  // Dynamic inputs (2x2 here; any MxK and KxN will work).
  Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::MatrixXd B0 = Eigen::MatrixXd::Random(state.range(0), state.range(0));

  // 1) Build expression graph: f = sum(A * B). No storage yet.
  ad::Var<Eigen::MatrixXd> A(A0);
  ad::Var<Eigen::MatrixXd> B(B0);
  auto f = ad::sum(A * B);

  // 2) Ask graph how much contiguous storage it needs.
  auto [vsize, asize] = ad::CacheBindSize(f);

  // 3) Provide contiguous buffers and bind them to the graph.
  Eigen::VectorXd values(vsize);
  Eigen::VectorXd adjs(asize);
  values.setZero();
  adjs.setZero();
  ad::Bind(f, values.data(), adjs.data());
  for (auto _ : state) {
    // 4) Autodiff (forward + reverse).
    ad::AutoDiff(f);
    adjs.setZero();
  }
}
BENCHMARK(expr_template)-> RangeMultiplier(2) -> Range(1, 4096);
