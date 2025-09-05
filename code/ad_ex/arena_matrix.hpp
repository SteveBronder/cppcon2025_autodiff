#ifndef STAN_MATH_REV_CORE_ARENA_MATRIX_HPP
#define STAN_MATH_REV_CORE_ARENA_MATRIX_HPP

#include <ad_ex/meta/is_eigen.hpp>
#include <ad_ex/eigen_numtraits.hpp>

namespace ad {

template <Matrix MatrixType>
class arena_matrix : public Eigen::Map<std::decay_t<MatrixType>> {
 public:
  using Scalar = typename std::decay_t<MatrixType>::Scalar;
  using Base = Eigen::Map<std::decay_t<MatrixType>>;
  using PlainObject = std::decay_t<MatrixType>;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;

  /**
   * Default constructor.
   */
  arena_matrix()
      : Base::Map(nullptr,
                  RowsAtCompileTime == Eigen::Dynamic ? 0 : RowsAtCompileTime,
                  ColsAtCompileTime == Eigen::Dynamic ? 0 : ColsAtCompileTime) {
  }

  /**
   * Constructs `arena_matrix` with given number of rows and columns.
   * @param rows number of rows
   * @param cols number of columns
   */
  arena_matrix(Eigen::Index rows, Eigen::Index cols)
      : Base::Map(
          (Scalar*)pa.allocate_bytes(sizeof(Scalar) * rows * cols),
          rows, cols) {}

  /**
   * Constructs `arena_matrix` with given size. This only works if
   * `MatrixType` is row or col vector.
   * @param size number of elements
   */
  explicit arena_matrix(Eigen::Index size)
      : Base::Map(
          (Scalar*)pa.allocate_bytes(sizeof(Scalar) * size),
          size) {}

 private:
  template <typename T>
  constexpr auto get_rows(const T& x) {
    return (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
                   || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
               ? x.cols()
               : x.rows();
  }
  template <typename T>
  constexpr auto get_cols(const T& x) {
    return (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1)
                   || (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
               ? x.rows()
               : x.cols();
  }

 public:
  /**
   * Constructs `arena_matrix` from an expression
   * @param other expression
   */
  template <typename T>
  arena_matrix(const T& other)  // NOLINT
      : Base::Map((Scalar*)pa.allocate_bytes(sizeof(Scalar) *
                      other.size()),
                  get_rows(other), get_cols(other)) {
    *this = other;
  }
  /**
   * Overwrite the current arena_matrix with new memory and assign a matrix to
   * it
   * @tparam T An eigen type inheriting from `Eigen::EigenBase`
   * @param other A matrix that will be copied over to the arena allocator
   */
  template <Matrix T>
  arena_matrix& operator=(const T& other) {
    new (this) Base(
        (Scalar*)pa.allocate_bytes(sizeof(Scalar) * other.size()),
        get_rows(other), get_cols(other));
    Base::operator=(other);
    return *this;
  }


  /**
   * Constructs `arena_matrix` from an expression. This makes an assumption that
   * any other `Eigen::Map` also contains memory allocated in the arena.
   * @param other expression
   */
  arena_matrix(const Base& other)  // NOLINT
      : Base::Map(other) {}

  /**
   * Copy constructor.
   * @param other matrix to copy from
   */
  arena_matrix(const arena_matrix<MatrixType>& other)
      : Base::Map(const_cast<Scalar*>(other.data()), other.rows(),
                  other.cols()) {}

  // without this using, compiler prefers combination of implicit construction
  // and copy assignment to the inherited operator when assigned an expression
  using Base::operator=;

  /**
   * Copy assignment operator.
   * @param other matrix to copy from
   * @return `*this`
   */
  arena_matrix& operator=(const arena_matrix<MatrixType>& other) {
    // placement new changes what data map points to - there is no allocation
    new (this)
        Base(const_cast<Scalar*>(other.data()), other.rows(), other.cols());
    return *this;
  }

  /**
   * Forces hard copying matrices into an arena matrix
   * @tparam T Any type assignable to `Base`
   * @param x the values to write to `this`
   */
  template <typename T>
  void deep_copy(const T& x) {
    Base::operator=(x);
  }
};

}  // namespace stan

namespace Eigen {
namespace internal {

template <typename T>
struct traits<ad::arena_matrix<T>> {
  using base = traits<Eigen::Map<T>>;
  using Scalar = typename base::Scalar;
  using XprKind = typename Eigen::internal::traits<std::decay_t<T>>::XprKind;
  using StorageKind = typename Eigen::internal::traits<std::decay_t<T>>::StorageKind;
  static constexpr int RowsAtCompileTime = Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime;
  enum {
    PlainObjectTypeInnerSize = base::PlainObjectTypeInnerSize,
    InnerStrideAtCompileTime = base::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = base::OuterStrideAtCompileTime,
    Alignment = base::Alignment,
    Flags = base::Flags
  };
};

}  // namespace internal
}  // namespace Eigen

namespace ad {
  namespace detail {
    template <typename T>
    struct arena_type {
      using type = arena_matrix<T>;
    };
    template <>
    struct arena_type<ad::var> {
      using type = ad::var;
    };
    template <typename T>
    struct arena_type<arena_matrix<T>> {
      using type = arena_matrix<T>;
    };
  }
  template <typename T>
  using arena_t = typename detail::arena_type<std::decay_t<T>>::type;
}
#endif
