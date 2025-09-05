#ifndef STAN_MATH_REV_CORE_EIGEN_NUMTRAITS_HPP
#define STAN_MATH_REV_CORE_EIGEN_NUMTRAITS_HPP

#include <ad_ex/meta/Eigen.hpp>
#include <limits>

namespace Eigen {

/**
 * Numerical traits template override for Eigen for automatic
 * gradient variables.
 *
 * Documentation here:
 *   http://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <>
struct NumTraits<ad::var> : GenericNumTraits<ad::var> {
  using Real = double;
  using NonInteger = ad::var;
  using Nested = ad::var;
  using Literal = ad::var;
  /**
   * Return the precision for <code>ad::var</code> delegates
   * to precision for <code>double</code>.
   *
   * @return precision
   */
  static inline Real dummy_precision() {
    return NumTraits<double>::dummy_precision();
  }

  static inline Real epsilon() { return NumTraits<double>::epsilon(); }

  static inline Real highest() { return NumTraits<double>::highest(); }
  static inline Real lowest() { return NumTraits<double>::lowest(); }

  enum {
    /**
     * ad::var is not complex.
     */
    IsComplex = 0,

    /**
     * ad::var is not an integer.
     */
    IsInteger = 0,

    /**
     * ad::var is signed.
     */
    IsSigned = 1,

    /**
     * ad::var does not require initialization.
     */
    RequireInitialization = 0,

    /**
     * Twice the cost of copying a double.
     */
    ReadCost = 2 * NumTraits<double>::ReadCost,

    /**
     * This is just forward cost, but it's the cost of a single
     * addition (plus memory overhead) in the forward direction.
     */
    AddCost = NumTraits<double>::AddCost,

    /**
     * Multiply cost is single multiply going forward, but there's
     * also memory allocation cost.
     */
    MulCost = NumTraits<double>::MulCost
  };

  /**
   * Return the number of decimal digits that can be represented
   * without change.  Delegates to
   * <code>std::numeric_limits<double>::digits10()</code>.
   */
  static int digits10() { return std::numeric_limits<double>::digits10; }
};

/**
 * Traits specialization for Eigen binary operations for reverse-mode
 * autodiff and `double` arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ad::var, double, BinaryOp> {
  using ReturnType = ad::var;
};

/**
 * Traits specialization for Eigen binary operations for `double` and
 * reverse-mode autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, ad::var, BinaryOp> {
  using ReturnType = ad::var;
};

/**
 * Traits specialization for Eigen binary operations for reverse-mode
 * autodiff and `int` arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ad::var, int, BinaryOp> {
  using ReturnType = ad::var;
};

/**
 * Traits specialization for Eigen binary operations for `int` and
 * reverse-mode autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<int, ad::var, BinaryOp> {
  using ReturnType = ad::var;
};

/**
 * Traits specialization for Eigen binary operations for reverse-mode
 autodiff
 * arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ad::var, ad::var, BinaryOp> {
  using ReturnType = ad::var;
};

/**
 * Traits specialization for Eigen binary operations for `double` and
 * complex autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, std::complex<ad::var>, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for complex
 * autodiff and `double` arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<ad::var>, double, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for `int` and
 * complex autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<int, std::complex<ad::var>, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for complex
 * autodiff and `int` arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<ad::var>, int, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for autodiff and
 * complex `double` arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ad::var, std::complex<double>, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for complex
 * double and autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<double>, ad::var, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for complex
 * double and complex autodiff arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<double>, std::complex<ad::var>,
                            BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

/**
 * Traits specialization for Eigen binary operations for complex
 * autodiff and complex double arguments.
 *
 * @tparam BinaryOp type of binary operation for which traits are
 * defined
 */
template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<ad::var>, std::complex<double>,
                            BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<ad::var, std::complex<ad::var>,
                            BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<ad::var>, ad::var,
                            BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<std::complex<ad::var>,
                            std::complex<ad::var>, BinaryOp> {
  using ReturnType = std::complex<ad::var>;
};

namespace internal {


/**
 * Partial specialization of Eigen's remove_all struct to stop
 * Eigen removing pointer from var_impl* variables
 */
template <>
struct remove_all<ad::var_impl<double>*> {
  using type = ad::var_impl<double>*;
};

}  // namespace internal
}  // namespace Eigen
#endif
