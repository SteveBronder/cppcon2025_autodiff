#ifndef STAN_MATH_PRIM_META_IS_EIGEN_HPP
#define STAN_MATH_PRIM_META_IS_EIGEN_HPP

#include <ad_ex/meta/Eigen.hpp>
#include <type_traits>
#include <ad_ex/meta/is_base_pointer_convertible.hpp>

namespace ad {

/**
 * Check if type derives from `EigenBase`
 * @tparam T Type to check if it is derived from `EigenBase`
 * @tparam Enable used for SFINAE deduction.
 * @ingroup type_trait
 **/
template <typename T>
struct is_eigen
    : std::bool_constant<is_base_pointer_convertible<Eigen::EigenBase, T>::value> {};

template <typename T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

namespace internal {
// primary template handles types that have no nested ::type member:
template <class, class = void>
struct has_internal_trait : std::false_type {};

// specialization recognizes types that do have a nested ::type member:
template <class T>
struct has_internal_trait<T,
                          std::void_t<Eigen::internal::traits<std::decay_t<T>>>>
    : std::true_type {};

// primary template handles types that have no nested ::type member:
template <class, class = void>
struct has_scalar_trait : std::false_type {};

// specialization recognizes types that do have a nested ::type member:
template <class T>
struct has_scalar_trait<T, std::void_t<typename std::decay_t<T>::Scalar>>
    : std::true_type {};

}  // namespace internal

template <typename T>
concept Matrix = is_eigen_v<std::decay_t<T>>;


}  // namespace ad
#endif
