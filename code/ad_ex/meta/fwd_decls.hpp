#pragma once

namespace ad {
// Forward declaration of the templated base implementation.
template <typename T>
struct var_base;

template <typename T>
struct var_impl;

using var = var_impl<double>;

} // namespace ad
