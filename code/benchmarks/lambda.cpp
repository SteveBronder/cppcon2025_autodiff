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

static std::pmr::monotonic_buffer_resource mbr{1<<16};
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
static alloc_t pa{&mbr};
struct var_impl {
    double value_;
    double adjoint_{0};
    virtual void chain() {};
    var_impl(double x) : value_(x), adjoint_(0) {}
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
    auto val() {
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
auto& adjoint(var x) { return x.vi_->adjoint_; }

auto value(var x) { return x.vi_->value_; }

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

template <typename Lambda>
struct lambda_var_impl : public var_impl {
    Lambda lambda_;
    lambda_var_impl(double val, Lambda&& lambda)
        : var_impl(val), lambda_(std::move(lambda)) {}
    void chain() final { 
      std::cout << "sizeof(lambda): " << sizeof(*this) << std::endl;
      lambda_(var(this)); 
    }
};
template <typename Lambda>
inline auto make_var(double ret_val, Lambda&& lambda) {
    return var(make_inbuffer<lambda_var_impl<Lambda>>(ret_val, std::move(lambda)));
}

template <typename T1, typename T2>
auto operator+(T1 lhs, T2 rhs) {
  return make_var(value(lhs) + value(rhs), [lhs, rhs](auto&& ret) {
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret);
    }
  });
}
var& var::operator+=(var x) {
    this->vi_ = ((*this) + x).vi_;
    return *this;
}

template <typename T1, typename T2>
auto operator*(T1 lhs, T2 rhs) {
  return make_var(value(lhs) + value(rhs), [lhs, rhs](auto&& ret) mutable {
    std::cout << "Mul: \n";
    if constexpr (!std::is_arithmetic_v<T1>) {
      adjoint(lhs) += adjoint(ret) * value(rhs);
    }
    if constexpr (!std::is_arithmetic_v<T2>) {
      adjoint(rhs) += adjoint(ret) * value(lhs);
    }
  });
}

auto log(var x) {
    return make_var(std::log(x.val()), [x](auto&& ret) mutable {
      std::cout << "Log: \n";
      x.adj() += ret.adj() / x.val();
    });
}

void grad(var z) noexcept {
    adjoint(z) = 1;
    std::cout << "\nStart Reverse: " << std::endl;
    for (auto&& x : var_vec | std::views::reverse) {
      x->chain();
    }
    std::cout << "End Reverse\n";
}
void clear_mem() {
    var_vec.clear();
    mbr.release();
}
int main() {
    {
        var x(2.0);
        var y(4.0);
        auto z = log(x*y) + y;
        grad(z);
        std::cout << "\nEnd: " << std::endl;
        std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")"
                  << std::endl;
        std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")"
                  << std::endl;
    }
    clear_mem();
    std::cout << "\n--------------------------\nNext: \n" << std::endl;
    {
        var x(2.0);
        var y(4.0);
        auto z = x * log(y*x);
        int iter = 0;
        while (value(z) < 10) {
            z += x * log(y) + log(x * y) * y;
            std::cout << "z: " << value(z) << std::endl;
            iter++;
        }
        grad(z);
        std::cout << "\nEnd: " << std::endl;
        std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")"
                  << std::endl;
        std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")"
                  << std::endl;
    }
}
