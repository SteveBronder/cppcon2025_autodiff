// Type your code here, or load an example.
#include <stdint.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
struct var_impl {
  double value_;
  double adjoint_{0};
  virtual void chain() {};
  var_impl(double x) : value_(x), adjoint_(0) {}
};

struct var {
  std::shared_ptr<var_impl> vi_;  // ptr to impl
  var(const std::shared_ptr<var_impl>& x) : vi_(x) {}
  // Put new var_impl on arena
  var(double x) : vi_(std::make_shared<var_impl>(x)) {}
  var& operator+=(var x);
  auto& adj() { return vi_->adjoint_; }
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
auto& adjoint(var x) { return x.vi_->adjoint_; }
auto& adjoint(const var_impl& x) { return x.adjoint_; }

auto value(var x) { return x.vi_->value_; }
auto value(const var_impl& x) { return x.value_; }

void print_var(const char* name, var_impl& ret, var x) {
  std::cout << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
            << std::endl;
  std::cout << name << " Op: (" << value(x) << ", " << adjoint(x) << ")"
            << std::endl;
}


void print_var(const char* name, var_impl& ret, var x, var y) {
  std::cout << "\t" << name << ": (" << value(ret) << ", " << adjoint(ret)
            << ")" << std::endl;
  std::cout << "\t" << " OpL: (" << value(x) << ", " << adjoint(x)
            << ")" << std::endl;
  std::cout << "\t" << " OpR: (" << value(y) << ", " << adjoint(y)
            << ")" << std::endl;
}

struct add_vv : public var_impl {
  var lhs_;
  var rhs_;
  add_vv(double val, var lhs, var rhs) : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    print_var("Add Pre:", (*this), lhs_, rhs_);
    lhs_.adj() += this->adjoint_;
    rhs_.adj() += this->adjoint_;
    print_var("Add Post:", (*this), lhs_, rhs_);
    lhs_.chain();
    rhs_.chain();
  }
};
struct add_dv : public var_impl {
  double lhs_;
  var rhs_;
  add_dv(double val, double lhs, var rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    rhs_.adj() += this->adjoint_;
    rhs_.chain();
  }
};
struct add_vd : public var_impl {
  var lhs_;
  double rhs_;
  add_vd(double val, var lhs, double rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += this->adjoint_;
    lhs_.chain();
  }
};
template <typename T1, typename T2>
auto operator+(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{std::make_shared<add_vv>(lhs.val() + rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{std::make_shared<add_vd>(lhs.val() + rhs), lhs, rhs};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{std::make_shared<add_dv>(lhs + rhs.val()), lhs, rhs};
  }
}
var& var::operator+=(var x) {
  this->vi_ = ((*this) + x).vi_;
  return *this;
}

struct mul_vv : public var_impl {
  var lhs_;
  var rhs_;
  mul_vv(double val, var lhs, var rhs) : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    print_var("Mul Pre:", (*this), lhs_, rhs_);
    lhs_.adj() += rhs_.val() * this->adjoint_;
    rhs_.adj() += lhs_.val() * this->adjoint_;
    print_var("Mul Post:", (*this), lhs_, rhs_);
    lhs_.chain();
    rhs_.chain();
  }
};

struct mul_dv : public var_impl {
  double lhs_;
  var rhs_;
  mul_dv(double val, double lhs, var rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}

  void chain() {
    rhs_.adj() += lhs_ * this->adjoint_;
    rhs_.chain();
  }
};

struct mul_vd : public var_impl {
  var lhs_;
  double rhs_;
  mul_vd(double val, var lhs, double rhs)
      : var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += rhs_ * this->adjoint_;
    lhs_.chain();
  }
};

template <typename T1, typename T2>
auto operator*(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{std::make_shared<mul_vv>(lhs.val() * rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{std::make_shared<mul_vd>(lhs.val() * rhs, lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{std::make_shared<mul_dv>(lhs * rhs.val(), lhs, rhs)};
  }
}

struct log_var : public var_impl {
  var in_;
  log_var(double x, var in) : var_impl(x), in_(in) {}
  void chain() {
    print_var("Log Pre:", (*this), in_);
    in_.adj() += this->adjoint_ / in_.val();
    print_var("Log Post:", (*this), in_);
    in_.chain();
  }
};

auto log(var x) {
  std::cout << "log: ";
  return var{std::make_shared<log_var>(std::log(x.val()), x)};
}

void grad(var z) noexcept {
  adjoint(z) = 1;
  std::cout << "\nStart Reverse: " << std::endl;
  z.chain();
}

int main() {
  {
    var x(2.0);
    var y(4.0);
    auto z = log(x * y) + y;
    grad(z);
    std::cout << "\nEnd: " << std::endl;
    std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")" << std::endl;
    std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")" << std::endl;
  }
  std::cout << "\n--------------------------\nNext: \n" << std::endl;
  {
    var x(2.0);
    var y(4.0);
    auto z = x * log(y * x);
    int iter = 0;
    while (value(z) < 10) {
      z += x * log(y) + log(x * y) * y;
      std::cout << "z: " << value(z) << std::endl;
      iter++;
    }
    grad(z);
    std::cout << "\nEnd: " << std::endl;
    std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")" << std::endl;
    std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")" << std::endl;
  }
}
