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
struct add_vv : public var_impl {
  var lhs_;
  var rhs_;
  add_vv(double val, var lhs, var rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    std::cout << "\tPre:\n";
    std::cout << "sizeof(add): " << sizeof(add_vv) << std::endl;
    print_var("Add", lhs_, rhs_);
    lhs_.adj() += this->adjoint_;
    rhs_.adj() += this->adjoint_;
    std::cout << "\tPost:\n";
    print_var("Add", lhs_, rhs_);
  }
};
struct add_dv : public var_impl {
  double lhs_;
  var rhs_;
  add_dv(double val, double lhs, var rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    rhs_.adj() += this->adjoint_;
  }
};
struct add_vd : public var_impl {
  var lhs_;
  double rhs_;
  add_vd(double val, var lhs, double rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += this->adjoint_;
  }
};
template <typename T1, typename T2>
auto operator+(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<add_vv>(lhs.val() + rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{make_inbuffer<add_vd>(lhs.val() + rhs), lhs, rhs};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<add_dv>(lhs + rhs.val()), lhs, rhs};
  }
}
var& var::operator+=(var x) {
    this->vi_ = ((*this) + x).vi_;
    return *this;
}

struct mul_vv : public var_impl {
  var lhs_;
  var rhs_;
  mul_vv(double val, var lhs, var rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    std::cout << "Mul(v,v):\n";
    std::cout << "\tPre:\n";
    std::cout << "\tlhs.val(" << lhs_.val() << ") rhs.val(" << rhs_.val() << ")\n";
    std::cout << "\tlhs.adj(" << lhs_.adj() << ") rhs.adj(" << rhs_.adj() << ")\n";
    std::cout << "\tthis.val(" << this->value_ << ") this.adj(" << this->adjoint_ << ")\n";
    lhs_.adj() += rhs_.val() * this->adjoint_;
    rhs_.adj() += lhs_.val() * this->adjoint_;
    std::cout << "\tPost:\n";
    std::cout << "\tlhs.val(" << lhs_.val() << ") rhs.val(" << rhs_.val() << ")\n";
    std::cout << "\tlhs.adj(" << lhs_.adj() << ") rhs.adj(" << rhs_.adj() << ")\n";
  }
};
struct mul_dv : public var_impl {
  double lhs_;
  var rhs_;
  mul_dv(double val, double lhs, var rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  
  void chain() {
    rhs_.adj() += lhs_ * this->adjoint_;
  }
};

struct mul_vd : public var_impl {
  var lhs_;
  double rhs_;
  mul_vd(double val, var lhs, double rhs) : 
    var_impl(val), lhs_(lhs), rhs_(rhs) {}
  void chain() {
    lhs_.adj() += rhs_ * this->adjoint_;
  }
};

template <typename T1, typename T2>
auto operator*(T1 lhs, T2 rhs) {
  if constexpr (!std::is_arithmetic_v<T1> && !std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<mul_vv>(lhs.val() * rhs.val(), lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T1>) {
    return var{make_inbuffer<mul_vd>(lhs.val() * rhs, lhs, rhs)};
  } else if constexpr (!std::is_arithmetic_v<T2>) {
    return var{make_inbuffer<mul_dv>(lhs * rhs.val(), lhs, rhs)};
  }
}

struct log_var : public var_impl {
  var in_;
  log_var(double x, var in) : var_impl(x), in_(in) {}
  void chain() { 
    std::cout << "Log(v):\n";
    std::cout << "\tPre:\n";
    std::cout << "\tin.val(" << in_.val() << ") \n";
    std::cout << "\tin.adj(" << in_.adj() << ") \n";
    std::cout << "\tthis.val(" << this->value_ << ") this.adj(" << this->adjoint_ << ")\n";
    in_.adj() += this->adjoint_ / in_.val();
    std::cout << "\tPost:\n";
    std::cout << "\tin.val(" << in_.val() << ") \n";
    std::cout << "\tin.adj(" << in_.adj() << ") \n";
    std::cout << "\tthis.val(" << this->value_ << ") this.adj(" << this->adjoint_ << ")\n";
  }
};

auto log(var x) {
    return var{make_inbuffer<log_var>(std::log(x.val()), x)};
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
