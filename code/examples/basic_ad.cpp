// Type your code here, or load an example.
#include <stdint.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define stan_inline __attribute__((always_inline)) inline

namespace internal {
constexpr size_t DEFAULT_INITIAL_NBYTES = 1 << 16;  // 64KB
}  // namespace internal

/**
 * An instance of this class provides a memory pool through
 * which blocks of raw memory may be allocated and then collected
 * simultaneously.
 *
 * This class is useful in settings where large numbers of small
 * objects are allocated and then collected all at once.  This may
 * include objects whose destructors have no effect.
 *
 * Memory is allocated on a stack of blocks.  Each block allocated
 * is twice as large as the previous one.  The memory may be
 * recovered, with the blocks being reused, or all blocks may be
 * freed, resetting the stack of blocks to its original state.
 *
 * Alignment up to 8 byte boundaries guaranteed for the first malloc,
 * and after that it's up to the caller.  On 64-bit architectures,
 * all struct values should be padded to 8-byte boundaries if they
 * contain an 8-byte member or a virtual function.
 */
class arena_alloc {
   private:
    std::vector<char*> blocks_;  // storage for blocks,
                                 // may be bigger than cur_block_
    std::vector<size_t> sizes_;  // could store initial & shift for others
    size_t cur_block_;           // index into blocks_ for next alloc
    char* cur_block_end_;        // ptr to cur_block_ptr_ + sizes_[cur_block_]
    char* next_loc_;             // ptr to next available spot in cur
                                 // block
    // next three for keeping track of nested allocations on top of stack:
    std::vector<size_t> nested_cur_blocks_;
    std::vector<char*> nested_next_locs_;
    std::vector<char*> nested_cur_block_ends_;

    /**
     * Moves us to the next block of memory, allocating that block
     * if necessary, and allocates len bytes of memory within that
     * block.
     *
     * @param len Number of bytes to allocate.
     * @return A pointer to the allocated memory.
     */
    stan_inline char* move_to_next_block(size_t len) {
        char* result;
        ++cur_block_;
        // Find the next block (if any) containing at least len bytes.
        while ((cur_block_ < blocks_.size()) && (sizes_[cur_block_] < len)) {
            ++cur_block_;
        }
        // Allocate a new block if necessary.
        if (unlikely(cur_block_ >= blocks_.size())) {
            // New block should be max(2*size of last block, len) bytes.
            size_t newsize = sizes_.back() * 2;
            if (newsize < len) {
                newsize = len;
            }
            blocks_.push_back(static_cast<char*>(malloc(newsize)));
            if (!blocks_.back()) {
                throw std::bad_alloc();
            }
            sizes_.push_back(newsize);
        }
        result = blocks_[cur_block_];
        // Get the object's state back in order.
        next_loc_ = result + len;
        cur_block_end_ = result + sizes_[cur_block_];
        return result;
    }

   public:
    /**
     * Construct a resizable stack allocator initially holding the
     * specified number of bytes.
     *
     * @param initial_nbytes Initial number of bytes for the
     * allocator.  Defaults to <code>(1 << 16) = 64KB</code> initial bytes.
     * @throws std::bad_alloc if malloc fails
     */
    explicit arena_alloc(
        size_t initial_nbytes = internal::DEFAULT_INITIAL_NBYTES)
        : blocks_(1, static_cast<char*>(malloc(initial_nbytes))),
          sizes_(1, initial_nbytes),
          cur_block_(0),
          cur_block_end_(blocks_[0] + initial_nbytes),
          next_loc_(blocks_[0]) {
        if (!blocks_[0]) {
            throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
        }
    }

    /**
     * Destroy this memory allocator.
     *
     * This is implemented as a no-op as there is no destruction
     * required.
     */
    ~arena_alloc() {
        // free ALL blocks
        for (auto& block : blocks_) {
            if (block) {
                free(block);
            }
        }
    }

    /**
     * Return a newly allocated block of memory of the appropriate
     * size managed by the stack allocator.
     *
     * The allocated pointer will be 8-byte aligned.
     *
     * This function may call C++'s <code>malloc()</code> function,
     * with any exceptions percolated through this function.
     *
     * @param len Number of bytes to allocate.
     * @return A pointer to the allocated memory.
     */
    stan_inline void* alloc(size_t len) {
        // Typically, just return and increment the next location.
        char* result = next_loc_;
        next_loc_ += len;
        // Occasionally, we have to switch blocks.
        if (unlikely(next_loc_ >= cur_block_end_)) {
            result = move_to_next_block(len);
        }
        return reinterpret_cast<void*>(result);
    }

    /**
     * Recover all the memory used by the stack allocator.  The stack
     * of memory blocks allocated so far will be available for further
     * allocations.  To free memory back to the system, use the
     * function free_all().
     */
    stan_inline void recover_all() {
        cur_block_ = 0;
        next_loc_ = blocks_[0];
        cur_block_end_ = next_loc_ + sizes_[0];
    }

};

/**
 * Start of autodiff
 */
struct var_impl;

/**
 * Tracks the autodiff tape for reverse mode
 */
static std::vector<var_impl*> reverse_stack;
/**
 * Holds memory for reverse mode
 */
static arena_alloc arena;

/** 
 * For example, show intermediary expression index
 */
static int tape_tracker = 0;

// Clear reverse tape and clean up memory
void release_mem() {
    reverse_stack.clear();
    arena.recover_all();
    tape_tracker = 0;
}

struct var_impl {
    double value_;
    double adjoint_{0};
    virtual void reverse() {
        std::cout << "Var: " << "(" << value_ << ", " << adjoint_ << ")" << std::endl;
    };
    stan_inline var_impl(double x) : value_(x), adjoint_(0) {
        tape_tracker++;
        std::cout << "v_" << tape_tracker << std::endl;
        reverse_stack.push_back(this);
    }
    // memory for this struct comes from the arena
    stan_inline void* operator new(size_t size) { return arena.alloc(size); }
};

struct var {
    var_impl* vi_; // ptr to impl
    // Put new var_impl on arena
    stan_inline var(double x) : vi_(new var_impl(x)) {}
    // Copy existing var impl to this
    template <typename VarImpl>
    stan_inline var(VarImpl* vi) : vi_(vi) {}
    stan_inline var& operator+=(var x);
};

/**
 * Helper functions
 */
stan_inline auto& adjoint(var x) { return x.vi_->adjoint_; }

stan_inline auto value(var x) { return x.vi_->value_; }

void print_var(const char* name, var& ret, var x) {
    std::cout << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
              << std::endl;
    std::cout << name << " Op: (" << value(x) << ", " << adjoint(x) << ")"
              << std::endl;
}

void print_var(const char* name, var& ret, var x, var y) {
    std::cout << name << ": (" << value(ret) << ", " << adjoint(ret) << ")"
              << std::endl;
    std::cout << name << " OpL: (" << value(x) << ", " << adjoint(x) << ")"
              << std::endl;
    std::cout << name << " OpR: (" << value(y) << ", " << adjoint(y) << ")"
              << std::endl;
}

/**
 * lambda_var_impl replaces having to do a new class 
 *  for each function. If we did not have `lambda_var_impl`
 *  we would generate code like 
   ```
   struct Add : public var_impl{
    var lhs_;
    var rhs_;
    Add(var lhs, var rhs) : 
     var_impl(value(lhs) + value(rhs)), 
     lhs_(lhs), rhs_(rhs) {}
    void reverse() {
        adjoint(lhs) += adjoint(this);
        adjoint(rhs) += adjoint(this);
        print_var("Add", *this, lhs, rhs);
    }
   };
   ```
 */
/**
 * Calls a user defined lambda during the reverse pass of the AD tape
 */
template <typename Lambda>
struct lambda_var_impl : public var_impl {
    Lambda lambda_;
    lambda_var_impl(double val, Lambda&& lambda)
        : var_impl(val), lambda_(std::move(lambda)) {}
    void reverse() final { lambda_(var(this)); }
};

/* Create a var that calls a user defined lambda*/
template <typename Lambda>
stan_inline auto make_var(double ret_val, Lambda&& lambda) {
    return var(new lambda_var_impl<Lambda>(ret_val, std::move(lambda)));
}


stan_inline auto operator+(var lhs, var rhs) {
    std::cout << "add: ";
    return make_var(value(lhs) + value(rhs), [lhs, rhs](var ret) {
        adjoint(lhs) += adjoint(ret);
        adjoint(rhs) += adjoint(ret);
        print_var("Add", ret, lhs, rhs);
    });
}
stan_inline var& var::operator+=(var x) {
    vi_ = (*this + x).vi_;
    return *this;
}

stan_inline auto operator*(var lhs, var rhs) {
    std::cout << "multiply: ";
    return make_var(value(lhs) * value(rhs), [lhs, rhs](var ret) {
        adjoint(lhs) += value(rhs) * adjoint(ret);
        adjoint(rhs) += value(lhs) * adjoint(ret);
        print_var("Multiply", ret, lhs, rhs);
    });
}

stan_inline auto log(var x) {
    std::cout << "log: ";
    return make_var(std::log(value(x)), [x](var ret) {
        adjoint(x) += (1.0 / value(x)) * adjoint(ret);
        print_var("log", ret, x);
    });
}
stan_inline auto sin(var x) {
    std::cout << "sin: ";
    return make_var(std::sin(value(x)), [x](var ret) {
      adjoint(x) += std::cos(value(x)) * adjoint(ret);
        print_var("sin", ret, x);
    });
} 

stan_inline void grad(var z) noexcept {
    adjoint(z) = 1;
    std::cout << "\nStart Reverse: " << std::endl;
    for (auto it = reverse_stack.rbegin(); it != reverse_stack.rend(); ++it) {
    std::cout << "-----------" << std::endl;
        std::cout << "v_" << tape_tracker << std::endl;
        (*it)->reverse();
        tape_tracker--;
    }
}

int main() {
    {
        var x(2.0);
        var y(4.0);
        auto z = log(x) * y + sin(x);
        grad(z);
        std::cout << "\nEnd: " << std::endl;
        std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")"
                  << std::endl;
        std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")"
                  << std::endl;
        release_mem();
    }
    std::cout << "\n--------------------------\nNext: \n" << std::endl;
    {
        var x(2.0);
        var y(4.0);
        auto z = x * log(y) + sin(y);
        int iter = 0;
        while (value(z) < 10) {
            z += x * log(y) + sin(y);
            std::cout << "z: " << value(z) << std::endl;
            iter++;
        }
        grad(z);
        std::cout << "\nEnd: " << std::endl;
        std::cout << "y: (" << value(y) << ", " << adjoint(y) << ")"
                  << std::endl;
        std::cout << "x: (" << value(x) << ", " << adjoint(x) << ")"
                  << std::endl;
        release_mem();
    }
}
