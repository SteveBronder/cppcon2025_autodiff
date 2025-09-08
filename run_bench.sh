# options shared_ptr mono_buffer_ex lambda_ex sct_ex
cmake --build build --target $1 -j4 && ./build/code/$1
