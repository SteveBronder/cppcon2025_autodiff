include(FetchContent)
set(BENCHMARK_ENABLE_TESTING       OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS   OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_EXCEPTIONS     ON CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_LTO ON CACHE BOOL "" FORCE)
add_compile_options(-Wno-error)
# try a system one first

message(STATUS "Fetching google/benchmark (main)")
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(benchmark)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)
include(FetchContent)
set(EIGEN_TAG "f426eff949a989b415dd19f59ae1d5c7e107566a")

# ----------------------------------------------------------------------
# Common discovery fallbacks
# ----------------------------------------------------------------------
# Eigen3 (many of your targets link Eigen3::Eigen). Provide a fallback.
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        ${EIGEN_TAG}
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Eigen)

set(SCALAR_EXECUTABLES
    baseline
    shared_ptr
    mono_buffer
    lambda
    sct
)

foreach(exe ${SCALAR_EXECUTABLES})
    add_executable(${exe} ${exe}.cpp)
    target_compile_options(${exe} PRIVATE -march=native -mtune=native -O3 -g0)
    target_link_libraries(${exe} PRIVATE benchmark::benchmark benchmark::benchmark_main)
    target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

set(MATRIX_EXECUTABLES
    lambda_eigen
    lambda_eigen_special
    lambda_var_eigen
)

foreach(exe ${MATRIX_EXECUTABLES})
    add_executable(${exe} ${exe}.cpp)
    target_compile_options(${exe} PRIVATE -march=native -mtune=native -O3 -g0)
    target_link_libraries(${exe} PRIVATE benchmark::benchmark benchmark::benchmark_main Eigen3::Eigen)
    target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
