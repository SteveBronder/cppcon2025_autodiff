include(FetchContent)
set(BENCHMARK_ENABLE_TESTING       OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS   OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_EXCEPTIONS     ON CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_LTO ON CACHE BOOL "" FORCE)
add_compile_options(-Wno-error)
# try a system one first
if(NOT TARGET benchmark::benchmark)
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
endif()


set(MY_EXECUTABLES
    baseline
    shared_ptr
    mono_buffer
    lambda
    sct
)

foreach(exe ${MY_EXECUTABLES})
    add_executable(${exe} ${exe}.cpp)
    target_compile_options(${exe} PRIVATE -march=native -mtune=native -O3 -g3)
    target_link_libraries(${exe} PRIVATE benchmark::benchmark benchmark::benchmark_main)
endforeach()
