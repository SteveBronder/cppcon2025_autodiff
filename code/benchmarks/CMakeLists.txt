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


set(AD_EX_OPTIMS OFF CACHE STRING "-march=native -mtune=native -O3" FORCE)

add_executable(shared_ptr_ex shared_ptr.cpp)
target_compile_definitions(shared_ptr_ex PRIVATE ${AD_EX_OPTIMS})
target_link_libraries(shared_ptr_ex PRIVATE benchmark::benchmark benchmark::benchmark_main)

add_executable(mono_buffer_ex mono_buffer.cpp)
target_compile_definitions(mono_buffer_ex PRIVATE ${AD_EX_OPTIMS})
target_link_libraries(mono_buffer_ex PRIVATE benchmark::benchmark benchmark::benchmark_main)

add_executable(lambda_ex lambda.cpp)
target_compile_definitions(lambda_ex PRIVATE ${AD_EX_OPTIMS})
target_link_libraries(lambda_ex PRIVATE benchmark::benchmark benchmark::benchmark_main)



add_executable(sct_ex source_code_transform.cpp)
target_compile_definitions(sct_ex PRIVATE ${AD_EX_OPTIMS})
target_link_libraries(sct_ex PRIVATE benchmark::benchmark benchmark::benchmark_main)
