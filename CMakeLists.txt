cmake_minimum_required(VERSION 3.7)
project("cppcon_ad"
        VERSION 1.0
        DESCRIPTION "Benchmarks etc for simple autodiff impls."
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if( ipo_supported )
    message(STATUS "IPO / LTO enabled")
    set(INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

# Automate the choosing of config
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()
message(STATUS "Compiling in ${CMAKE_BUILD_TYPE} mode")

add_subdirectory(${PROJECT_SOURCE_DIR}/code/benchmarks)
