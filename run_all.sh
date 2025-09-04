cmake --build build -j4
find ./build/code/benchmarks/ -maxdepth 1 -type f -exec {} \;
