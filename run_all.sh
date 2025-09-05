cmake --build build -j4
find ./build/code/ -maxdepth 1 -type f -exec {} \;
