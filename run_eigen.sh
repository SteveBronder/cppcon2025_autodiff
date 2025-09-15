cmake --build build --target lambda_eigen lambda_eigen_special lambda_var_eigen expr_template -j4 && \
./build/code/expr_template --benchmark_out=./build/res/expr_template.csv --benchmark_out_format=csv && \
./build/code/lambda_var_eigen --benchmark_out=./build/res/lambda_var_eigen.csv --benchmark_out_format=csv && \
./build/code/lambda_eigen --benchmark_out=./build/res/lambda_eigen.csv --benchmark_out_format=csv && \
./build/code/lambda_eigen_special --benchmark_out=./build/res/lambda_eigen_special.csv --benchmark_out_format=csv 
