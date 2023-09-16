cd Release
make all
cd ..

stdbuf -oL ./Release/SGD 0.01 32 100 > log_print_output.txt