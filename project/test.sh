#!/bin/bash
export PKG_CONFIG_PATH=/home/jasparke/CSC485C/pkgconfig
# g++ -std=c++17 benchmark.cpp -o test_run -mavx -march=native -fopenmp `pkg-config --cflags --libs opencv` -lpthread


# # full run.
declare -a images=("frac0.png" "frac1.png" "frac2.png" "frac3.png" "frac4.png" "frac5.png" "frac6.png" "frac7.png" "frac8.png" "frac9.png")
# declare -a test_cases=("0" "1" "2" "3" "4" "5" "6" "7" "8")
# declare -a thread_count=("1" "2" "4" "6" "12")


# declare -a images=("new-4000x2250.jpg" "new-3200-1800.jpg" "new-2800-1575.jpg" "new-2400-1350.jpg" "new-2000-1125.jpg" "new-1600-900.jpg" "new-1200-675.jpg" "new-800-450.jpg" "new-400-225.jpg" "new-200-114.jpg")


declare -a test_cases=("0" "1" "2" "3" "4" "5" "6" "7" "8")

# declare -a test_cases=("4")

declare -a thread_count=("1" "2" "4" "6" "12")

# cd project
mkdir -p outputs/basic
mkdir -p outputs/simd
mkdir -p outputs/time
# g++ -std=c++17 benchmark.cpp -o ./sobel -mavx -march=native -fopenmp `pkg-config --cflags --libs opencv` -lpthread
g++ -std=c++17 -Xpreprocessor -fopenmp -lomp benchmark.cpp -o sobel `pkg-config --cflags --libs opencv4` -mavx -march=native

for tc in "${test_cases[@]}"; do for img in "${images[@]}"; do for threads in "${thread_count[@]}"; do 
    ./sobel "sampleset/${img}" "$tc" "$threads" # > ./outputs/${img}_${tc}_${threads}
    # perf stat -o ./outputs/basic/${img}_${tc}_${threads} -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./sobel "sampleset/${img}" "$tc" "$threads" > ./outputs/time/${img}_${tc}_${threads}
    # perf stat -o ./outputs/simd/${img}_${tc}_${threads} -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single ./sobel "sampleset/${img}" "$tc" "$threads"
done; done; done


