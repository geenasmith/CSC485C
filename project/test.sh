#!/bin/bash
NUM_TRIALS=1000
# # full run.
declare -a images=("earth-220-220.jpg" "earth-441-441.jpg" "earth-883-883.jpg" "earth-1250-1250.jpg" "earth-1767-1767.jpg" "earth-2500-2500.jpg" "earth-3535-3535.jpg" "earth-5000-5000.jpg" "earth-7071-7071.jpg" "earth-10000-10000.jpg" "earth-15000-15000.jpg")

# declare -a images=("new-4000x2250.jpg" "new-3200-1800.jpg" "new-2800-1575.jpg" "new-2400-1350.jpg" "new-2000-1125.jpg" "new-1600-900.jpg" "new-1200-675.jpg" "new-800-450.jpg" "new-400-225.jpg" "new-200-114.jpg")


declare -a test_cases_cpu=("0" "1")
declare -a test_cases_gpu=("0" "1" "2")
declare -a blocksize_gpu=("1 1" "32 1" "32 2" "32 4" "32 8" "32 16" "32 32" "22 22" "4 32")

# cd project
mkdir -p outputs/logs
nvcc -O3 sobel.cu -o sobel `pkg-config --cflags --libs opencv4`

for tc in "${test_cases_cpu[@]}"; do for img in "${images[@]}"; do 
    ./sobel $NUM_TRIALS "images/earth_set/${img}" "cpu" ${tc} > "outputs/logs/${img}_cpu${tc}.log"
done; done;

for tc in "${test_cases_cpu[@]}"; do for img in "${images[@]}"; do for bs in "${blocksize_gpu[@]}"; do
    ./sobel $NUM_TRIALS "images/earth_set/${img}" "gpu" ${tc} $bs > "outputs/logs/${img}_gpu${tc}_${bs}.log"
done; done; done;
