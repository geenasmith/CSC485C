#!/bin/bash
declare -a images=("sampleset/frac0.png" "sampleset/frac1.png" "sampleset/frac2.png" "sampleset/frac3.png" "sampleset/frac4.png" "sampleset/frac5.png" "sampleset/frac6.png" "sampleset/frac7.png" "sampleset/frac8.png" "sampleset/frac9.png")
declare -a test_cases=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a thread_count=("1")

for tc in "${test_cases[@]}"; do for img in "${images[@]}"; do for threads in "${thread_count[@]}"; do 
    ./sobel "${img}" "$tc" "$threads"
done; done; done