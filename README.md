# CSC485C

Project and course work

# Compile

```
g++ -std=c++17 benchmark.cpp -o ./sobel -mavx -march=native -fopenmp `pkg-config --cflags --libs opencv` -lpthread
```

# Running

Implementations are runnable using a command line argument:
```bash
# usage:
./sobel "IMAGE_PATH" CASE_TO_RUN THREADS_TO_USE


# Samples #

# Baseline, 4k image, 1 thread. Expected: ~504000us
./sobel "sampleset/frac0.png" 0 1

# Initial optimization, 4k image, 1 thread. Expected: ~194295us
./sobel "sampleset/frac0.png" 1 1

# Best Optimization, report1, 4k image, 12 threads. Expected: ~135000us
./sobel "sampleset/frac0.png" 3 12 

# Best Optimization, report2, 4k image, 12 threads. Uses simd and openmp. Expected: ~11950us
./sobel "sampleset/frac0.png" 8 12 

# SIMD only, 4k image, 1 threads. Expected: ~67449.5us
./sobel "sampleset/frac0.png" 6 1
```

All arguments are required, and the full spec of cases are listed in this code block from benchmark.cpp:

```c++
switch(test_case) {
case 0:
    sum += baseline(file_name, benchmark_trials, display_outputs);
    break;
case 1:
    sum += report1_float(file_name, benchmark_trials, display_outputs);
    break;
case 2:
    sum += report1_uint8(file_name, benchmark_trials, display_outputs);
    break;
case 3:
    sum += report1_uint8_fastsqrt(file_name, benchmark_trials, display_outputs);
    break;

/******************
 * Report 2 OpenMP*
 ******************/
case 4:
    sum += report2_openmp_coarse(file_name, benchmark_trials, display_outputs);
    break;
case 5:
    sum += report2_openmp_fine(file_name, benchmark_trials, display_outputs);
    break;

/******************
 * Report 2 SIMD  *
 ******************/
case 6:
    sum += report2_simd(file_name, benchmark_trials, display_outputs);
    break;
case 7:
    sum += report2_simd_ompcoarse(file_name, benchmark_trials, display_outputs);
    break;
case 8:
    sum += report2_simd_ompfine(file_name, benchmark_trials, display_outputs);
    break;
case 9:
    // sum += report2_simd_uint8(file_name, benchmark_trials, display_outputs);
    break;
default:
    std::cout << "invalid test#" << std::endl;
}
```

`project/test.sh` provides a way to run multiple iterations at once across multiple images. There is a bug in some of these implementations that will cause segfaults in specific images.

# Prerequisits

## Install OpenCV and Google benchmark without sudo

### Install OpenCV

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
cd opencv-master
mkdir -p build && cd build
mkdir $HOME/local
cmake -D CMAKE_INSTALL_PREFIX="$HOME/local" ..
cmake  ..
cmake --build . -- -j8
make install
// The following two commands need to be run every time you remotely connect to the lab computer
echo "export LD_LIBRARY_PATH=$HOME/local/lib" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=$HOME/CSC485C/pkgconfig" >> ~/.bashrc
```

Modify the prefix in opencv.pc under pkgconfig to 
> /home/**your_username**/local


### Install Google Benchmark
```
cd ~
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
```

