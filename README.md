# CSC485C
Project and course work

## Install OpenCV and Google benchmark without sudo

### Install OpenCV

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
mkdir -p build && cd build
cmake -D CMAKE_INSTALL_PREFIX='/home/**your_username**/local' ..
cmake  ../opencv-master
cmake --build .
export LD_LIBRARY_PATH=/home/**your_username**/local/lib
export PKG_CONFIG_PATH=**path_to_CSC485C_repo**/pkgconfig
```

Modify the prefix in opencv.pc under pkgconfig to 
> /home/**your_username**/local


### Install Google Benchmark
```
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
```

### Compile

Update the **path_to_benchmark** before use

```
g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv` -isystem **path_to_benchmark**/include -L**path_to_benchmark**/build/src -lbenchmark -lpthread
```
