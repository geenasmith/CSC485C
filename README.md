# CSC485C
Project and course work

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
cmake --build .
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

### Compile

Update the **path_to_benchmark** before use

```
g++ -std=c++11 project/sobel.cpp -o sobel `pkg-config --cflags --libs opencv` -isystem $HOME/benchmark/include -L$HOME/benchmark/build/src -lbenchmark -lpthread
```
