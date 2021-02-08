# CSC485C
Project and course work


# Installing google benchmark
1. Ensure cmake is available:
```bash
# macos:
brew install cmake
# linux:
sudo apt install cmake
# windows:
https://cmake.org/download/
```
2. outside of this repo, run this for linux/mac:
```bash
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
```

3. Install globally:
```bash
sudo cmake --build "build" --config Release --target install
```