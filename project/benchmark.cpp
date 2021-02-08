#include <benchmark/benchmark.h> // Include Google Benchmark


// Define a new 
static void BM_Example(benchmark::State& state) {
// Initialize anything needed for the benchmark here
    int sum = 0;
    std::string x = "hello";
    for (auto _ : state) {
        std::string copy(x);
        sum += 1;
    }
    // A no-op to tell the compiler not to 
    benchmark::DoNotOptimize(sum);
}
// Register the function as a benchmark
BENCHMARK(BM_Example);

// Functions as a main function, no explicit main required
BENCHMARK_MAIN();