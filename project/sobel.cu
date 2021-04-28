#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdlib.h>

#include "lib/helpers.h"
#include "lib/base.cpp"
// #include "lib/simd.cpp"
// #include "lib/omp.cpp"

using namespace cv;
using namespace std;

#define BLOCK_X_DIM 32
#define BLOCK_Y_DIM 32

namespace GPGPU {
    string prefix = "GPGPU";
    
    namespace base {
        string impl = prefix + "_baseline";
            
        /*
        * Takes a 1d array for input and output, as well as the corresponding dimensions.
        * Calculates and updates the output array at position [y][x].
        */
        __global__ void sobel(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int r0 = y * p_cols;
            int r1 = (y + 1) * p_cols;
            int r2 = (y + 2) * p_cols;

            /**
            * The input and output arrays are offset by 1 pixel. 
            * So when computing output at (0,0) we are centered on the input pixel (1,1) due to padding
            */
            if (x < cols && y < rows) {
                float mag_x = input[r0 + x]
                            - input[r0 + x + 2]
                            + input[r1 + x] * 2
                            - input[r1 + x + 2] * 2
                            + input[r2 + x]
                            - input[r2 + x + 2];

                float mag_y = input[r0 + x]
                            + input[r0 + x + 1] * 2
                            + input[r0 + x + 2]
                            - input[r2 + x]
                            - input[r2 + x + 1] * 2
                            - input[r2 + x + 2];

                output[y*cols + x] = sqrt(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }

    namespace fastsqrt {
        string impl = prefix + "_fastsqrt";
        __global__ void sobel(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int r0 = y * p_cols;
            int r1 = (y + 1) * p_cols;
            int r2 = (y + 2) * p_cols;
    
            /**
             * The input and output arrays are offset by 1 pixel. 
             * So when computing output at (0,0) we are centered on the input pixel (1,1) due to padding
             */
            if (x < cols && y < rows) {
                float mag_x = input[r0 + x]
                            - input[r0 + x + 2]
                            + input[r1 + x] * 2
                            - input[r1 + x + 2] * 2
                            + input[r2 + x]
                            - input[r2 + x + 2];
    
                float mag_y = input[r0 + x]
                            + input[r0 + x + 1] * 2
                            + input[r0 + x + 2]
                            - input[r2 + x]
                            - input[r2 + x + 1] * 2
                            - input[r2 + x + 2];
    
                // Union struct for fast_sqrt
                union {
                    int i;
                    float x;
                } u;
                // Equivelant to a newtons approximation of sqrt.
                u.x = mag_x * mag_x + mag_y * mag_y;
                u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
                output[y*cols + x] = u.x;
            }
        }
    }

    namespace sharedmemory {
        string impl = prefix + "_sharedmemory";

        // NOTE: This kernel is specifically designed for a blocksize of 32x16!
        // This grants 1024B per block of shared memory

        /*
        * Takes a 1d array for input and output, as well as the corresponding dimensions.
        * Calculates and updates the output array at position [y][x].
        */
        __global__ void sobel(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols)
        {
            // given a padding
            __shared__ uint8_t tile[BLOCK_Y_DIM + 2][BLOCK_X_DIM + 2]; // 612 bytes/positions. 16 rows + apron, 32 cols + apron
            int x = threadIdx.x + blockIdx.x * blockDim.x; // output x & y
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            
            int tx = threadIdx.x + 1; // tile x
            int ty = threadIdx.y + 1; // tile y

            tile[ty][tx] = input[y * p_cols + x];

            if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == 31 || threadIdx.y == 5) { //first/last threads
                tile[threadIdx.y][threadIdx.x] = 0.0;
            }

            /* loads to shared memory are done, now do computation */
            __syncthreads();

            if (x < cols && y < rows) {
                float mag_x = tile[ty - 1][tx - 1]
                            + tile[ty    ][tx - 1] * 2
                            + tile[ty + 1][tx - 1]
                            - tile[ty - 1][tx + 1]
                            - tile[ty    ][tx + 1] * 2
                            - tile[ty + 1][tx + 1];

                float mag_y = tile[ty - 1][tx - 1]
                            + tile[ty - 1][tx] * 2
                            + tile[ty - 1][tx + 1]
                            - tile[ty + 1][tx - 1]
                            - tile[ty + 1][tx] * 2
                            - tile[ty + 1][tx + 1];

                output[y * cols + x] = sqrt(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }

    /**
     * Runner runs a set number of benchmarks, and returns the average runtime in microseconds (us)
     * the input image is always a CV_8UC1, any conversions need to be handled by the runner.
     */
    auto runner(const int trials, const Mat INPUT_IMAGE, const int version, bool write_output_image, const int block_x = 32, const int block_y = 16) {
        int rows = INPUT_IMAGE.rows;
        int cols = INPUT_IMAGE.cols;
        int N = rows * cols;

        /**
        * Preprocessing
        */
        Mat GPU_INPUT = preprocess(INPUT_IMAGE, 32, 32, 3);
        uint8_t* gpu_input = convertMat<uint8_t>(GPU_INPUT);
        size_t gpu_input_size = sizeof(uint8_t) * GPU_INPUT.rows * GPU_INPUT.cols;

        float *output = zeroed_array<float>(rows, cols);;

        /**
         * Default block size is manually found optimal for a RTX 3080. 
         * Cases 1-7 list attempted blocksizes using the baseline implementation.
         */
        dim3 pixels_per_block(block_x, block_y); // max 512, keep divisible by 32.
        dim3 num_blocks(ceil(cols / pixels_per_block.x), ceil(rows / pixels_per_block.y));
        /**
         * This logic allows switching implementation namespaces based on the "version" flag
         */
        typedef void (*funcp)(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols);
        funcp dev_func; 
        ostringstream implementation_stream;
        string blocksize;
        switch(version) {
        case 0: // base
            implementation_stream << base::impl;
            dev_func = base::sobel;
            break;
        case 1: // fastsqrt
            dev_func = fastsqrt::sobel;
            implementation_stream << fastsqrt::impl;
            break;
        case 2: // sharedmemory, 32x16
            pixels_per_block = dim3(BLOCK_X_DIM, BLOCK_Y_DIM);
            num_blocks = dim3(ceil(cols / pixels_per_block.x), ceil(rows / pixels_per_block.y));
            dev_func = sharedmemory::sobel;
            implementation_stream << sharedmemory::impl;
            break;
        default: // optimal
            dev_func = base::sobel;
            implementation_stream << base::impl;
            break;
        }

        implementation_stream << "_BS(" << unsigned(pixels_per_block.x) << ", " << unsigned(pixels_per_block.y) << ")";
        string implementation = implementation_stream.str();

        uint8_t *dev_input;
        float *dev_output;
        cudaMalloc((void**)&dev_input, gpu_input_size);
        cudaMalloc((void**)&dev_output, sizeof(float) * N);

        // ghost value and stopwatch
        auto sum = 0.0;
        int elapsed_time = 0; 
        for (int i = 0; i < trials; i++) {
            // Malloc GPU resources & prep inputs
            cudaMemset(dev_output, 0, (rows * cols));
            cudaMemcpy(dev_input, gpu_input, gpu_input_size, cudaMemcpyHostToDevice);

            auto start_time = chrono::system_clock::now();
            dev_func<<<num_blocks, pixels_per_block>>>(dev_input, dev_output, GPU_INPUT.rows, GPU_INPUT.cols, rows, cols);
            cudaDeviceSynchronize(); // waits for completion
            auto end_time = chrono::system_clock::now();
            elapsed_time += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

            cudaMemcpy(output, dev_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
            sum += output[1];
        }
        cudaFree(dev_output);
        cudaFree(dev_input);

        print_log(implementation, elapsed_time, trials, rows, cols);

        if (write_output_image == true) {
            Mat output_image = from_float_array(output, rows, cols);
            normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_32F);
            imwrite(OUTPUT_DIR + implementation + ".jpg", output_image);
            imwrite(OUTPUT_DIR + "GPGPU_INPUT" + ".jpg", GPU_INPUT);
        }

        // cleanup
        free(gpu_input);
        free(output);

        ResultData res;
        res.rows = rows;
        res.cols = cols;
        res.avg_time = elapsed_time / static_cast<float>(trials);;
        res.name = implementation;
        return res;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("USAGE: ./sobel NUM_TRIALS IMAGE_PATH [cpu|gpu] [test_case_number]\n");
        return 1;
    }
    auto num_trials = atoi(argv[1]);
    string file_name = argv[2];
    bool write_outputs = true;
    printf("SOBEL Test Case Runner\n");
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
    printf("GPGPU: %s, CUDA %d.%d\n", props.name, props.major, props.minor);
    
    Mat INPUT_IMAGE = imread(file_name, IMREAD_GRAYSCALE);
    printf("BEGINNING %d TRIALS ON %s(%dx%d):\n\n", num_trials, file_name.c_str(), INPUT_IMAGE.cols, INPUT_IMAGE.rows);

    /*
     * comment out test cases to ignore during a run, or use the switch case argument to run a specific from command line.
     */
    if (argc >= 5) {
        string t = argv[3];
        if (t.compare("gpu") == 0) {
            auto a = atoi(argv[4]);
            if (argc == 7) {
                GPGPU::runner(num_trials, INPUT_IMAGE, a, write_outputs, atoi(argv[5]), atoi(argv[6]));
            } else {
                GPGPU::runner(num_trials, INPUT_IMAGE, a, write_outputs);
            }
        } else if (t.compare("cpu") == 0) {
            auto a = atoi(argv[4]);
            BASE::runner(num_trials, INPUT_IMAGE, a, write_outputs);
        } else {
            printf("Specify gpu or cpu\nUSAGE: ./sobel NUM_TRIALS IMAGE_PATH [cpu|gpu] [test_case_number]\n");
            return 1;
        }
    } else {
        printf("USAGE: ./sobel NUM_TRIALS IMAGE_PATH [cpu|gpu] [test_case_number]\n");
        printf("Not enough args, running presets:\n");
        auto gpu_base_1_1 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 1, 1);
        auto gpu_base_32_1 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 1);
        auto gpu_base_32_2 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 2);
        auto gpu_base_32_4 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 4);
        auto gpu_base_32_8 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 8);
        auto gpu_base_32_16 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 16);
        auto gpu_base_32_32 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 32, 32);
        auto gpu_base_22_22 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 22, 22);
        auto gpu_base_23_22 = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs, 23, 22);
        auto gpu_fastsqrt = GPGPU::runner(num_trials, INPUT_IMAGE, 1, write_outputs);
        auto gpu_base_sharedmem = GPGPU::runner(num_trials, INPUT_IMAGE, 2, write_outputs);
        auto base_sqrt = BASE::runner(num_trials, INPUT_IMAGE, 0, write_outputs);
        auto base_fastsqrt = BASE::runner(num_trials, INPUT_IMAGE, 1, write_outputs);
    }

    // printf("\n\nSpeedups:\n");
    // show_speedup(gpu_base, gpu_fastsqrt);
    // show_speedup(base_sqrt, base_fastsqrt);
    // show_speedup(base_fastsqrt, gpu_fastsqrt);

}