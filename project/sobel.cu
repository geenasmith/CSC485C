#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "lib/helpers.h"
#include "lib/base.cpp"
// #include "lib/simd.cpp"
// #include "lib/omp.cpp"

using namespace cv;
using namespace std;

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

    /**
     * Runner runs a set number of benchmarks, and returns the average runtime in microseconds (us)
     * the input image is always a CV_8UC1, any conversions need to be handled by the runner.
     */
    auto runner(const int trials, const Mat INPUT_IMAGE, const int version, bool write_output_image) {
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
        
        dim3 pixels_per_block(32, 16); // max 512, keep divisible by 32.
        dim3 num_blocks(ceil(cols / pixels_per_block.x), ceil(rows / pixels_per_block.y));
        /**
         * This logic allows switching implementation namespaces based on the "version" flag
         */
        typedef void (*funcp)(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols);
        funcp dev_func; 
        string implementation;
        switch(version) {
            case 0: // base
            dev_func = base::sobel;
            implementation = base::impl;
            break;
            case 1: // fastsqrt
            dev_func = fastsqrt::sobel;
            implementation = fastsqrt::impl;
            break;
            default: // base
            dev_func = base::sobel;
            implementation = base::impl;
            break;
        }

        // ghost value and stopwatch
        auto sum = 0.0;
        int elapsed_time = 0; 
        for (int i = 0; i < trials; i++) {
            // Malloc GPU resources & prep inputs
            uint8_t *dev_input;
            float *dev_output;
            cudaMalloc((void**)&dev_input, gpu_input_size);
            cudaMalloc((void**)&dev_output, sizeof(float) * N);
            cudaMemcpy(dev_input, gpu_input, gpu_input_size, cudaMemcpyHostToDevice);

            auto start_time = chrono::system_clock::now();
            dev_func<<<num_blocks, pixels_per_block>>>(dev_input, dev_output, GPU_INPUT.rows, GPU_INPUT.cols, rows, cols);
            cudaDeviceSynchronize(); // waits for completion
            auto end_time = chrono::system_clock::now();
            elapsed_time += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

            cudaMemcpy(output, dev_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
            cudaFree(dev_output);
            cudaFree(dev_input);
            
            sum += output[1];
        }

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
    string file_name = argv[1];

    printf("SOBEL Test Case Runner\n");
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
    printf("GPGPU: %s, CUDA %d.%d\n", props.name, props.major, props.minor);
    
    Mat INPUT_IMAGE = imread(file_name, IMREAD_GRAYSCALE);

    bool write_outputs = true;
    int num_trials = 100;

    printf("BEGINNING %d TRIALS ON %s(%dx%d):\n\n", num_trials, file_name.c_str(), INPUT_IMAGE.cols, INPUT_IMAGE.rows);

    /*
     * comment out test cases to ignore
     */
    auto gpu_base = GPGPU::runner(num_trials, INPUT_IMAGE, 0, write_outputs);
    auto gpu_fastsqrt = GPGPU::runner(num_trials, INPUT_IMAGE, 1, write_outputs);
    auto base_sqrt = BASE::runner(num_trials, INPUT_IMAGE, 0, write_outputs);
    auto base_fastsqrt = BASE::runner(num_trials, INPUT_IMAGE, 1, write_outputs);

    printf("\n\nSpeedups:\n");
    show_speedup(gpu_base, gpu_fastsqrt);
    show_speedup(base_sqrt, base_fastsqrt);
    show_speedup(base_fastsqrt, gpu_fastsqrt);

}