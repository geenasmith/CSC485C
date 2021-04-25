#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "helpers.h"


// Mat postprocess() {
//     return Mat;
// };

namespace GPGPU {
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
    __global__ void sobel_fastsqrt(uint8_t* input, float* output, int p_rows, int p_cols, int rows, int cols)
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
        union {
            int i;
            float x;
        } u;
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

            u.x = mag_x * mag_x + mag_y * mag_y;
            u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
            output[y*cols + x] = u.x;
        }
    }

}

int main(int argc, char **argv)
{
    // if (argc < 4) {
    //     printf("USAGE: ./sobel [image] [num_trials] [test_to_run] [subtest_if_any]");
    //     printf("0: all\t1: fast_uint8\t2: simd\t3: omp\t4: simt");
    //     return 1;
    // }
    string file_name = "../images/frac3.png";//argv[1];

    printf("SOBEL\n");
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
    printf("Using GPGPU: %s, CUDA %d.%d\n", props.name, props.major, props.minor);
    
    Mat INPUT_IMAGE = imread(file_name, IMREAD_GRAYSCALE); 
    Mat INPUT_IMAGE_AS_FLOAT; INPUT_IMAGE.convertTo(INPUT_IMAGE_AS_FLOAT, CV_32F);
    int rows = INPUT_IMAGE.rows;
    int cols = INPUT_IMAGE.cols;
    int N = rows * cols;



    // Mat SIMD_INPUT = preprocess(INPUT_IMAGE_AS_FLOAT, 6, 1, 3);
    // float* simd_input = convertMat<float>(SIMD_INPUT);
    // imwrite("simd_input.jpg", SIMD_INPUT);

    /**
     * GPGPU Iterations
     */
    Mat GPU_INPUT = preprocess(INPUT_IMAGE, 32, 32, 3);
    uint8_t* gpu_input = convertMat<uint8_t>(GPU_INPUT);
    size_t gpu_input_size = sizeof(uint8_t) * GPU_INPUT.rows * GPU_INPUT.cols;
    imwrite("gpgpu_input.jpg", GPU_INPUT);

    uint8_t *dev_input;
    float *dev_output;
    float *output = zeroed_array<float>(rows, cols);
    
    cudaMalloc((void**)&dev_input, gpu_input_size);
    cudaMalloc((void**)&dev_output, sizeof(float) * N);
    cudaMemcpy(dev_input, gpu_input, gpu_input_size, cudaMemcpyHostToDevice);

    dim3 pixels_per_block(32, 16); // max 512, keep divisible by 32.
    dim3 num_blocks(ceil(cols / pixels_per_block.x), ceil(rows / pixels_per_block.y));
    
    auto start_time = std::chrono::system_clock::now();
    GPGPU::sobel_fastsqrt<<<num_blocks, pixels_per_block>>>(dev_input, dev_output, GPU_INPUT.rows, GPU_INPUT.cols, rows, cols);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    auto end_time = std::chrono::system_clock::now();
    output_time("GPGPU", start_time, end_time, 1u, rows, cols);

    

    cudaMemcpy(output, dev_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(dev_input);
    cudaFree(dev_output);

    Mat output_image = from_float_array(output, rows, cols);
    normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_32F);

    imwrite("out.jpg",output_image);
    

    // uint8_t *input;
    // float *output;
 
    // input  = (uint8_t*)malloc(sizeof(uint8_t) * N);
    // output = (float*)malloc(sizeof(float) * N);

    imwrite("orig.jpg", INPUT_IMAGE);

    waitKey(0);
}