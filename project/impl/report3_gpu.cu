#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "../sobel.h"

using namespace cv;

namespace report3 {
std::string base = "report3";

namespace uint8FastSqrt
{
    std::string implementation = base + "_" + "uint8FastSqrt";

    float sqrt_impl(const float x)
    {
        union {
            int i;
            float x;
        } u;
        
        u.x = x;
        u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
        return u.x;
    }

    void sobel_fastsqrt(uint8_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
    {
        for (int r = 0; r < orig_n_rows; r++)
        {
            for (int c = 0; c < orig_n_cols; c++)
            {
                float mag_x = padded_array[r][c] * 1 +
                            padded_array[r][c + 2] * -1 +
                            padded_array[r + 1][c] * 2 +
                            padded_array[r + 1][c + 2] * -2 +
                            padded_array[r + 2][c] * 1 +
                            padded_array[r + 2][c + 2] * -1;

                float mag_y = padded_array[r][c] * 1 +
                            padded_array[r][c + 1] * 2 +
                            padded_array[r][c + 2] * 1 +
                            padded_array[r + 2][c] * -1 +
                            padded_array[r + 2][c + 1] * -2 +
                            padded_array[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }
} // namespace uint8Array

namespace GPUBaseline
{
    std::string implementation = base + "_" + "GPUBaseline";
    Mat postprocessing(float *output_array, int orig_n_rows, int orig_n_cols)
    {
        Mat output_image = Mat(orig_n_rows, orig_n_cols, CV_32F);
        for (int i = 0; i < orig_n_rows; ++i)
            for (int j = 0; j < orig_n_cols; ++j)
                output_image.at<float>(i, j) = output_array[i*orig_n_cols+j];

        // Use opencv's normalization function
        cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

        return output_image;
    }

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

}
void output_time(std::string name, std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end, uint trials, int h, int w) {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << name << " with " << trials << " trials" << std::endl;
    std::cout << "Average time per run: " << elapsed_time.count() / static_cast<float>(trials) << " us" << std::endl;
    std::cout << "Input resolution " << w << "x" << h << std::endl << std::endl;
}
/**
 * Host code and preprocessor
 */
int gpu_runner(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {

    #undef VERS
    #define VERS report3::GPUBaseline

    auto sum = 0;
    auto image = preprocessing_uint8_gpu(file_name, 32);

    const int blocksize = 512; // mult of 32
    int num_blocks = ceil(image.orig_rows * image.orig_cols / blocksize);

    // for (auto i = 0u; i < benchmark_trials; ++i)
    // {
    /* 0 the output array */
    for (int i = 0; i < image.orig_rows; i++)
        for (int j = 0; j < image.orig_cols; j++)
            image.output[i * image.orig_cols + j] = 0.0;

    uint8_t* dev_in;
    float* dev_out;
    cudaMalloc((void**) &dev_in, sizeof(image.input));
    cudaMalloc((void**) &dev_out, sizeof(image.output));
    cudaMemcpy(dev_in, image.input, sizeof(image.input), cudaMemcpyHostToDevice);

    VERS::sobel<<< num_blocks, blocksize >>>(dev_in, dev_out, image.padded_rows, image.padded_cols, image.orig_rows, image.orig_cols);

    cudaMemcpy( image.output, dev_out, sizeof(image.output), cudaMemcpyDeviceToHost );

    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 32; i++) {
            std::cout << unsigned(image.input[j*image.orig_cols+i]) << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(dev_in);
    cudaFree(dev_out);

    sum += image.output[1];
    // }
    
    if(display) imwrite("out.jpg", report3::GPUBaseline::postprocessing(image.output, image.orig_rows, image.orig_cols));
    std::cout << "END ITER: " << sum << std::endl;
    return sum;
}



int main(int argc, char **argv)
{
    auto sum = 0;
    auto const benchmark_trials = 10u;
    bool const display_outputs = false;

    gpu_runner("rgb1.jpg", benchmark_trials, true);
    waitKey(0);
}