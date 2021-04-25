#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>

// #include "../sobel.h"

using namespace cv;

namespace report3 {
std::string base = "report3";

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
            // output[y*cols + x] = (float)input[y * cols + x] * 0.5;
            // printf("%f", input[y*cols + x]);
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
    Mat image = imread(file_name, IMREAD_GRAYSCALE);
    Mat padded;
    int rows = image.rows;
    int cols = image.cols;

    printf("Test");
    /** CONFIGURATION **/
    /**
     * pixelsPerBlock is specifying 32x16 block of pixels:
     * - 2048 bytes float
     * - 32*16 = 512 -> max block size
     */
    dim3 pixels_per_block(32,16);
    dim3 num_blocks(ceil(cols / pixels_per_block.x), ceil(rows / pixels_per_block.y));
    
    /**
     * PREPROCESSING
     */
    image.convertTo(image, CV_8UC1);
    GaussianBlur(image, image,Size(3, 3), 0, 0);
    
    /**
     * Right side padding must be at LEAST 1, and also make the total image be divisible by block_size
     * Bottom side padding must follow the same
     */
    int N = rows * cols;
    int xmod = (cols % 32 == 0) ? 0 : cols % 32;
    int ymod = (rows % 32 == 0) ? 0 : rows % 32;
    copyMakeBorder(image, padded, 1, ymod+1, 1, xmod+1, BORDER_CONSTANT, 0);

    uint8_t *input;
    float *output;
 
    input  = (uint8_t*)malloc(sizeof(uint8_t) * N);
    output = (float*)malloc(sizeof(float) * N);

    // Initialize the input array
    for (int r = 0; r < padded.rows; r++)
        for (int c = 0; c < padded.cols; c++) 
            input[r * padded.cols + c] = padded.at<uint8_t>(r, c);
    
    auto sum = 0;
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        /* 0 the output array */
        for (int r = 0; r < cols; r++)
            for (int c = 0; c < rows; c++) 
                output[r * cols + c] = 0.0;

        uint8_t *dev_input;
        float *dev_output;
        cudaMalloc((void**)&dev_input, sizeof(uint8_t) * N);
        cudaMalloc((void**)&dev_output, sizeof(float) * N);
        cudaMemcpy(dev_input, input, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);

        VERS::sobel<<< 1, pixels_per_block >>>(dev_input, dev_output, padded.rows, padded.cols, rows, cols);

        cudaMemcpy( output, dev_output, sizeof(float) * N, cudaMemcpyDeviceToHost );

        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < 32; i++) {
                std::cout << output[j*cols+i] << " ";
            }
            std::cout << std::endl;
        }

        cudaFree(dev_input);
        cudaFree(dev_output);

        sum += output[1];
    }
    
    if(display) imshow("out.jpg", report3::GPUBaseline::postprocessing(output, rows, cols));
    std::cout << "END ITER: " << sum << std::endl;
    return sum;
}



int main(int argc, char **argv)
{
    printf("Test");
    auto sum = 0;
    auto const benchmark_trials = 10u;
    bool const display_outputs = false;

    gpu_runner("rgb1.jpg", 1, true);
    waitKey(0);
}