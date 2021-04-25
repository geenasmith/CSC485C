#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;


__global__ void image_darken(const float *img, float *out_img, const float mult, int rows, int cols) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // if (x < cols && y < rows) {
    //     out_img[y * cols + x] *= mult;
    // }
    out_img[0] = 1;
}

void printdata(float* data, int cols, int rm, int cm) {
    for (int r = 0; r < rm; r++) {
        for (int c = 0; c < cm; c++) {
            printf("%3.f ", data[c+r*cols]);
        }
        printf("\n");
    }
}

int main(){

    Mat img = imread("rgb1.jpg", IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F);

    auto N = img.rows * img.cols;

    float *dev_input, *dev_output;
    float *input, *output;

    input  = (float*)malloc(sizeof(float) * N);
    output = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++) {
            input[r*img.cols + c] = img.at<float>(r,c);
            output[r*img.cols + c] = 0;
        }

    // Allocate device memory
    cudaMalloc((void**)&dev_input, sizeof(float) * N);
    cudaMalloc((void**)&dev_output, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(dev_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    image_darken<<<1,1>>>(dev_input, dev_output, 0.5, img.rows, img.cols);
    
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    printf("OUTPUT_BEFORE_SAVE\n");
    printdata(output, img.cols, 32,32);


    // Transfer data back to host memory
    cudaMemcpy(output, dev_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("OUTPUT_AFTER_SAVE\n");
    printdata(output, img.cols, 32,32);

    // Verification
    auto diff = 0.0;
    auto sum = 0.0;
    for(int i = 0; i < N; i++) {
        diff += input[i] * 0.5 - output[i];
        sum += input[i];
    }
    Mat outimg(img.rows, img.cols, CV_32F);
        // Initialize host arrays
    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++) {
            outimg.at<float>(r,c) = output[r*img.cols + c];
        }

    imwrite("out.jpg", outimg);

    printf("variance %f\n", diff);
    printf("variance %f\n", diff/N);

    // Deallocate device memory
    cudaFree(dev_input);
    cudaFree(dev_output);

    // Deallocate host memory
    free(input); 
    free(output); 
}
