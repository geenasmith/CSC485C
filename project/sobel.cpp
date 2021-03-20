#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <benchmark/benchmark.h>
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX


using namespace cv;

std::string FILENAME = "images/rgb1.jpg";
int DENSERANGEEND = 0;

/*
    Apply a gaussian filter and pad the image with a 1px border of 0s
*/
Mat preprocessing(Mat image)
{
    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    copyMakeBorder(image, padded_image, 1, 1, 1, 1, BORDER_CONSTANT, 0);

    return padded_image;
}

/*
    Returns input_image resized by 1/(2<<resize_factor) along each dimension
    eg. resize_factor=1 will resize by 0.5
*/
Mat resizeImage(Mat input_image, int resize_factor)
{
    double resize_amount = 1.0 / (1 << resize_factor);
    Mat resized_image;
    resize(input_image, resized_image, Size(), resize_amount, resize_amount);
    return resized_image;
}

/*
    Reference: http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
    Algorithm: Log base 2 approximation and Newton's Method
*/
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

/*
    Input image is uint8 type. Hardcode the kernal values into the multiplication. Uses our normalization.
*/
/*
static void BENCH_Sobeluint8InputImplementationDiagTiling(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_8UC1 (uint8)
    Mat image;
    input_image.convertTo(image, CV_8UC1);

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    uint8_t padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        auto const x_tile_size = 4u;
        auto const y_tile_size = 4u;

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
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
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}
*/


/*
    Use sqrt() found in http://ilab.usc.edu/wiki/index.php/Fast_Square_Root

    Justification: built in sqrt function may not be the most efficient
    Based on: BENCH_SobelCombineMaxMinImplementation
*/
static void BENCH_SobelHardcodePowAndSqrtImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_8UC1 (uint8)
    Mat image;
    resized_image.convertTo(image, CV_8UC1);

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    uint8_t padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        float max = -INFINITY;
        float min = INFINITY;
    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        
        float output_array[n_rows][n_cols];
        auto const x_tile_size = 4u;
        auto const y_tile_size = 4u;

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {

                // float mag_x = padded_array[r][c] * 1 +
                //               padded_array[r][c + 2] * -1 +
                //               padded_array[r + 1][c] * 2 +
                //               padded_array[r + 1][c + 2] * -2 +
                //               padded_array[r + 2][c] * 1 +
                //               padded_array[r + 2][c + 2] * -1;

                // float mag_y = padded_array[r][c] * 1 +
                //               padded_array[r][c + 1] * 2 +
                //               padded_array[r][c + 2] * 1 +
                //               padded_array[r + 2][c] * -1 +
                //               padded_array[r + 2][c + 1] * -2 +
                //               padded_array[r + 2][c + 2] * -1;

                

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt_impl((mag_x * mag_x) + (mag_y * mag_y));

            }
        }

        // Normalization is not done in this benchmark

        benchmark::DoNotOptimize(output_array[0][0]);
    }
}

/*
    Use sqrt() found in http://ilab.usc.edu/wiki/index.php/Fast_Square_Root

    Justification: built in sqrt function may not be the most efficient
    Based on: BENCH_SobelCombineMaxMinImplementation
*/
static void BENCH_SobelTiling(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_8UC1 (uint8)
    Mat image;
    resized_image.convertTo(image, CV_8UC1);

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    uint8_t padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes sobel operation only, not normalization
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
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
                output_array[r][c] = sqrt_impl((mag_x * mag_x) + (mag_y * mag_y));
            }
        }

        // Normalization is not done in this benchmark

        benchmark::DoNotOptimize(output_array[0][0]);
    }
}


/*
Can only pass in arguments to benchmark function that are integers.
To run on different sized images, will pass in how small to shrink the image.
Eg. passing in 0 will resize image by 1/2^0 on each axis, 1 will resize to 1/2^1 etc.
*/
BENCHMARK(BENCH_SobelHardcodePowAndSqrtImplementation)->DenseRange(0, DENSERANGEEND, 1);

// Calls and runs the benchmark program
BENCHMARK_MAIN();
