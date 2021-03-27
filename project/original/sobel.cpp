#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <benchmark/benchmark.h>

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
    First imlementation that uses opencv Mat objects and uses opencv's implementation of normalization.

    Justification: 
    Based on:
*/
static void BENCH_SobelOriginalMatImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Convolution kernels
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        Mat output_image(n_rows, n_cols, CV_32F);
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_image.at<float>(r, c) * g_x[0][0] +
                              padded_image.at<float>(r, c + 1) * g_x[0][1] +
                              padded_image.at<float>(r, c + 2) * g_x[0][2] +
                              padded_image.at<float>(r + 1, c) * g_x[1][0] +
                              padded_image.at<float>(r + 1, c + 1) * g_x[1][1] +
                              padded_image.at<float>(r + 1, c + 2) * g_x[1][2] +
                              padded_image.at<float>(r + 2, c) * g_x[2][0] +
                              padded_image.at<float>(r + 2, c + 1) * g_x[2][1] +
                              padded_image.at<float>(r + 2, c + 2) * g_x[2][2];

                float mag_y = padded_image.at<float>(r, c) * g_y[0][0] +
                              padded_image.at<float>(r, c + 1) * g_y[0][1] +
                              padded_image.at<float>(r, c + 2) * g_y[0][2] +
                              padded_image.at<float>(r + 1, c) * g_y[1][0] +
                              padded_image.at<float>(r + 1, c + 1) * g_y[1][1] +
                              padded_image.at<float>(r + 1, c + 2) * g_y[1][2] +
                              padded_image.at<float>(r + 2, c) * g_y[2][0] +
                              padded_image.at<float>(r + 2, c + 1) * g_y[2][1] +
                              padded_image.at<float>(r + 2, c + 2) * g_y[2][2];

                output_image.at<float>(r, c) = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Use opencv's normalization function
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);

        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Use Mat objects and implement our own normalization.

    Justification: opencv's normalization is highly optimized. We consider normalization part of our benchmarking.
    Based on: BENCH_SobelOriginalMatImplementation

    **This is our starting implementation
*/
static void BENCH_SobelOriginalNormalizationImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Convolution kernels
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        Mat output_image(n_rows, n_cols, CV_32F);
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_image.at<float>(r, c) * g_x[0][0] +
                              padded_image.at<float>(r, c + 1) * g_x[0][1] +
                              padded_image.at<float>(r, c + 2) * g_x[0][2] +
                              padded_image.at<float>(r + 1, c) * g_x[1][0] +
                              padded_image.at<float>(r + 1, c + 1) * g_x[1][1] +
                              padded_image.at<float>(r + 1, c + 2) * g_x[1][2] +
                              padded_image.at<float>(r + 2, c) * g_x[2][0] +
                              padded_image.at<float>(r + 2, c + 1) * g_x[2][1] +
                              padded_image.at<float>(r + 2, c + 2) * g_x[2][2];

                float mag_y = padded_image.at<float>(r, c) * g_y[0][0] +
                              padded_image.at<float>(r, c + 1) * g_y[0][1] +
                              padded_image.at<float>(r, c + 2) * g_y[0][2] +
                              padded_image.at<float>(r + 1, c) * g_y[1][0] +
                              padded_image.at<float>(r + 1, c + 1) * g_y[1][1] +
                              padded_image.at<float>(r + 1, c + 2) * g_y[1][2] +
                              padded_image.at<float>(r + 2, c) * g_y[2][0] +
                              padded_image.at<float>(r + 2, c + 1) * g_y[2][1] +
                              padded_image.at<float>(r + 2, c + 2) * g_y[2][2];

                output_image.at<float>(r, c) = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_image.at<float>(r, c) > max)
                {
                    max = output_image.at<float>(r, c);
                }
                if (output_image.at<float>(r, c) < min)
                {
                    min = output_image.at<float>(r, c);
                }
            }
        }

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_image.at<float>(r, c) = (output_image.at<float>(r, c) - min) * (255) / (max - min);
            }
        }

        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Change datastructure from Mat to float arrays. Uses opencv normalization

    Justification: Arrays should be faster than Mat objects due to less overhead.
    Based on: BENCH_SobelOriginalNormalizationImplementation
*/
static void BENCH_SobelArrayImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Convolution kernels
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Convert padded image from cv::Mat to array
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_array[r][c] * g_x[0][0] +
                              padded_array[r][c + 1] * g_x[0][1] +
                              padded_array[r][c + 2] * g_x[0][2] +
                              padded_array[r + 1][c] * g_x[1][0] +
                              padded_array[r + 1][c + 1] * g_x[1][1] +
                              padded_array[r + 1][c + 2] * g_x[1][2] +
                              padded_array[r + 2][c] * g_x[2][0] +
                              padded_array[r + 2][c + 1] * g_x[2][1] +
                              padded_array[r + 2][c + 2] * g_x[2][2];

                float mag_y = padded_array[r][c] * g_y[0][0] +
                              padded_array[r][c + 1] * g_y[0][1] +
                              padded_array[r][c + 2] * g_y[0][2] +
                              padded_array[r + 1][c] * g_y[1][0] +
                              padded_array[r + 1][c + 1] * g_y[1][1] +
                              padded_array[r + 1][c + 2] * g_y[1][2] +
                              padded_array[r + 2][c] * g_y[2][0] +
                              padded_array[r + 2][c + 1] * g_y[2][1] +
                              padded_array[r + 2][c + 2] * g_y[2][2];

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_image);
    }
}

/*
    Change to use 2d vectors for comparing to array implementation

    Justification: Test to see if arrays or std::vectors are faster
    Based on: BENCH_SobelOriginalNormalizationImplementation
*/
static void BENCH_SobelVectorImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Convolution kernels
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    // uint8_t padded_array[padded_image.rows][padded_image.cols];
    std::vector<std::vector<float>> padded_array(padded_image.rows, std::vector<float>(padded_image.cols, 0));
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        // float output_array[n_rows][n_cols];
        std::vector<std::vector<float>> output_array(n_rows, std::vector<float>(n_cols, 0));
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_array[r][c] * g_x[0][0] +
                              padded_array[r][c + 1] * g_x[0][1] +
                              padded_array[r][c + 2] * g_x[0][2] +
                              padded_array[r + 1][c] * g_x[1][0] +
                              padded_array[r + 1][c + 1] * g_x[1][1] +
                              padded_array[r + 1][c + 2] * g_x[1][2] +
                              padded_array[r + 2][c] * g_x[2][0] +
                              padded_array[r + 2][c + 1] * g_x[2][1] +
                              padded_array[r + 2][c + 2] * g_x[2][2];

                float mag_y = padded_array[r][c] * g_y[0][0] +
                              padded_array[r][c + 1] * g_y[0][1] +
                              padded_array[r][c + 2] * g_y[0][2] +
                              padded_array[r + 1][c] * g_y[1][0] +
                              padded_array[r + 1][c + 1] * g_y[1][1] +
                              padded_array[r + 1][c + 2] * g_y[1][2] +
                              padded_array[r + 2][c] * g_y[2][0] +
                              padded_array[r + 2][c + 1] * g_y[2][1] +
                              padded_array[r + 2][c + 2] * g_y[2][2];

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Hardcode the kernel values into the multiplication. Uses opencv normalization.

    Justification: Avoids having to look up values from kernel arrays.
    Based on: BENCH_SobelArrayImplementation
*/
static void BENCH_SobelHardcodeKernelsImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
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

        // Convert array to Mat, base on documentation, this function only create a header that points to the data
        Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

        // Use opencv's normalization function
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);
        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Hardcode the kernal values into the multiplication. Uses our normalization.

    Justification: Since we are using our own normalization, need to see how hardcoding kernels performs
    Based on: BENCH_SobelHardcodeKernelsImplementation, BENCH_SobelOriginalNormalizationImplementation
*/
static void BENCH_SobelHardcodeKernelsNormalizationImplementation(benchmark::State &state)
{
    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread(FILENAME, IMREAD_GRAYSCALE);

    Mat resized_image = resizeImage(input_image, state.range(0));

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    resized_image.convertTo(image, CV_32F);

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
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

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }
        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Input image is uint8 type. Hardcode the kernal values into the multiplication. Uses our normalization.

    Justification: greyscale images in range of [0,255]. Should allow more values in cache
    Based on: BENCH_SobelHardcodeKernelsNormalizationImplementation
*/
static void BENCH_Sobeluint8InputImplementation(benchmark::State &state)
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

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Uses the BENCH_Sobeluint8InputImplementation as a base, but transformes the input into a 1D array

    Justification: this should see biggest speedup in smaller images as multiple lines will sit in the cacheline
    BUT speculative prefetching might pick up on the jumps
    Base on: BENCH_Sobeluint8InputImplementation
*/
static void BENCH_Sobeluint8Input1DArray(benchmark::State &state)
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

    int p_cols = padded_image.cols;
    int p_rows = padded_image.rows;
    // Use 1D array to store the image value
    uint8_t padded_array[p_cols * p_rows];
    for (int r = 0; r < p_rows; ++r)
    {
        for (int c = 0; c < p_cols; ++c)
        {
            padded_array[r * p_cols + c] = padded_image.at<float>(r, c);
        }
    }

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_array[r * p_cols + c] * 1 +
                              padded_array[r * p_cols + c + 2] * -1 +
                              padded_array[(r + 1) * p_cols + c] * 2 +
                              padded_array[(r + 1) * p_cols + c + 2] * -2 +
                              padded_array[(r + 2) * p_cols + c] * 1 +
                              padded_array[(r + 2) * p_cols + c + 2] * -1;

                float mag_y = padded_array[r * p_cols + c] * 1 +
                              padded_array[r * p_cols + c + 1] * 2 +
                              padded_array[r * p_cols + c + 2] * 1 +
                              padded_array[(r + 2) * p_cols + c] * -1 +
                              padded_array[(r + 2) * p_cols + c + 1] * -2 +
                              padded_array[(r + 2) * p_cols + c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2)); // TODO: pow slow, use mag_y*mag_y. Likewise, use
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Separate kernel multiplication by row.

    Justification: Separating kernel multiplication by rows should avoid having to read rows in multiple times during one calculation
    Based on: BENCH_Sobeluint8InputImplementation
*/
static void BENCH_Sobeluint8MathOperationReorder(benchmark::State &state)
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
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {

                float mag_y = padded_array[r][c] * 1 +
                              padded_array[r][c + 1] * 2 +
                              padded_array[r][c + 2] * 1;

                float mag_x = padded_array[r][c] * 1 +
                              padded_array[r][c + 2] * -1;

                mag_x += padded_array[r + 1][c] * 2 +
                         padded_array[r + 1][c + 2] * -2;

                mag_x += padded_array[r + 2][c] * 1 +
                         padded_array[r + 2][c + 2] * -1;

                mag_y += padded_array[r + 2][c] * -1 +
                         padded_array[r + 2][c + 1] * -2 +
                         padded_array[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Combines 1d array input and reordering math operations

    Justification: Reordering math operations may have a larger effect on 1d array
    Based on: BENCH_Sobeluint8Input1DArray, BENCH_Sobeluint8MathOperationReorder
*/
static void BENCH_Sobeluint8Input1DArrayMathOperationReorder(benchmark::State &state)
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

    int p_cols = padded_image.cols;
    int p_rows = padded_image.rows;
    // Use 1D array to store the image value
    uint8_t padded_array[p_cols * p_rows];
    for (int r = 0; r < p_rows; ++r)
    {
        for (int c = 0; c < p_cols; ++c)
        {
            padded_array[r * p_cols + c] = padded_image.at<float>(r, c);
        }
    }

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {

                float mag_x = padded_array[r * p_cols + c] * 1 +
                              padded_array[r * p_cols + c + 2] * -1;

                float mag_y = padded_array[r * p_cols + c] * 1 +
                              padded_array[r * p_cols + c + 1] * 2 +
                              padded_array[r * p_cols + c + 2] * 1;

                mag_x += padded_array[(r + 1) * p_cols + c] * 2 +
                         padded_array[(r + 1) * p_cols + c + 2] * -2;

                mag_x += padded_array[(r + 2) * p_cols + c] * 1 +
                         padded_array[(r + 2) * p_cols + c + 2] * -1;
                mag_y += padded_array[(r + 2) * p_cols + c] * -1 +
                         padded_array[(r + 2) * p_cols + c + 1] * -2 +
                         padded_array[(r + 2) * p_cols + c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2)); // TODO: pow slow, use mag_y*mag_y. Likewise, use
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Input image is uint8 type as above, but does basic loop unrolling on rows of array.

    Justification:
    Based on: BENCH_Sobeluint8Input1DArray

    ** maybe im dumb, but where is the loop unrolling??
*/
static void BENCH_Sobeluint8InputImplementationLoopUnroll(benchmark::State &state)
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
        for (int j = 0; j < padded_image.cols; j += 2)
        {
            padded_array[i][j] = padded_image.at<float>(i, j);
            padded_array[i][j + 1] = padded_image.at<float>(i, j + 1);
        }

    // Benchmark includes convolution and normalization back to [0,255]
    for (auto _ : state)
    {
        float output_array[n_rows][n_cols];
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                int mag_x = padded_array[r][c] * 1 +
                            padded_array[r][c + 2] * -1 +
                            padded_array[r + 1][c] * 2 +
                            padded_array[r + 1][c + 2] * -2 +
                            padded_array[r + 2][c] * 1 +
                            padded_array[r + 2][c + 2] * -1;

                int mag_y = padded_array[r][c] * 1 +
                            padded_array[r][c + 1] * 2 +
                            padded_array[r][c + 2] * 1 +
                            padded_array[r + 2][c] * -1 +
                            padded_array[r + 2][c + 1] * -2 +
                            padded_array[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
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

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}
*/

/*
    Change mag_x and mag_y to be integers

    Justification: Input image and kernel values are both integers. May be faster to write to an integer instead of float.
    Based on: BENCH_Sobeluint8InputImplementation
*/
static void BENCH_SobelMagIntsImplementation(benchmark::State &state)
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
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                int32_t mag_x = padded_array[r][c] * 1 +
                                padded_array[r][c + 2] * -1 +
                                padded_array[r + 1][c] * 2 +
                                padded_array[r + 1][c + 2] * -2 +
                                padded_array[r + 2][c] * 1 +
                                padded_array[r + 2][c + 2] * -1;

                int32_t mag_y = padded_array[r][c] * 1 +
                                padded_array[r][c + 1] * 2 +
                                padded_array[r][c + 2] * 1 +
                                padded_array[r + 2][c] * -1 +
                                padded_array[r + 2][c + 1] * -2 +
                                padded_array[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    Use x*x instad of pow(x,2)

    Justification: C++ pow function is slower than explicit x*x when squaring
    Based on: BENCH_Sobeluint8InputImplementation
*/
static void BENCH_SobelHardcodePowImplementation(benchmark::State &state)
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
                output_array[r][c] = sqrt((mag_x * mag_x) + (mag_y * mag_y));
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        float max = -INFINITY;
        float min = INFINITY;
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
    combine the determination of max and min for normalization into convolution loops (was previously separated)

    Justification: combining max/min determination cuts down on looping overhead
    Based on: BENCH_Sobeluint8InputImplementation
*/
static void BENCH_SobelCombineMaxMinImplementation(benchmark::State &state)
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
                output_array[r][c] = sqrt((mag_x * mag_x) + (mag_y * mag_y));

                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin

        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

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

                if (output_array[r][c] > max)
                {
                    max = output_array[r][c];
                }
                if (output_array[r][c] < min)
                {
                    min = output_array[r][c];
                }
            }
        }

        // Implement our own normalization
        // For each pixel I, I_norm = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
        for (int r = 0; r < n_rows; r++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                output_array[r][c] = (output_array[r][c] - min) * (255) / (max - min);
            }
        }

        // benchmark::DoNotOptimize(output_array);
    }
}

/*
Can only pass in arguments to benchmark function that are integers.
To run on different sized images, will pass in how small to shrink the image.
Eg. passing in 0 will resize image by 1/2^0 on each axis, 1 will resize to 1/2^1 etc.
*/
BENCHMARK(BENCH_SobelOriginalMatImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelOriginalNormalizationImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelArrayImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelVectorImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodeKernelsImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodeKernelsNormalizationImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8InputImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8Input1DArray)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8MathOperationReorder)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8Input1DArrayMathOperationReorder)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8InputImplementationLoopUnroll)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelMagIntsImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodePowImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelCombineMaxMinImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodePowAndSqrtImplementation)->DenseRange(0, DENSERANGEEND, 1);

// Calls and runs the benchmark program
BENCHMARK_MAIN();
