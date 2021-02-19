#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <benchmark/benchmark.h>

using namespace cv;

/*
compile (cause pkg-config is annoying): g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv`
run: ./sobel <input image>
*/

Mat preprocessing(Mat image)
{
    /*
    Apply preprocessing:
      - apply gaussian filter to smooth image (removes noise that might be considered an edge)
      - pad the image with a 1px border of 0s for the sobel filter
    */

    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    copyMakeBorder(image, padded_image, 1, 1, 1, 1, BORDER_CONSTANT, 0);

    return padded_image;
}

/*
    Initial benchmarking implementation
      - Uses opencv Mat objects and opencv normalization
*/
static void BENCH_SobelOriginalMatImplementation(benchmark::State &state)
{
    // SETUP BENCHMARK

    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    input_image.convertTo(image, CV_32F); // TODO: opt 1 - change to uint8

    /**
     * Convolution Kernel
     **/
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

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

        // normalize to 0-255
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);
        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Initial benchmarking implementation with our own normalization implemented (not using opencv's)
*/
static void BENCH_SobelOriginalNormalizationImplementation(benchmark::State &state)
{
    // SETUP BENCHMARK

    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    input_image.convertTo(image, CV_32F); // TODO: opt 1 - change to uint8

    /**
     * Convolution Kernel
     **/
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

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

        // normalize to 0-255
        // I_N = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
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
*/
static void BENCH_SobelArrayImplementation(benchmark::State &state)
{
    // SETUP BENCHMARK

    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    input_image.convertTo(image, CV_32F); // TODO: opt 1 - change to uint8

    /**
     * Convolution Kernel
     **/
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

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

        // Convert array to Mat, base on documentation, this function only create a header that points to the data
        Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

        // normalize to 0-255
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);
        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Hardcode the kernel values into the multiplication. Uses opencv normalization.
*/
static void BENCH_SobelHardcodeKernelsImplementation(benchmark::State &state)
{
    // SETUP BENCHMARK

    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    input_image.convertTo(image, CV_32F); // TODO: opt 1 - change to uint8

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

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

        // normalize to 0-255
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);
        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Hardcode the kernal values into the multiplication. Uses our normalization.
*/
static void BENCH_SobelHardcodeKernelsNormalizationImplementation(benchmark::State &state)
{
    // SETUP BENCHMARK

    // read in image as grayscale OpenCV Mat Object
    Mat input_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    input_image.convertTo(image, CV_32F); // TODO: opt 1 - change to uint8

    // Filtered image definitions
    int n_rows = image.rows;
    int n_cols = image.cols;

    Mat padded_image = preprocessing(image);

    // Use array to store the image value
    float padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

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

        // normalize to 0-255
        // I_N = (I-Min) * (newMax-newMin) / (Max-Min) + newMin
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

BENCHMARK(BENCH_SobelOriginalMatImplementation);
BENCHMARK(BENCH_SobelOriginalNormalizationImplementation);
BENCHMARK(BENCH_SobelArrayImplementation);
BENCHMARK(BENCH_SobelHardcodeKernelsImplementation);
BENCHMARK(BENCH_SobelHardcodeKernelsNormalizationImplementation);

// Calls and runs the benchmark program
BENCHMARK_MAIN();
