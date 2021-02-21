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

Mat resizeImage(Mat input_image, int resize_factor)
{
    double resize_amount = 1.0 / (1 << resize_factor);
    Mat resized_image;
    resize(input_image, resized_image, Size(), resize_amount, resize_amount);
    return resized_image;
}
/*
    Initial benchmarking implementation
      - Uses opencv Mat objects and opencv normalization
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
    Initial benchmarking implementation with our own normalization implemented (not using opencv's)
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

        // Convert array to Mat, base on documentation, this function only create a header that points to the data
        Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

        // Use opencv's normalization function
        normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);

        benchmark::DoNotOptimize(output_image);
    }
}

/*
    Hardcode the kernel values into the multiplication. Uses opencv normalization.
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

    # Note/Guess: this should see biggest speedup in smaller images as multiple lines will sit in the cacheline
    BUT speculative prefetching might pick up on the jumps
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
    Input image is uint8 type. Hardcode the kernal values into the multiplication. Uses our normalization.
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

                register float mag_y = padded_array[r][c] * 1 +
                                       padded_array[r][c + 1] * 2 +
                                       padded_array[r][c + 2] * 1;

                register float mag_x = padded_array[r][c] * 1 +
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
    Uses the BENCH_Sobeluint8Input1DArray as a base, but reorders the math operations

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
    Change to use 2d vectors. Input image is uint8 type. Hardcode the kernal values into the multiplication. Uses our normalization.
*/
static void BENCH_SobelVectorImplementation(benchmark::State &state)
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
    // uint8_t padded_array[padded_image.rows][padded_image.cols];
    std::vector<std::vector<uint8_t>> padded_array(padded_image.rows, std::vector<uint8_t>(padded_image.cols, 0));
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
    Change mag_x and mag_y to be integers
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
    combine the determination of max and min for normalization into convolution loops
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
    Based on BENCH_SobelCombineMaxMinImplementation
    combine the determination of max and min for normalization into convolution loops
*/
static void BENCH_SobelTiledImplementation(benchmark::State &state)
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

    /**
     * Tiling Concept/Pseudocode:
     * 
     * Break the input image into blocks of X*Y pixels where X*Y <= CacheLineSize.
     * 
     * Given an image that is 14x14, with X being a padded border on the outer edge, split into 8x8 tiles and compute. The first tile operates on the #:
     * X X X X X X X X X X X X X X X X X   ->  X X X X X X X X X X X X X X X X X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X # # # # # # 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X 1 2 3 4 5 6 7 8 9 0 A B C D E X   ->  X 1 2 3 4 5 6 7 8 9 0 A B C D E X 
     * X X X X X X X X X X X X X X X X X   ->  X X X X X X X X X X X X X X X X X 

    To do this, we can write the 8x8 block as contiguous in memory, and each 8x8 block overlaps on the border edge.
    Then, we iterate through each tile.
     **/


    auto const ptile_width = 8u;
    auto const ptile_height = 8u;
    auto const tile_width = ptile_width - 2u;
    auto const tile_height = ptile_height - 2u;

    /*
    Apply preprocessing: This isa modified version of preprocessing to not pad directly.
      - apply gaussian filter to smooth image (removes noise that might be considered an edge)
      - pad the image with a 1px border of 0s for the sobel filter
    */
    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    // Padded image needs to be 2 + tile_width * ceil(n_cols / tile_width). 1 is the top border, so:
    int b_b = 1 + (tile_height * ceil(n_rows/tile_height)) - n_rows
    int b_r = 1 + (tile_width * ceil(n_cols/tile_width)) - n_cols

    Mat pi;
    copyMakeBorder(image, pi, 1, b_b, 1, b_r, BORDER_CONSTANT, 0);
    
    auto int tiles_x = ceil(n_cols / tile_width);
    auto int tiles_y = ceil(n_rows / tile_height);

    uint8_t tiled_array[tiles_x*tiles_y][ptile_width*ptile_height]; // 2d array of "tiles". A ptile is a 8x8 block of pixels
    for (int r = 0; r < pi.rows; ++r) {
        for (int c = 0; c < pi.cols; ++c) {
            tiled_array[][]
        }
    }

    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) { // for tile at (tx, ty)


            // for (int y = 0; y < ptile_height; ++y) {
            //     for (int x = 0; x < ptile_width; ++x) {
            //         tiled_array[ty][tx][y * ptile_width + x] = pi.at<int>(tx*tile_width + x, y - 1);
            //     }
            // }

            // for (int pty = 0; pty < ptile_height; ++pty) { // 0, 1, 2, 3, 4, 5, 6, 7
            //     for (int ptx = 0; ptx < ptile_width; ++ptx) {
            //         tiled_array[ty*tx][pty*ptile_width + ptx] = padded_image.at<int>(ptx, pty)
            //     }
            // }
        }
    }

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
Can only pass in arguments to benchmark function that are integers.
To run on different sized images, will pass in how small to shrink the image.
Eg. passing in 0 will resize image by 1/2^0 on each axis, 1 will resize to 1/2^1 etc.
*/
BENCHMARK(BENCH_SobelOriginalMatImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelOriginalNormalizationImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelArrayImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodeKernelsImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodeKernelsNormalizationImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8InputImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelVectorImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelMagIntsImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelHardcodePowImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_SobelCombineMaxMinImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8InputImplementation)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8InputImplementationLoopUnroll)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8Input1DArray)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8Input1DArrayMathOperationReorder)->DenseRange(0, DENSERANGEEND, 1);
BENCHMARK(BENCH_Sobeluint8MathOperationReorder)->DenseRange(0, DENSERANGEEND, 1);

// Calls and runs the benchmark program
BENCHMARK_MAIN();
