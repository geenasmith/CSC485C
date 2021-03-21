#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <ctime>

using namespace cv;

/*
compile (cause pkg-config is annoying): g++ -std=c++11 -fopenmp sobel_openmp.cpp -o sobel `pkg-config --cflags --libs opencv`
run: ./sobel <input image>

g++ -std=c++11 -Xpreprocessor -fopenmp -lomp sobel_openmp.cpp -o sobel `pkg-config --cflags --libs opencv`
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

Mat sobel_original(Mat padded_image)
{
    // Define convolution kernels Gx and Gy
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;
    Mat sobel_image(n_rows, n_cols, CV_32F);

    // loop through calculating G_x and G_y
    // mag is sqrt(G_x^2 + G_y^2)

    clock_t start = clock();

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

            sobel_image.at<float>(r, c) = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
        }
    }

    clock_t end = clock();

    // normalize to 0-255
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8UC1);

    std::cout << "sobel original: " << end - start << std::endl;

    return sobel_image;
}

Mat sobel_float_array(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    // float padded_array[padded_image.rows][padded_image.cols];
    float **padded_array = (float **)malloc(padded_image.rows * sizeof(float *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (float *)malloc(padded_image.cols * sizeof(float));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<float>(i, j);

    clock_t start = clock();

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
            output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(n_rows, n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel float: " << end - start << std::endl;

    return output_image;
}

Mat sobel_int_array(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    // float output_array[n_rows][n_cols];
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    // uint8_t padded_array[padded_image.rows][padded_image.cols];
    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    // Mat sobel_image(n_rows, n_cols, CV_32F);

    clock_t start = clock();

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
            output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(n_rows, n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel int: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp_coarsegrain(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    // std::cout << "num threads:  " << num_threads << std::endl;

#pragma omp parallel for
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
            output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(n_rows, n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel openmp coarse: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp_coarsegrain_blocking(Mat padded_image)
{
    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    // std::cout << "num threads:  " << num_threads << std::endl;

    int row_jump = n_rows / num_threads;

#pragma omp parallel for
    for (int r = 0; r < n_rows; r += row_jump)
    {
        for (int j = r; j < n_rows || j < (r + row_jump); j++)
        {
            for (int c = 0; c < n_cols; c++)
            {
                float mag_x = padded_array[j][c] * 1 +
                              padded_array[j][c + 2] * -1 +
                              padded_array[j + 1][c] * 2 +
                              padded_array[j + 1][c + 2] * -2 +
                              padded_array[j + 2][c] * 1 +
                              padded_array[j + 2][c + 2] * -1;

                float mag_y = padded_array[j][c] * 1 +
                              padded_array[j][c + 1] * 2 +
                              padded_array[j][c + 2] * 1 +
                              padded_array[j + 2][c] * -1 +
                              padded_array[j + 2][c + 1] * -2 +
                              padded_array[j + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[j][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(n_rows, n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel openmpcoarse blocking: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp_finegrain(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    // std::cout << "num threads:  " << num_threads << std::endl;

    for (int r = 0; r < n_rows; r++)
    {
#pragma omp parallel for
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
            output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(n_rows, n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel openmp finegrain: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp_finegrain_blocking(Mat padded_image)
{
    // pad the right side of the padded image so n_rows is divisible by 8
    // keep track of the original size so we can trim after

    int padding_amount = (padded_image.cols - 2) % 8;
    // std::cout << "padding amount: " << padding_amount << std::endl;

    int orig_n_rows = padded_image.rows - 2;
    int orig_n_cols = padded_image.cols - 2;
    // std::cout << "original size: " << orig_n_rows << " by " << orig_n_cols << std::endl;

    copyMakeBorder(padded_image, padded_image, 0, 0, 0, padding_amount, BORDER_CONSTANT, 0);

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // std::cout << "padded size: " << n_rows << " by " << n_cols << std::endl;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    // std::cout << "num threads:  " << num_threads << std::endl;

    // cacheline size is 64B
    // block by 8 columns to reduce cache misses
    for (int r = 0; r < n_rows; r++)
    {
#pragma omp parallel for
        for (int c = 0; c < n_cols; c += 8)
        {
            // #pragma omp parallel for
            for (int j = 0; j < 8; j++)
            {
                float mag_x = padded_array[r][c + j] * 1 +
                              padded_array[r][c + j + 2] * -1 +
                              padded_array[r + 1][c + j] * 2 +
                              padded_array[r + 1][c + j + 2] * -2 +
                              padded_array[r + 2][c + j] * 1 +
                              padded_array[r + 2][c + j + 2] * -1;

                float mag_y = padded_array[r][c + j] * 1 +
                              padded_array[r][c + j + 1] * 2 +
                              padded_array[r][c + j + 2] * 1 +
                              padded_array[r + 2][c + j] * -1 +
                              padded_array[r + 2][c + j + 1] * -2 +
                              padded_array[r + 2][c + j + 2] * -1;

                // Instead of Mat, store the value into an array
                output_array[r][c + j] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(orig_n_rows, orig_n_cols, CV_32F);
    for (int i = 0; i < orig_n_rows; ++i)
        for (int j = 0; j < orig_n_cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel openmp finegrain blocking: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp_finegrain_dual_blocking(Mat padded_image)
{
    // pad the right side of the padded image so n_rows is divisible by 8
    // keep track of the original size so we can trim after

    int padding_amount = (padded_image.cols - 2) % 8;
    // std::cout << "padding amount: " << padding_amount << std::endl;

    int orig_n_rows = padded_image.rows - 2;
    int orig_n_cols = padded_image.cols - 2;
    // std::cout << "original size: " << orig_n_rows << " by " << orig_n_cols << std::endl;

    copyMakeBorder(padded_image, padded_image, 0, 0, 0, padding_amount, BORDER_CONSTANT, 0);

    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // std::cout << "padded size: " << n_rows << " by " << n_cols << std::endl;

    // Use array to store the image value
    float **output_array = (float **)malloc(n_rows * sizeof(float *));
    for (int i = 0; i < n_rows; i++)
    {
        output_array[i] = (float *)malloc(n_cols * sizeof(float));
    }

    uint8_t **padded_array = (uint8_t **)malloc(padded_image.rows * sizeof(uint8_t *));
    for (int i = 0; i < padded_image.rows; i++)
    {
        padded_array[i] = (uint8_t *)malloc(padded_image.cols * sizeof(uint8_t));
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    // std::cout << "num threads:  " << num_threads << std::endl;

    // cacheline size is 64B
    // block by 8 columns to reduce cache misses
    for (int r = 0; r < n_rows; r += 8)
    {
#pragma omp parallel for
        for (int c = 0; c < n_cols; c += 8)
        {
            for (int j = 0; j < 8; j++)
            {
                // #pragma omp parallel for

                for (int k = 0; k < 8; k++)
                {
                    float mag_x = padded_array[r + k][c + j] * 1 +
                                  padded_array[r + k][c + j + 2] * -1 +
                                  padded_array[r + k + 1][c + j] * 2 +
                                  padded_array[r + k + 1][c + j + 2] * -2 +
                                  padded_array[r + k + 2][c + j] * 1 +
                                  padded_array[r + k + 2][c + j + 2] * -1;

                    float mag_y = padded_array[r + k][c + j] * 1 +
                                  padded_array[r + k][c + j + 1] * 2 +
                                  padded_array[r + k][c + j + 2] * 1 +
                                  padded_array[r + k + 2][c + j] * -1 +
                                  padded_array[r + k + 2][c + j + 1] * -2 +
                                  padded_array[r + k + 2][c + j + 2] * -1;

                    // Instead of Mat, store the value into an array
                    output_array[r + k][c + j] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
                }
            }
        }
    }

    clock_t end = clock();

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat output_image = Mat(orig_n_rows, orig_n_cols, CV_32F);
    for (int i = 0; i < orig_n_rows; ++i)
        for (int j = 0; j < orig_n_cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel openmp finegrain dual blocking: " << end - start << std::endl;

    return output_image;
}

int main(int argc, char **argv)
{

    omp_set_num_threads(4); // TODO: change to input variable
    std::cout << "num threads:  " << omp_get_max_threads() << std::endl;

    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat raw_image = imread("images/frac3.png", IMREAD_GRAYSCALE);

    if (raw_image.empty())
    {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return 1;
    }

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);

    // ----- Initial Sobel Implementation -----

    Mat padded_img = preprocessing(image);

    Mat sobel_original_img = sobel_original(padded_img);
    // imwrite("sobel_original_img.jpg", sobel_original_img);
    // imshow("Detected Edges", sobel_original_img);
    // waitKey(0);

    Mat sobel_float_array_img = sobel_float_array(padded_img);
    // imwrite("sobel_float_array_img.jpg", sobel_float_array_img);
    // imshow("Detected Edges", sobel_float_array_img);
    // waitKey(0);

    Mat sobel_int_array_img = sobel_int_array(padded_img);
    // imwrite("sobel_int_array_img.jpg", sobel_int_array_img);
    // imshow("Detected Edges", sobel_int_array_img);
    // waitKey(0);

    Mat sobel_openmp_coarsegrain_img = sobel_openmp_coarsegrain(padded_img);
    // imwrite("sobel_openmp_coarsegrain_img.jpg", sobel_openmp_coarsegrain_img);
    // imshow("Detected Edges", sobel_openmp_coarsegrain_img);
    // waitKey(0);

    Mat sobel_openmp_coarsegrain_blocking_img = sobel_openmp_coarsegrain_blocking(padded_img);
    // imwrite("sobel_openmp_coarsegrain_blocking_img.jpg", sobel_openmp_coarsegrain_blocking_img);
    // imshow("Detected Edges", sobel_openmp_coarsegrain_blocking_img);
    // waitKey(0);

    Mat sobel_openmp_finegrain_img = sobel_openmp_finegrain(padded_img);
    // imwrite("sobel_openmp_finegrain_img.jpg", sobel_openmp_finegrain_img);
    // imshow("Detected Edges", sobel_openmp_finegrain_img);
    // waitKey(0);

    Mat sobel_openmp_finegrain_blocking_img = sobel_openmp_finegrain_blocking(padded_img);
    // imwrite("sobel_openmp_finegrain_blocking_img.jpg", sobel_openmp_finegrain_blocking_img);
    // imshow("Detected Edges", sobel_openmp_finegrain_blocking_img);
    // waitKey(0);

    //sobel_openmp_finegrain_dual_blocking
    Mat sobel_openmp_finegrain_dual_blocking_img = sobel_openmp_finegrain_dual_blocking(padded_img);
    // imwrite("sobel_openmp_finegrain_dual_blocking_img.jpg", sobel_openmp_finegrain_dual_blocking_img);
    // imshow("Detected Edges", sobel_openmp_finegrain_dual_blocking_img);
    // waitKey(0);

    return 0;
}