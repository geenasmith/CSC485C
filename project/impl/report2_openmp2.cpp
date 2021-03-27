#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <omp.h>

using namespace cv;

/*
auto [padded_array, output_array, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols] = report2::preprocessing("images/rgb1.jpg");
report2::openmp::sobel_coarse(padded_array, output_array, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols);
auto output_image = report2::postprocessing(output_array, orig_n_rows, orig_n_cols);
*/

namespace report2OpenMP
{
std::string base = "report2";

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

std::tuple<uint8_t **, float **, int, int, int, int> preprocessing(std::string filename)
{
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);
    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);
    // gaussian filter to remove noise
    // GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    copyMakeBorder(image, padded_image, 1, 1, 1, 1, BORDER_CONSTANT, 0);

    uint8_t **padded_array;
    padded_array = new uint8_t *[padded_image.rows];
    padded_array[0] = new uint8_t[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        padded_array[i] = padded_array[i - 1] + padded_image.cols;
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    float **output_array;
    output_array = new float *[image.rows];
    output_array[0] = new float[image.rows * image.cols];
    for (int i = 1; i < image.rows; i++)
    {
        output_array[i] = output_array[i - 1] + image.cols;
    }

    return {padded_array, output_array, image.rows, image.cols, padded_image.rows, padded_image.cols};
}

Mat postprocessing(float **output_array, int orig_n_rows, int orig_n_cols)
{
    Mat output_image = Mat(orig_n_rows, orig_n_cols, CV_32F);
    for (int i = 0; i < output_image.rows; ++i)
        for (int j = 0; j < output_image.cols; ++j)
            output_image.at<float>(i, j) = output_array[i][j];

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    return output_image;
}

namespace openmp
{
std::string implementation = base + "_" + "openmp";

void sobel_coarse(uint8_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
{
#pragma omp parallel for
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

void sobel_coarse_blocking(uint8_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
{
    auto const num_threads = omp_get_max_threads();
    int row_jump = orig_n_rows / num_threads;

#pragma omp parallel for
    for (int r = 0; r < orig_n_rows; r += row_jump)
    {
        for (int j = r; j < orig_n_rows || j < (r + row_jump); j++)
        {
            for (int c = 0; c < orig_n_cols; c++)
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
}

void sobel_fine(uint8_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
{
    for (int r = 0; r < orig_n_rows; r++)
    {
#pragma omp parallel for
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

} // namespace openmp

} // namespace report2