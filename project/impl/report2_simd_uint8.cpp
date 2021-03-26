#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX
#include "emmintrin.h" // SSE

using namespace cv;

/*
auto [padded_array, output_array, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols] = report2::preprocessing("images/rgb1.jpg");
report2::openmp::sobel_coarse(padded_array, output_array, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols);
auto output_image = report2::postprocessing(output_array, orig_n_rows, orig_n_cols);
*/

namespace report2SIMD
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
    auto right_pad_width = 6 - (image.cols % 6) + 1; // for 16 values use 14. for 8 value use 6.

    copyMakeBorder(image, padded_image, 1, 1, 1, right_pad_width, BORDER_CONSTANT, 0);

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

namespace SIMD_Original
{
std::string implementation = base + "_" + "Original";


const __m256i shift_left = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
const __m256i shift_right = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);

void sobel(int32_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
{
    for (int r = 0; r < orig_n_rows; r++)
    {
        for (int c = 0; c < orig_n_cols; c+=6)
        {
            __m256i r0 = _mm256_loadu_si256((__m256i*) &(padded_array[r][c]));
            __m256i r1 = _mm256_loadu_si256((__m256i*) &(padded_array[r+1][c])); // calculated row
            __m256i r2 = _mm256_loadu_si256((__m256i*) &(padded_array[r+2][c]));

            __m256i temp = _mm256_add_epi32(r1, r1); // "times 2"
            temp = _mm256_add_epi32(temp, r0);
            temp = _mm256_add_epi32(temp, r2);

            __m256i mag_x = _mm256_sub_epi32(_mm256_permutevar8x32_epi32(temp, shift_left),_mm256_permutevar8x32_epi32(temp, shift_right));

            __m256i mag_y = _mm256_add_epi32(r0, r0);
            mag_y = _mm256_sub_epi32(mag_y, r2);
            mag_y = _mm256_sub_epi32(mag_y, r2);
            mag_y = _mm256_sub_epi32(mag_y, _mm256_permutevar8x32_epi32(r2,shift_left));
            mag_y = _mm256_add_epi32(mag_y, _mm256_permutevar8x32_epi32(r0,shift_left));
            mag_y = _mm256_sub_epi32(mag_y, _mm256_permutevar8x32_epi32(r2,shift_right));
            mag_y = _mm256_add_epi32(mag_y, _mm256_permutevar8x32_epi32(r0,shift_right));

            __m256 fmx = _mm256_cvtepi32_ps(mag_x);
            __m256 fmy = _mm256_cvtepi32_ps(mag_y);

            __m256 g = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(fmx, fmx), _mm256_mul_ps(fmy, fmy)));

            auto g2 = reinterpret_cast< float * >( &g );
            for (int i = 1; i < 7; i++) output_array[r][c+i-1] = g2[i];
        }
    }
}

} // namespace simd original

} // namespace report2 simd