#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <omp.h>
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX
#include "emmintrin.h" // SSE

#include "helpers.h"

using namespace cv;
using namespace std;


namespace SIMD
{
std::string base = "report2_SIMD";

const __m256i shift_left = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
const __m256i shift_right = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);

namespace Original
{
std::string implementation = base + "_" + "Original";

// void sobel(int32_t **padded_array, float **output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
void sobel(Sobel_int32 img)
{

    for (int r = 0; r < img.orig_rows; r++)
    {
        
        for (int c = 0; c < img.orig_cols; c+=6)
        {
            __m256i r0 = _mm256_loadu_si256((__m256i*) &(img.input[r][c]));
            __m256i r1 = _mm256_loadu_si256((__m256i*) &(img.input[r+1][c])); // calculated row
            __m256i r2 = _mm256_loadu_si256((__m256i*) &(img.input[r+2][c]));

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
            for (int i = 1; i < 7; i++) img.output[r][c+i-1] = g2[i];

        }
    }

}

} // namespace simd original