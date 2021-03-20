#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX
#include "emmintrin.h" // SSE


using namespace cv;

/*
compile (cause pkg-config is annoying): g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv`
run: ./sobel <input image>
*/

/**
 * Fast Square Root
 **/
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

Mat sobel(Mat padded_image, int orig_cols, int orig_rows)
{
    /*
        Applies the sobel filtering to the padded image and normalizes to [0,255]
    */

    // Use array to store the image value
    int16_t padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = padded_image.at<int16_t>(i, j);

    int n_rows = padded_image.rows;
    int n_cols = padded_image.cols;

    // output of convolution
    int16_t output_image[n_rows][n_cols];

    // loop through calculating G_x and G_y
    // mag is sqrt(G_x^2 + G_y^2)
    /*
    for (auto r = 0; r < image.rows - 1; r++) {
        for (auto c = 1; c < image.cols - 1; c+= 14) {


            simd_ints r0 = simd_load(image[r][c-1:c+15]); // 16 pixels total
            simd_ints r1 = simd_load(image[r+1][c-1:c+15]); // 16 pixels total. Calculated row
            simd_ints r2 = simd_load(image[r+2][c-1:c+15]); // 16 pixels total

            simd_ints colx = simd_add(r1, r1);
            colx = simd_add(colx, r0);
            colx = simd_add(colx, r2);

            simd_ints mag_x = simd_subtract(simd_shift_right(colx), simd_shift_left(coly))

            simd_ints mag_y = simd_subtract(r0, r2);
            mag_y = simd_add(rowy, rowy); // the times 2 component
            mag_y = simd_add(mag_y, simd_shift_right(mag_y));
            mag_y = simd_add(mag_y, simd_shift_left(mag_y));

            simd_ints g = simd_add(
                simd_mult(mag_x, mag_x),
                simd_mult(mag_y, mag_y),
            )

            return sqrt(g); // not sure how to approach this yet

        }
    } */
    for (int r = 0; r < n_rows; r++)
    {
        for (int c = 0; c < n_cols; c+=6)
        {
            // _mm_loadu_si128: 6 | 0.5 -> maybe _mm_lddqu_si128 "when data crosses a cahce line boundary"
            // _mm_add_epi16: 1 | 0.33
            // _mm_sub_epi16: 1 | 0.33
            // _mm_slli_epi16: 1 | 0.5
            // _mm_srli_epi16: 1 | 0.5
            // _mm256_cvtepi16_epi32  3 | 1
            // _mm256_cvtepi32_ps -> 32int to 32f. might just be a direct cast


            // can try using 256i later, and then when doing the i->f conversion split into 2x 256sp


            __m128i r0 = _mm_loadu_si128((__m128i*) &(output_image[r][c]));
            __m128i r1 = _mm_loadu_si128((__m128i*) &(output_image[r+1][c])); // calculated row
            __m128i r2 = _mm_loadu_si128((__m128i*) &(output_image[r+2][c]));

            // Note: this should probably just be r1, not colx. saves 1 add
            __m128i temp = _mm_add_epi16(r1, r1); // "times 2"
            temp = _mm_add_epi16(temp, r0);
            temp = _mm_add_epi16(temp, r2);

            __m128i mag_x = _mm_sub_epi16(_mm_srli_epi16(temp, 1), _mm_slli_epi16(temp, 1));
            
            __m128i mag_y = _mm_setzero_si128();

            mag_y = _mm_add_epi16(r0, r0);
            mag_y = _mm_sub_epi16(mag_y, r2);
            mag_y = _mm_sub_epi16(mag_y, r2);
            mag_y = _mm_sub_epi16(mag_y, _mm_slli_epi16(r2,1));
            mag_y = _mm_add_epi16(mag_y, _mm_slli_epi16(r0,1));
            mag_y = _mm_sub_epi16(mag_y, _mm_srli_epi16(r2,1));
            mag_y = _mm_add_epi16(mag_y, _mm_srli_epi16(r0,1));

            _m256i g = _mm256_cvtepi16_epi32(mag_x), _mm256_cvtepi16_epi32 (mag_y)
            
            /*
                Two routes for computing g.

                1: Convert to float using _mm256_cvtepi32_ps (check that valeus are correct and that it doesn't just cast)
                Compute sqrt(x^2+y^2) as floats (fast)

                2: idk yet
            */

        }
    }

    Mat sobel_image(n_rows, n_cols, CV_32F,);

    // normalize to 0-255
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8UC1);

    return sobel_image;
}

int main(int argc, char **argv)
{
    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat raw_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

    if (raw_image.empty())
    {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return 1;
    }

    // convert image to CV_16S (equivalent to a int16_t)
    Mat image;
    raw_image.convertTo(image, CV_16S);

    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    // store original image size
    auto n_rows = image.rows;
    auto n_cols = image.cols;

    /**
     * Full image needs a 1px pad on all sides. Top, bottom and left are set in the copyMakeBorder call.
     * 
     * Right border needs to be 1 + a factor that makes the original image itself divisible by N where N is the number of values storable in a SIMD register - 2.
     */
    auto right_pad_width = 6 - (n_cols % 6) + 1; // for 16 values use 14. for 8 value use 6.

    // pad image with 1 px of 0s
    Mat padded_image;
    copyMakeBorder(image, padded_image, 1, 1, 1, right_pad_width, BORDER_CONSTANT, 0);

    // ----- Array Sobel Implementation -----

    Mat sobel_img = sobel(padded_image, n_cols, n_rows);

    imshow("Image", sobel_img);
    waitKey();

    return 0;
}