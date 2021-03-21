#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX
#include "emmintrin.h" // SSE


using namespace std;
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

    const __m256i shift_left = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
    const __m256i shift_right = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);

    // Use array to store the image value
    int32_t padded_array[padded_image.rows][padded_image.cols];
    int32_t padded_left[padded_image.rows][padded_image.cols];
    int32_t padded_right[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i) {
        for (int j = 0; j < padded_image.cols; ++j) {
            padded_array[i][j] = padded_image.at<int32_t>(i, j);
            // padded_left[i][j-1] = padded_image.at<int32_t>(i, j);
            // padded_right[i][j+1] = padded_image.at<int32_t>(i, j);
        }
    }
    // output of convolution
    float output_image[orig_rows][orig_cols];

    for (int r = 0; r < orig_rows; r++)
    {
        for (int c = 0; c < orig_cols; c+=6)
        {
            // can try using 256i later, and then when doing the i->f conversion split into 2x 256sp

            // load 3 rows of interest into 256ints
            __m256i r0 = _mm256_loadu_si256((__m256i*) &(padded_array[r][c]));
            __m256i r1 = _mm256_loadu_si256((__m256i*) &(padded_array[r+1][c])); // calculated row
            __m256i r2 = _mm256_loadu_si256((__m256i*) &(padded_array[r+2][c]));

            // cout << "act0\t\t"; for (int i = 0; i < 8; i++ ) std::cout << padded_array[r][c+i] << "\t"; cout << endl;
            // cout << "act1\t\t"; for (int i = 0; i < 8; i++ ) std::cout << padded_array[r][c+i] << "\t"; cout << endl;
            // cout << "act2\t\t"; for (int i = 0; i < 8; i++ ) std::cout << padded_array[r][c+i] << "\t"; cout << endl;
            
            // auto temp_ = reinterpret_cast< int * >( &r0 );
            // cout << "r0\t\t"; for (int i = 0; i < 8; i++ ) std::cout << temp_[i] << "\t"; cout << endl;
            // temp_ = reinterpret_cast< int * >( &r1 );
            // cout << "r1\t\t"; for (int i = 0; i < 8; i++ ) std::cout << temp_[i] << "\t"; cout << endl;
            // temp_ = reinterpret_cast< int * >( &r2 );
            // cout << "r2\t\t"; for (int i = 0; i < 8; i++ ) std::cout << temp_[i] << "\t"; cout << endl;

            // Note: this should probably just be r1, not colx. saves 1 add
            __m256i temp = _mm256_add_epi32(r1, r1); // "times 2"
            temp = _mm256_add_epi32(temp, r0);
            temp = _mm256_add_epi32(temp, r2);

            __m256i mag_x = _mm256_sub_epi32(_mm256_permutevar8x32_epi32(temp, shift_left),_mm256_permutevar8x32_epi32(temp, shift_right));

            // temp_ = reinterpret_cast< int * >( &mag_x );
            // cout << "mag_x\t\t"; for (int i = 1; i < 7; i++ ) std::cout << temp_[i] << "\t"; cout << endl;

            __m256i mag_y = _mm256_add_epi32(r0, r0);
            mag_y = _mm256_sub_epi32(mag_y, r2);
            mag_y = _mm256_sub_epi32(mag_y, r2);
            mag_y = _mm256_sub_epi32(mag_y, _mm256_permutevar8x32_epi32(r2,shift_left));
            mag_y = _mm256_add_epi32(mag_y, _mm256_permutevar8x32_epi32(r0,shift_left));
            mag_y = _mm256_sub_epi32(mag_y, _mm256_permutevar8x32_epi32(r2,shift_right));
            mag_y = _mm256_add_epi32(mag_y, _mm256_permutevar8x32_epi32(r0,shift_right));

            // auto temp_ = reinterpret_cast< int * >( &mag_y );
            // cout << "mag_y\t\t"; for (int i = 1; i < 7; i++ ) std::cout << temp_[i] << "\t"; cout << endl;

            __m256 fmx = _mm256_cvtepi32_ps(mag_x);
            __m256 fmy = _mm256_cvtepi32_ps(mag_y);

            __m256 g = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(fmx, fmx), _mm256_mul_ps(fmy, fmy)));
            auto g2 = reinterpret_cast< float * >( &g );
            for (int i = 1; i < 7; i++) output_image[r][c+i-1] = g2[i];

        }
    }

    Mat sobel_image(orig_rows, orig_cols, CV_32F, output_image);


    // normalize to 0-255
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8U);

    return sobel_image;
}


int main(int argc, char **argv)
{
    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);
    // GaussianBlur(image, image, Size(3, 3), 0, 0);
    
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
    padded_image.convertTo(padded_image, CV_32S);
    

    // ----- Array Sobel Implementation -----

    auto sum = 0.0;
    auto const num_trials = 20000u;

    auto const start_time = std::chrono::system_clock::now();
    Mat sobel_img;

    for( auto i = 0u; i < num_trials; ++i )
    {
        sobel_img = sobel(padded_image, n_cols, n_rows);
        sum += sobel_img.at<float>(rand() %200, rand() % 200);
    }
    std::cout << "notrelevant: " << sum << std::endl;

    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

    std::cout << "avg time: " << ( elapsed_time.count() / static_cast< float >( num_trials ) ) << " us" << std::endl;
    std::cout << "total time: " << ( elapsed_time.count() ) << " us on " << num_trials << " iterations" << std::endl;


    // imshow("output", sobel_img);
    // waitKey();

    return 0;
}