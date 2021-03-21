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
    // Filtered image definitions
    int n_rows = padded_image.rows - 2;
    int n_cols = padded_image.cols - 2;

    // Use array to store the image value
    uint8_t** padded_array;

    padded_array = new uint8_t*[padded_image.rows];
    padded_array[0] = new uint8_t[padded_image.rows * padded_image.cols];

    for(int i = 1; i < padded_image.rows; i++) {
    padded_array[i] = padded_array[i-1] + padded_image.cols;
    }
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    float** output_image;

    output_image = new float*[orig_rows];
    output_image[0] = new float[orig_rows * orig_cols];

    for(int i = 1; i < padded_image.rows; i++) {
        output_image[i] = output_image[i-1] + orig_cols;
    }

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
            output_image[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    // Have to write each pixel back to a Mat since we used malloc (not contiguous in memory anymore)
    Mat sobel_image = Mat(n_rows, n_cols, CV_32F, output_image);
    // for (int i = 0; i < sobel_image.rows; ++i)
    //     for (int j = 0; j < sobel_image.cols; ++j)
    //         sobel_image.at<float>(i, j) = output_image[i][j];

    // Use opencv's normalization function
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8U);

    return sobel_image;
}


int main(int argc, char **argv)
{
    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat image = imread("images/YODER.jpeg", IMREAD_GRAYSCALE);
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
    copyMakeBorder(image, padded_image, 1, 1, 1, 1, BORDER_CONSTANT, 0);
    padded_image.convertTo(padded_image, CV_32S);
    

    // ----- Array Sobel Implementation -----

    auto sum = 0.0;
    auto const num_trials = 100u;

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