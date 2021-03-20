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

Mat sobel_float_array(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;

    // Use array to store the image value
    float output_array[n_rows][n_cols];
    float padded_array[padded_image.rows][padded_image.cols];
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
    Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    std::cout << "sobel float: " << end - start << std::endl;

    return output_image;
}

Mat sobel_int_array(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;

    // Use array to store the image value
    float output_array[n_rows][n_cols];
    uint8_t padded_array[padded_image.rows][padded_image.cols];
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

    Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    // imshow("Detected Edges", output_image);
    // waitKey(0);

    std::cout << "sobel int: " << end - start << std::endl;

    return output_image;
}

Mat sobel_openmp(Mat padded_image)
{

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;

    // Use array to store the image value
    float output_array[n_rows][n_cols];
    uint8_t padded_array[padded_image.rows][padded_image.cols];
    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            padded_array[i][j] = (uint8_t)padded_image.at<float>(i, j);

    clock_t start = clock();

    auto const num_threads = omp_get_max_threads();
    std::cout << "num threads:  " << num_threads << std::endl;

    float thread_outputs[num_threads][n_rows][n_cols];

#pragma omp parallel for
    for (int r = 0; r < n_rows; r++)
    {
        auto const t = omp_get_thread_num();

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
            // output_array[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
            thread_outputs[t][r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
        }
    }

    // clock_t end = clock();

    for (int r = 0; r < n_rows; r++)
    {
        for (int c = 0; c < n_cols; c++)
        {
            float sum = 0;
            for (int t = 0; t < num_threads; t++)
            {
                sum += thread_outputs[t][r][c];
            }
            output_array[r][c] = sum;
        }
    }

    clock_t end = clock();

    Mat output_image = Mat(n_rows, n_cols, CV_32F, output_array);

    // Use opencv's normalization function
    cv::normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_8U);

    // imshow("Detected Edges", output_image);
    // waitKey(0);

    std::cout << "sobel openmp: " << end - start << std::endl;

    return output_image;
}

int main(int argc, char **argv)
{

    omp_set_num_threads(2); // TODO: change to input variable

    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat raw_image = imread("images/rgb1.jpg", IMREAD_GRAYSCALE);

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

    Mat sobel_float_array_img = sobel_float_array(padded_img);
    // imwrite("mat.jpg", sobel_float_array_img);
    // imshow("Detected Edges", sobel_float_array_img);
    // waitKey(0);

    Mat sobel_int_array_img = sobel_int_array(padded_img);
    // imwrite("mat.jpg", sobel_int_array_img);
    // imshow("Detected Edges", sobel_int_array_img);
    // waitKey(0);

    Mat sobel_openmp_img = sobel_openmp(padded_img);

    return 0;
}