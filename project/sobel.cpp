#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

/*
compile (cause pkg-config is annoying): g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv`
run: ./sobel <input image>
*/

Mat preprocessing(char *filename)
{
    /*
    Apply preprocessing:
      - read image as grayscale
      - convert to float equivalent (needed to access elements in Mat object)
      - apply gaussian filter to smooth image (removes noise that might be considered an edge)
      - pad the image with a 1px border of 0s for the sobel filter
    */

    // read in image as grayscale
    // Mat is an OpenCV data structure
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);

    if (raw_image.empty())
    {
        std::cout << "Could not read image: " << filename << std::endl;
        return raw_image;
    }

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);

    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    copyMakeBorder(image, padded_image, 1, 1, 1, 1, BORDER_CONSTANT, 0);

    return padded_image;
}

Mat sobel(Mat padded_image)
{
    /*
        Applies the sobel filtering to the padded image and normalizes to [0,255]
    */

    // Define convolution kernels Gx and Gy
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;
    Mat sobel_image(n_rows, n_cols, CV_32F);

    // loop through calculating G_x and G_y
    // mag is sqrt(G_x^2 + G_y^2)
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

    // normalize to 0-255
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8UC1);

    return sobel_image;
}

Mat sobel_array(Mat padded_image)
{
    /*
        Applies the sobel filtering to the padded image and normalizes to [0,255]
    */

    // Define convolution kernels Gx and Gy
    int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Filtered image definitions
    int n_rows = padded_image.rows - 1;
    int n_cols = padded_image.cols - 1;
    Mat sobel_image(n_rows, n_cols, CV_32F);

    // convert Mat to 2d array
    float imageArr[padded_image.rows][padded_image.cols];

    for(int i = 0 ;i < padded_image.rows; ++i)
      for(int j = 0; j < padded_image.cols; ++j)
        imageArr[i][j] = padded_image.at<float>(i, j);

    // loop through calculating G_x and G_y
    // mag is sqrt(G_x^2 + G_y^2)
    for (int r = 0; r < n_rows; r++)
    {
        for (int c = 0; c < n_cols; c++)
        {
            float mag_x = imageArr[r][c] * g_x[0][0] +
                          imageArr[r][c+1] * g_x[0][1] +
                          imageArr[r][c+2] * g_x[0][2] +
                          imageArr[r+1][c] * g_x[1][0] +
                          imageArr[r+1][c+1] * g_x[1][1] +
                          imageArr[r+1][c+2] * g_x[1][2] +
                          imageArr[r+2][c] * g_x[2][0] +
                          imageArr[r+2][c+1] * g_x[2][1] +
                          imageArr[r+2][c+2] * g_x[2][2];

            float mag_y = imageArr[r][c] * g_y[0][0] +
                          imageArr[r][c+1] * g_y[0][1] +
                          imageArr[r][c+2] * g_y[0][2] +
                          imageArr[r+1][c] * g_y[1][0] +
                          imageArr[r+1][c+1] * g_y[1][1] +
                          imageArr[r+1][c+2] * g_y[1][2] +
                          imageArr[r+2][c] * g_y[2][0] +
                          imageArr[r+2][c+1] * g_y[2][1] +
                          imageArr[r+2][c+2] * g_y[2][2];

            sobel_image.at<float>(r, c) = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
        }
    }

    // normalize to 0-255
    normalize(sobel_image, sobel_image, 0, 255, NORM_MINMAX, CV_8UC1);
    
    return sobel_image;
}


int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cout << "Usage: sobel <input image" << std::endl;
        return 0;
    }

    Mat padded_img = preprocessing(argv[1]);

    if (padded_img.empty())
    {
        return 1;
    }

    Mat sobel_img = sobel(padded_img);

    imshow("Detected Edges", sobel_img);
    waitKey(0);

    return 0;
}