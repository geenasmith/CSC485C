#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "../sobel.h"

using namespace cv;

namespace report1 {
std::string base = "report1";

namespace baseline {
    std::string implementation = base + "_" + "baseline";


    std::tuple<Mat, int, int, int, int> preprocessing(std::string filename)
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

        return {padded_image, image.rows, image.cols, padded_image.rows, padded_image.cols};
    }

    Mat postprocessing(Mat sobel_image)
    {
        Mat output_image;
        normalize(sobel_image, output_image, 0, 255, NORM_MINMAX, CV_8UC1);

        return output_image;
    }


    Mat sobel(Mat padded_image, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
    {
        // Define convolution kernels Gx and Gy
        int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        Mat sobel_image(orig_n_rows, orig_n_cols, CV_32F);

        // loop through calculating G_x and G_y
        // mag is sqrt(G_x^2 + G_y^2)
        for (int r = 0; r < orig_n_rows; r++)
        {
            for (int c = 0; c < orig_n_cols; c++)
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

        return sobel_image;
    }

} //namespace baseline

namespace floatArray
{
    std::string implementation = base + "_" + "floatArray";

    void sobel(Sobel_float img)
    {
        for (int r = 0; r < img.orig_rows; r++)
        {
            for (int c = 0; c < img.orig_cols; c++)
            {
                float mag_x = img.input[r][c] * 1 +
                            img.input[r][c + 2] * -1 +
                            img.input[r + 1][c] * 2 +
                            img.input[r + 1][c + 2] * -2 +
                            img.input[r + 2][c] * 1 +
                            img.input[r + 2][c + 2] * -1;

                float mag_y = img.input[r][c] * 1 +
                            img.input[r][c + 1] * 2 +
                            img.input[r][c + 2] * 1 +
                            img.input[r + 2][c] * -1 +
                            img.input[r + 2][c + 1] * -2 +
                            img.input[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                img.output[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }
    }

} // namespace floatArray


namespace uint8Array
{
    std::string implementation = base + "_" + "uint8Array";

    void sobel(Sobel_uint8 img)
    {
        for (int r = 0; r < img.orig_rows; r++)
        {
            for (int c = 0; c < img.orig_cols; c++)
            {
                float mag_x = img.input[r][c] * 1 +
                            img.input[r][c + 2] * -1 +
                            img.input[r + 1][c] * 2 +
                            img.input[r + 1][c + 2] * -2 +
                            img.input[r + 2][c] * 1 +
                            img.input[r + 2][c + 2] * -1;

                float mag_y = img.input[r][c] * 1 +
                            img.input[r][c + 1] * 2 +
                            img.input[r][c + 2] * 1 +
                            img.input[r + 2][c] * -1 +
                            img.input[r + 2][c + 1] * -2 +
                            img.input[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                img.output[r][c] = sqrt(pow(mag_x, 2) + pow(mag_y, 2));
            }
        }
    }
} // namespace uint8Array

namespace uint8FastSqrt
{
    std::string implementation = base + "_" + "uint8FastSqrt";


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

    void sobel(Sobel_uint8 img)
    {
        for (int r = 0; r < img.orig_rows; r++)
        {
            for (int c = 0; c < img.orig_cols; c++)
            {
                float mag_x = img.input[r][c] * 1 +
                            img.input[r][c + 2] * -1 +
                            img.input[r + 1][c] * 2 +
                            img.input[r + 1][c + 2] * -2 +
                            img.input[r + 2][c] * 1 +
                            img.input[r + 2][c + 2] * -1;

                float mag_y = img.input[r][c] * 1 +
                            img.input[r][c + 1] * 2 +
                            img.input[r][c + 2] * 1 +
                            img.input[r + 2][c] * -1 +
                            img.input[r + 2][c + 1] * -2 +
                            img.input[r + 2][c + 2] * -1;

                // Instead of Mat, store the value into an array
                img.output[r][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
            }
        }
    }
} // namespace uint8Array

}