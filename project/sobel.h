#ifndef SOBEL_H
#define SOBEL_H

#include <type_traits>
#include <opencv2/core.hpp>

using namespace cv;


struct Sobel_uint8 {
    int orig_rows = 0;
    int orig_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;
    
    uint8_t **input;
    float **output;
};

struct Sobel_int32 {
    int orig_rows = 0;
    int orig_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;
    
    int32_t **input;
    float **output;
};

struct Sobel_float {
    int orig_rows = 0;
    int orig_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;
    
    float **input;
    float **output;
};

struct Sobel_uint8_gpu {
    int orig_rows = 0;
    int orig_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;
    
    uint8_t *input;
    float *output;
};

auto preprocessing_uint8_gpu(std::string filename, int tile_size)
{
    Sobel_uint8_gpu res;
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);
    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_8UC1);
    // gaussian filter to remove noise
    // GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;

    /**
     * Right side padding must be at LEAST 1, and also make the total image be divisible by block_size
     * Bottom side padding must follow the same
     * 
     * image is x,y
     * 
     */
    int y = image.rows;
    int x = image.cols;
    int xmod = (x % 32 == 0) ? 0 : x % 32;
    int ymod = (y % 32 == 0) ? 0 : y % 32;

    copyMakeBorder(image, padded_image, 1, ymod+1, 1, xmod+1, BORDER_CONSTANT, 0);

    res.input = new uint8_t[padded_image.rows * padded_image.cols];
    std::cout << "TEST"<< std::endl;

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            res.input[i * padded_image.rows + j] = (uint8_t)padded_image.at<uint8_t>(i, j);

    res.output = new float [image.rows*image.cols];
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            res.output[i * image.cols + j] = 0.0;

    res.orig_rows = image.rows;
    res.orig_cols = image.cols;
    res.padded_rows = padded_image.rows;
    res.padded_cols = padded_image.cols;
    return res;
};

/* Subset of 6 means compute innermost X pixels with simd.
 * # # # # # # # # - > # X X X X X X #
 */

auto preprocessing_uint8(std::string filename, int subset = 1, int right_pad=1, int left_pad=1, int top_pad=1, int bot_pad=1)
{
    Sobel_uint8 res;
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);
    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);
    // gaussian filter to remove noise
    // GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    
    int pad_to_subset = (image.cols % subset == 0) ? 0 : subset - (image.cols % subset);
    
    copyMakeBorder(image, padded_image, top_pad, bot_pad, left_pad, pad_to_subset + right_pad, BORDER_CONSTANT, 0);

    res.input = new uint8_t *[padded_image.rows];
    res.input[0] = new uint8_t[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.input[i] = res.input[i - 1] + padded_image.cols;
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            res.input[i][j] = (uint8_t)padded_image.at<float>(i, j);

    res.output = new float *[image.rows];
    res.output[0] = new float[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.output[i] = res.output[i - 1] + padded_image.cols;
    }

    res.orig_rows = image.rows;
    res.orig_cols = image.cols;
    res.padded_rows = padded_image.rows;
    res.padded_cols = padded_image.cols;
    return res;
};

auto preprocessing_int32(std::string filename, int subset = 1, int right_pad=1, int left_pad=1, int top_pad=1, int bot_pad=1)
{
    Sobel_int32 res;
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);

    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);
    // gaussian filter to remove noise
    // GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;

    // right_pad = (subset > 1) ? subset - (image.cols % subset) + right_pad : right_pad;
    
    int pad_to_subset = (image.cols % subset == 0) ? 0 : subset - (image.cols % subset);
    
    copyMakeBorder(image, padded_image, top_pad, bot_pad, left_pad, pad_to_subset + right_pad, BORDER_CONSTANT, 0);

    res.input = new int32_t *[padded_image.rows];
    res.input[0] = new int32_t[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.input[i] = res.input[i - 1] + padded_image.cols;
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            res.input[i][j] = (int32_t)padded_image.at<float>(i, j);

    // res.output = new float *[image.rows];
    // res.output[0] = new float[image.rows * (image.cols + pad_to_subset)];
    // for (int i = 1; i < image.rows; i++)
    // {
    //     res.output[i] = res.output[i - 1] + image.cols + pad_to_subset;
    // }
    res.output = new float *[image.rows];
    res.output[0] = new float[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.output[i] = res.output[i - 1] + padded_image.cols;
    }

    res.orig_rows = image.rows;
    res.orig_cols = image.cols;
    res.padded_rows = padded_image.rows;
    res.padded_cols = padded_image.cols;
    // printf("subset %d | rpad %d | lpad %d | tpad %d | bpad %d \n", subset, pad_to_subset + right_pad, left_pad, top_pad, bot_pad);
    return res;
};

auto preprocessing_float(std::string filename, int subset = 1, int right_pad=1, int left_pad=1, int top_pad=1, int bot_pad=1)
{
    Sobel_float res;
    Mat raw_image = imread(filename, IMREAD_GRAYSCALE);
    // convert image to CV_32F (equivalent to a float)
    Mat image;
    raw_image.convertTo(image, CV_32F);
    // gaussian filter to remove noise
    // GaussianBlur(image, image, Size(3, 3), 0, 0);

    // pad image with 1 px of 0s
    Mat padded_image;
    int pad_to_subset = (image.cols % subset == 0) ? 0 : subset - (image.cols % subset);
    
    copyMakeBorder(image, padded_image, top_pad, bot_pad, left_pad, pad_to_subset + right_pad, BORDER_CONSTANT, 0);

    res.input = new float *[padded_image.rows];
    res.input[0] = new float[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.input[i] = res.input[i - 1] + padded_image.cols;
    }

    for (int i = 0; i < padded_image.rows; ++i)
        for (int j = 0; j < padded_image.cols; ++j)
            res.input[i][j] = (float)padded_image.at<float>(i, j);

    res.output = new float *[image.rows];
    res.output[0] = new float[padded_image.rows * padded_image.cols];
    for (int i = 1; i < padded_image.rows; i++)
    {
        res.output[i] = res.output[i - 1] + padded_image.cols;
    }


    res.orig_rows = image.rows;
    res.orig_cols = image.cols;
    res.padded_rows = padded_image.rows;
    res.padded_cols = padded_image.cols;
    return res;
};

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

#endif


