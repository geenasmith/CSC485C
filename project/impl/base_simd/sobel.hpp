#ifndef SOBEL_H
#define SOBEL_H

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

Mat sobel(Mat padded_image, int orig_cols, int orig_rows);
Mat preprocess(std::string);
Mat postprocess(Mat );

#endif