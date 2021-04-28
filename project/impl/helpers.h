#ifndef HELPERS_H
#define HELPERS_H

#include <opencv2/core.hpp>


using namespace cv;
using namespace std;

string OUTPUT_DIR = "outputs/";

void print_log(const string title, const int duration_ms, const int trials, const int rows, const int cols) {
    printf("%s (%d x %d): %f us", title, cols, rows, duration_ms / static_cast<float>(trials));
}

void output_time(std::string name, std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end, uint trials, int h, int w) {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << name << " with " << trials << " trials" << std::endl;
    std::cout << "Average time per run: " << elapsed_time.count() / static_cast<float>(trials) << " us" << std::endl;
    std::cout << "Input resolution " << w << "x" << h << std::endl << std::endl;
}

Mat preprocess(const Mat input, const int x_divisor, const int y_divisor, const int gaussian = 1) {
    Mat padded;
    GaussianBlur(input, padded, Size(gaussian, gaussian), 0, 0);
    /**
     * Right side padding must be at LEAST 1, and also make the total image be divisible by block_size
     * Bottom side padding must follow the same
     */
    int xmod = (input.cols % x_divisor == 0) ? 0 : input.cols % x_divisor;
    int ymod = (input.rows % y_divisor == 0) ? 0 : input.rows % y_divisor;

    copyMakeBorder(padded, padded, 1, ymod+1, 1, xmod+1, BORDER_CONSTANT, 0);
    return padded;
}

void printfloat(float* data, int cols, int rm, int cm) {
    for (int r = 0; r < rm; r++) {
        for (int c = 0; c < cm; c++) {
            printf("%3.f ", data[c+r*cols]);
        }
        printf("\n");
    }
}

void printuint(uint8_t* data, int cols, int rm, int cm) {
    for (int r = 0; r < rm; r++) {
        for (int c = 0; c < cm; c++) {
            printf("%3.d ", unsigned(data[c+r*cols]));
        }
        printf("\n");
    }
}

template <typename T>
T* convertMat(const Mat input) {
    T* array = (T*)malloc(sizeof(T) * input.cols * input.rows);

    for (int r = 0; r < input.rows; r++)
        for (int c = 0; c <input.cols; c++)
            array[r*input.cols + c] = input.at<T>(r, c);

    return array;
}

template <typename T>
T* zeroed_array(const int rows, const int cols) {
    T* array = (T*)malloc(sizeof(T) * cols * rows);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c <cols; c++)
            array[r*cols + c] = 0;

    return array;
}

Mat from_float_array(float* input, const int rows, const int cols) {
    Mat img(rows, cols, CV_32F);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c <cols; c++)
            img.at<float>(r,c) = input[r*cols + c];
    return img;
}


#endif


