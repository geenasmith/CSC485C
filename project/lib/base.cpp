#include "helpers.h"

using namespace cv;
using namespace std;

namespace BASE {
    string prefix = "BASE";
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

    namespace baseline {
        string impl = prefix + "_baseline";
        void sobel(uint8_t *input, float *output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
        {
            for (int r = 0; r < orig_n_rows; r++)
            {
                for (int c = 0; c < orig_n_cols; c++)
                {
                    int r0 = r * padded_n_cols;
                    int r1 = (r + 1) * padded_n_cols;
                    int r2 = (r + 2) * padded_n_cols;
            
                    float mag_x = input[r0 + c]
                                - input[r0 + c + 2]
                                + input[r1 + c] * 2
                                - input[r1 + c + 2] * 2
                                + input[r2 + c]
                                - input[r2 + c + 2];
        
                    float mag_y = input[r0 + c]
                                + input[r0 + c + 1] * 2
                                + input[r0 + c + 2]
                                - input[r2 + c]
                                - input[r2 + c + 1] * 2
                                - input[r2 + c + 2];
                    // Instead of Mat, store the value into an array
                    output_array[r*orig_n_cols + c] = sqrt(mag_x * mag_x + mag_y * mag_y);
                }
            }
        }
    }

    namespace fastsqrt {
        string impl = prefix + "_fastsqrt";
        void sobel(uint8_t *input, float *output_array, int orig_n_rows, int orig_n_cols, int padded_n_rows, int padded_n_cols)
        {
            for (int r = 0; r < orig_n_rows; r++)
            {
                for (int c = 0; c < orig_n_cols; c++)
                {
                    int r0 = r * padded_n_cols;
                    int r1 = (r + 1) * padded_n_cols;
                    int r2 = (r + 2) * padded_n_cols;
            
                    float mag_x = input[r0 + c]
                                - input[r0 + c + 2]
                                + input[r1 + c] * 2
                                - input[r1 + c + 2] * 2
                                + input[r2 + c]
                                - input[r2 + c + 2];
        
                    float mag_y = input[r0 + c]
                                + input[r0 + c + 1] * 2
                                + input[r0 + c + 2]
                                - input[r2 + c]
                                - input[r2 + c + 1] * 2
                                - input[r2 + c + 2];
                    // Instead of Mat, store the value into an array
                    output_array[r*orig_n_cols + c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);
                }
            }
        }
    }

    /**
     * Runner runs a set number of benchmarks, and returns the average runtime in microseconds (us)
     * the input image is always a CV_8UC1, any conversions need to be handled by the runner.
     */
    auto runner(const int trials, const Mat INPUT_IMAGE, const int version, bool write_output_image) {
        int rows = INPUT_IMAGE.rows;
        int cols = INPUT_IMAGE.cols;

        string implementation;
        switch(version) {
            case 0: // base
            implementation = baseline::impl;
            break;
            case 1: // fastsqrt
            implementation = fastsqrt::impl;
            break;
            default: // base
            implementation = baseline::impl;
            break;
        }

        /**
        * Preprocessing
        */

        Mat UINT8_PADDED = preprocess(INPUT_IMAGE, 1, 1, 3);
        uint8_t* input = convertMat<uint8_t>(UINT8_PADDED);
        float* output = zeroed_array<float>(rows, cols);
        
        // ghost value and stopwatch
        auto sum = 0.0;
        int elapsed_time = 0; 

        //trials
        for (int i = 0; i < trials; i++) {
            auto start_time = chrono::system_clock::now();
            switch(version) {
                case 0: // base
                baseline::sobel(input, output, rows, cols, UINT8_PADDED.rows, UINT8_PADDED.cols);
                break;
                case 1: // fastsqrt
                fastsqrt::sobel(input, output, rows, cols, UINT8_PADDED.rows, UINT8_PADDED.cols);
                break;
                default: // base
                baseline::sobel(input, output, rows, cols, UINT8_PADDED.rows, UINT8_PADDED.cols);
                break;
            }
            auto end_time = chrono::system_clock::now();
            elapsed_time += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
            sum += output[1];
        }

        print_log(implementation, elapsed_time, trials, rows, cols);

        if (write_output_image == true) {
            Mat output_image = from_float_array(output, rows, cols);
            normalize(output_image, output_image, 0, 255, NORM_MINMAX, CV_32F);
            imwrite(OUTPUT_DIR + implementation + ".jpg", output_image);
            imwrite(OUTPUT_DIR + "BASE_INPUT" + ".jpg", UINT8_PADDED);
        }

        // cleanup
        free(input);
        free(output);

        ResultData res;
        res.rows = rows;
        res.cols = cols;
        res.avg_time = elapsed_time / static_cast<float>(trials);;
        res.name = implementation;
        return res;
    }
}