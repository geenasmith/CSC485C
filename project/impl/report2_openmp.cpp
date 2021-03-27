#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <omp.h>
#include "../sobel.h"

namespace report2OpenMP {
    std::string base = "report2_OpenMP";


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

    namespace coarse {
        std::string implementation = base + "_" + "coarse";
        
        void sobel(Sobel_uint8 img)
        {
            #pragma omp parallel for
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
    }

    // namespace coarse_blocking {
    //     std::string implementation = base + "_" + "coarse_blocking";

        
    //     void sobel(Sobel_uint8 img)
    //     {
    //         auto const num_threads = omp_get_max_threads();
    //         int row_jump = img.orig_rows / num_threads;
    //         // std::cout << "ROWJUMP" << row_jump << "NUM_THREADS" << num_threads << "ROWS" << img.orig_rows << std::endl;

    //         #pragma omp parallel for
    //         for (int r = 0; r < img.orig_rows; r += num_threads)
    //         {
    //             for(int j = r; j < img.orig_rows || j < (r + num_threads); j++)
    //             {
    //                 for (int c = 0; c < img.orig_cols; c++)
    //                 {
    //                     // float mag_x = img.input[j][c] * 1 +
    //                     //             img.input[j][c + 2] * -1 +
    //                     //             img.input[j + 1][c] * 2 +
    //                     //             img.input[j + 1][c + 2] * -2 +
    //                     //             img.input[j + 2][c] * 1 +
    //                     //             img.input[j + 2][c + 2] * -1;

    //                     // float mag_y = img.input[j][c] * 1 +
    //                     //             img.input[j][c + 1] * 2 +
    //                     //             img.input[j][c + 2] * 1 +
    //                     //             img.input[j + 2][c] * -1 +
    //                     //             img.input[j + 2][c + 1] * -2 +
    //                     //             img.input[j + 2][c + 2] * -1;

    //                     // // Instead of Mat, store the value into an array
    //                     // img.output[j][c] = sqrt_impl(mag_x * mag_x + mag_y * mag_y);

    //                     auto sum = img.input[j+2][0] + img.input[j+1][0] + img.input[j][0];

    //                     auto tn = omp_get_thread_num();
    //                     std::cout<<"T"<<tn<<"@ROW"<<r<<"(j"<<j<<") @COL"<<c <<" sum"<< sum<<std::endl;
    //                     img.output[j][c] = 1;
    //                 }
    //             }
    //         }
    //     }
    // }

    namespace fine {
        std::string implementation = base + "_" + "fine";
        
        void sobel(Sobel_uint8 img)
        {
            for (int r = 0; r < img.orig_rows; r++)
            {
            #pragma omp parallel for
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
    }

}