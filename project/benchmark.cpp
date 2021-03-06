#include "impl/report1.cpp"
#include "impl/report2_openmp.cpp"
#include "impl/report2_simd.cpp"
#include "sobel.h"
#include <cstdlib>

#include <omp.h>
using namespace cv;

void output_time(std::string name, std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end, uint trials, int h, int w) {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << name << " with " << trials << " trials" << std::endl;
    std::cout << "Average time per run: " << elapsed_time.count() / static_cast<float>(trials) << " us" << std::endl;
    std::cout << "Input resolution " << w << "x" << h << std::endl << std::endl;
}

int baseline(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report1::baseline
    auto [image, rows, cols, p_rows, p_cols] = VERS::preprocessing(file_name);
    Mat output;
    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        output = VERS::sobel(image, rows, cols, p_rows, p_cols);
        sum += output.at<float>(1,1);
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, rows, cols);

    if(display) imshow(VERS::implementation, VERS::postprocessing(output));
    return sum;
}

int report1_float(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report1::floatArray
    
    auto image = preprocessing_float(file_name);

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}

int report1_uint8(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report1::uint8Array
    auto image = preprocessing_uint8(file_name);

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}

int report1_uint8_fastsqrt(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report1::uint8FastSqrt
    auto image = preprocessing_uint8(file_name);

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}

int report2_openmp_coarse(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report2OpenMP::coarse
    auto image = preprocessing_uint8(file_name);

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}

int report2_openmp_fine(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report2OpenMP::fine
    auto image = preprocessing_uint8(file_name);

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}


int report2_simd(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report2SIMD::Original
    auto image = preprocessing_int32(file_name, 6);
    // std::cout << image.orig_rows << " " << image.orig_cols << " | " << image.padded_rows << " " << image.padded_cols << std::endl;

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}
int report2_simd_ompcoarse(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report2SIMD::OMPCoarse
    auto image = preprocessing_int32(file_name, 6);
    // std::cout << image.orig_rows << " " << image.orig_cols << " | " << image.padded_rows << " " << image.padded_cols << std::endl;

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}

int report2_simd_ompfine(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
    #undef VERS
    #define VERS report2SIMD::OMPFine
    auto image = preprocessing_int32(file_name, 6);
    // std::cout << image.orig_rows << " " << image.orig_cols << " | " << image.padded_rows << " " << image.padded_cols << std::endl;

    auto sum = 0;
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        VERS::sobel(image);
        sum += image.output[1][1];
    }
    auto end_time = std::chrono::system_clock::now();
    output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

    if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
    return sum;
}
// int report2_simd_uint8(std::string file_name="images/rgb1.jpg", uint benchmark_trials = 2000u, bool display = true) {
//     #undef VERS
//     #define VERS report2SIMD::UInt8
//     auto image = preprocessing_uint8(file_name, 6);
//     // std::cout << image.orig_rows << " " << image.orig_cols << " | " << image.padded_rows << " " << image.padded_cols << std::endl;

//     auto sum = 0;
//     auto start_time = std::chrono::system_clock::now();
//     for (auto i = 0u; i < benchmark_trials; ++i)
//     {
//         VERS::sobel(image);
//         sum += image.output[1][1];
//     }
//     auto end_time = std::chrono::system_clock::now();
//     output_time(VERS::implementation, start_time, end_time, benchmark_trials, image.orig_rows, image.orig_cols);

//     if(display) imshow(VERS::implementation, postprocessing(image.output, image.orig_rows, image.orig_cols));
//     return sum;
// }



/*
g++ -std=c++17 -Xpreprocessor -fopenmp -lomp benchmark.cpp -o sobel `pkg-config --cflags --libs opencv`
*/

int main(int argc, char **argv)
{

    auto sum = 0;
    auto const benchmark_trials = 10u;
    bool const display_outputs = false;
    
    // std::string file_name="sampleset/frac1.png";
    // std::string file_name="images/frac2.png";
    // std::string file_name = "images/rgb1.jpg";

    std::string file_name = argv[1];
    int test_case = atoi(argv[2]);
    int threads = atoi(argv[3]);
    omp_set_num_threads(threads);
    std::cout << "IMAGE: " << file_name << " TEST_CASE: " << test_case << " NUM_THREADS: " << threads << std::endl;

    /******************
     * Report 1 Bench *
     ******************/
    switch(test_case) {
    case 0:
        sum += baseline(file_name, benchmark_trials, display_outputs);
        break;
    case 1:
        sum += report1_float(file_name, benchmark_trials, display_outputs);
        break;
    case 2:
        sum += report1_uint8(file_name, benchmark_trials, display_outputs);
        break;
    case 3:
        sum += report1_uint8_fastsqrt(file_name, benchmark_trials, display_outputs);
        break;
    
    /******************
     * Report 2 OpenMP*
     ******************/
    case 4:
        sum += report2_openmp_coarse(file_name, benchmark_trials, display_outputs);
        break;
    case 5:
        sum += report2_openmp_fine(file_name, benchmark_trials, display_outputs);
        break;

    /******************
     * Report 2 SIMD  *
     ******************/
    case 6:
        sum += report2_simd(file_name, benchmark_trials, display_outputs);
        break;
    case 7:
        sum += report2_simd_ompcoarse(file_name, benchmark_trials, display_outputs);
        break;
    case 8:
        sum += report2_simd_ompfine(file_name, benchmark_trials, display_outputs);
        break;
    case 9:
        // sum += report2_simd_uint8(file_name, benchmark_trials, display_outputs);
        break;
    default:
        std::cout << "invalid test#" << std::endl;
    }

    // /******************
    //  * Report 1 Bench *
    //  ******************/
    // // sum += baseline(file_name, benchmark_trials, display_outputs);
    // // sum += report1_float(file_name, benchmark_trials, display_outputs);
    // // sum += report1_uint8(file_name, benchmark_trials, display_outputs);
    // // sum += report1_uint8_fastsqrt(file_name, benchmark_trials, display_outputs);
    
    // /******************
    //  * Report 2 OpenMP*
    //  ******************/
    // sum += report2_openmp_coarse(file_name, benchmark_trials, display_outputs);
    // sum += report2_openmp_fine(file_name, benchmark_trials, display_outputs);

    // /******************
    //  * Report 2 SIMD  *
    //  ******************/
    // sum += report2_simd(file_name, benchmark_trials, display_outputs);
    // sum += report2_simd_uint8(file_name, benchmark_trials, display_outputs);


    // /******************
    //  * Report 2 Both  *
    //  ******************/
    // sum += report2_simd(file_name, benchmark_trials, display_outputs);

    
    if(display_outputs) waitKey(0);

    std::cout << "Not Relevant: " << sum << std::endl; 
    return 0;
}
