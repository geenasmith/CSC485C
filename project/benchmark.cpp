#include "impl/report1_baseline.cpp"
#include "impl/report1_float.cpp"
#include "impl/report1_uint8.cpp"
#include "impl/report2_openmp.cpp"

#include <omp.h>
using namespace cv;

/*
g++ -std=c++17 -Xpreprocessor -fopenmp -lomp benchmark.cpp -o sobel `pkg-config --cflags --libs opencv`
*/

int main()
{
    auto sum = 0;
    auto const benchmark_trials = 2000u;

    // ----- Original Mat -----

    auto [padded_image, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols] = report1Baseline::preprocessing("images/rgb1.jpg");

    auto const start_time = std::chrono::system_clock::now();

    Mat sobel_image;

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        sobel_image = report1Baseline::baseline::sobel(padded_image, orig_n_rows, orig_n_cols, padded_n_rows, padded_n_cols);
        sum += sobel_image.at<float>(1, 1);
    }

    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    auto output_image = report1Baseline::postprocessing(sobel_image);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Original Mat: average time per run: "
              << elapsed_time.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    // imshow("test", output_image);
    // waitKey(0);

    // ----- Array float -----

    sum = 0;

    auto [padded_array1, output_array1, orig_n_rows1, orig_n_cols1, padded_n_rows1, padded_n_cols1] = report1Float::preprocessing("images/rgb1.jpg");

    auto const start_time1 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report1Float::floatArray::sobel(padded_array1, output_array1, orig_n_rows1, orig_n_cols1, padded_n_rows1, padded_n_cols1);
        sum += output_array1[1][1];
    }

    auto const end_time1 = std::chrono::system_clock::now();
    auto const elapsed_time1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time1 - start_time1);

    auto output_image1 = report1Float::postprocessing(output_array1, orig_n_rows1, orig_n_cols1);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Float Array: average time per run: "
              << elapsed_time1.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    // imshow("test", output_image);
    // waitKey(0);

    // ----- Array uint8 -----

    auto [padded_array2, output_array2, orig_n_rows2, orig_n_cols2, padded_n_rows2, padded_n_cols2] = report1Uint8::preprocessing("images/rgb1.jpg");

    sum = 0;
    auto const start_time2 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report1Uint8::uint8Array::sobel_sqrt(padded_array2, output_array2, orig_n_rows2, orig_n_cols2, padded_n_rows2, padded_n_cols2);
        sum += output_array2[1][1];
    }

    auto const end_time2 = std::chrono::system_clock::now();
    auto const elapsed_time2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);

    auto output_image2 = report1Uint8::postprocessing(output_array2, orig_n_rows2, orig_n_cols2);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Uint8 Array sqrt: average time per run: "
              << elapsed_time2.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    sum = 0;
    auto const start_time3 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report1Uint8::uint8Array::sobel_fastsqrt(padded_array2, output_array2, orig_n_rows2, orig_n_cols2, padded_n_rows2, padded_n_cols2);
        sum += output_array2[1][1];
    }

    auto const end_time3 = std::chrono::system_clock::now();
    auto const elapsed_time3 = std::chrono::duration_cast<std::chrono::microseconds>(end_time3 - start_time3);

    output_image2 = report1Uint8::postprocessing(output_array2, orig_n_rows2, orig_n_cols2);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Uint8 Array fast sqrt: average time per run: "
              << elapsed_time3.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    // imshow("test", output_image);
    // waitKey(0);

    // ----- openmp coarse -----

    omp_set_num_threads(8); // TODO: change to input variable
    std::cout << "num threads:  " << omp_get_max_threads() << std::endl;

    auto [padded_array3, output_array3, orig_n_rows3, orig_n_cols3, padded_n_rows3, padded_n_cols3] = report2OpenMP::preprocessing("images/rgb1.jpg");

    sum = 0;

    auto const start_time4 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report2OpenMP::openmp::sobel_coarse(padded_array3, output_array3, orig_n_rows3, orig_n_cols3, padded_n_rows3, padded_n_cols3);
        sum += output_array3[1][1];
    }

    auto const end_time4 = std::chrono::system_clock::now();
    auto const elapsed_time4 = std::chrono::duration_cast<std::chrono::microseconds>(end_time4 - start_time4);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Openmp  coarse: average time per run: "
              << elapsed_time4.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    sum = 0;
    auto const start_time5 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report2OpenMP::openmp::sobel_coarse_blocking(padded_array3, output_array3, orig_n_rows3, orig_n_cols3, padded_n_rows3, padded_n_cols3);
        sum += output_array3[1][1];
    }

    auto const end_time5 = std::chrono::system_clock::now();
    auto const elapsed_time5 = std::chrono::duration_cast<std::chrono::microseconds>(end_time5 - start_time5);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Openmp coarse blocking: average time per run: "
              << elapsed_time5.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    sum = 0;
    auto const start_time6 = std::chrono::system_clock::now();

    for (auto i = 0u; i < benchmark_trials; ++i)
    {
        report2OpenMP::openmp::sobel_fine(padded_array3, output_array3, orig_n_rows3, orig_n_cols3, padded_n_rows3, padded_n_cols3);
        sum += output_array3[1][1];
    }

    auto const end_time6 = std::chrono::system_clock::now();
    auto const elapsed_time6 = std::chrono::duration_cast<std::chrono::microseconds>(end_time6 - start_time6);

    std::cout << "sum = " << (sum / static_cast<float>(benchmark_trials)) << std::endl;
    std::cout << "Openmp fine: average time per run: "
              << elapsed_time6.count() / static_cast<float>(benchmark_trials)
              << " us" << std::endl;

    auto output_image3 = report2OpenMP::postprocessing(output_array3, orig_n_rows3, orig_n_cols3);

    // imshow("test", output_image);
    // waitKey(0);

    return 0;
}
