
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    Mat input_image = imread("sampleset/4000x2250.jpg", IMREAD_COLOR);
    cout << "input_img" << input_image.cols << "x" << input_image.rows << endl;

    // for (int i = 1; i < 9; i++) {

    //     Mat resized_image;
    //     resize(input_image, resized_image, Size((int)(4000 * i * 0.1),(int)(2250 * i * 0.1)), 0, 0);
    //     string name = format("sampleset/new-%d-%d.jpg", resized_image.cols, resized_image.rows);
    //     cout << "output: "<<name<< " "<< input_image.cols << "x" << input_image.rows << endl;
    //     imwrite(name, resized_image);
    // }


    Mat resized_image;

    resize(input_image, resized_image, Size(200,114), 0, 0);
    string name = format("sampleset/new-%d-%d.jpg", resized_image.cols, resized_image.rows);
    imwrite(name, resized_image);

    return 0;
}