
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <stdlib.h>

using namespace cv;
using namespace std;


int main()
{
    Mat input_image = imread("images/earth.jpg", IMREAD_COLOR);
    cout << "input_img" << input_image.cols << "x" << input_image.rows << endl;

    int s [10] = {
        10000,
        7071,
        5000,
        3535,
        2500,
        1767,
        1250,
        883,
        441,
        220};

    for (int i = 0; i < 10; i++) {

        Mat resized_image;
        resize(input_image, resized_image, Size(s[i],s[i]), 0, 0);
        string name = format("images/earth_set/earth-%d-%d.jpg", resized_image.cols, resized_image.rows);
        cout << "output: "<<name<< " "<< input_image.cols << "x" << input_image.rows << endl;
        imwrite(name, resized_image);

    }


    // Mat resized_image;

    // resize(input_image, resized_image, Size(200,114), 0, 0);
    // string name = format("inputs/new-%d-%d.jpg", resized_image.cols, resized_image.rows);
    // imwrite(name, resized_image);

    return 0;
}