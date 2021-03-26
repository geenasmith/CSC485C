
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
    Mat input_image = imread("sampleset/frac8.png", IMREAD_COLOR);
    cout << "input_img" << input_image.cols << "x" << input_image.rows << endl;

    for (int i = 0; i < 6; i++) {

        Mat resized_image;
        resize(input_image, resized_image, Size(input_image.cols - i, input_image.rows - i), 0, 0);
        string name = format("sampleset/test/frac8-%dx%d.png", resized_image.cols, resized_image.rows);
        cout << "output: "<<name<< " "<< input_image.cols << "x" << input_image.rows << endl;
        imwrite(name, resized_image);
    }

    return 0;
}