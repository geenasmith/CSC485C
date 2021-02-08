#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

/*
compile: g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv`
*/

void preprocessing()
{
}

void sobel()
{
}

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cout << "Usage: sobel <input image" << std::endl;
        return 0;
    }

    // read in image as grayscale
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return 1;
    }

    // gaussian filter to remove noise
    GaussianBlur(image, image, Size(3, 3), 0, 0);

    imshow("Display window", image);
    int k = waitKey(0);

    // pad image with 1 px of 0s

    // convert to vector of vectors

    // loop through calculating G_x and G_y
    // mag is sqrt(G_x^2 + G_y^2)

    // int g_x[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    // int g_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // normalize to 0-255

    // save edge detection

    return 0;
}