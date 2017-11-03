#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\videoio\videoio.hpp"
#include "opencv2\imgcodecs\imgcodecs.hpp"
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <vector>
#include <Windows.h>
#include <iostream>
#include <math.h>
#include "stdafx.h"

using namespace cv;
using namespace std;

void sobel(Mat &input);
void flips(Mat &input, bool display, bool save);
void blur(Mat &input, int blurSize, bool display, bool save);

int main() {
	/*
	//create a black 256x256, 8bit, gray scale image in a matrix container
	Mat image(256, 256, CV_8UC1, Scalar(0));
	//draw white text HelloOpenCV!
	putText(image, "HelloOpenCV!", Point(70, 70),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255), 1, CV_AA);

	*/

	Mat image;
	image = imread("coins3.png", 1);

	if (!image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	
	//sobel(image);
	flips(image,true,false);
	blur(gray_image, 2, false, false);
	gray_image.release();
	image.release();
	return 0;
}

void sobel(Mat &input) {

	//initialise x-gradient image
	Mat dX;
	dX.create(input.size(), input.type());

	//initialise Y-gradient image
	Mat dY;
	dY.create(input.size(), input.type());

	//initialise gradient magnitude image
	Mat gradientMagnitude;
	gradientMagnitude.create(input.size(), input.type());

	//initialise gradient direction image
	Mat gradientDirection;
	gradientDirection.create(input.size(), input.type());

	//df/dx gradient kernel
	int xKernel[3][3] = { { -1, 0, 1 }, { -2,0, 2 }, { -1, 0, 1 } };
	//int yKernel[3][3] = { { -1,-2,-1 }, { 0, 0, 0 }, { 1, 2, 1 } };
	int yKernel[3][3] = { { 1,1,1 },{ 1, 1, 1 },{ 1, 1, 1 } };

	//intensity
	float intensityX;
	float intensityY;

	//loop over image pixels
	for (int x = 1; x < input.rows - 1; x++) {
		for (int y = 1; y < input.cols - 1; y++) {

			float totalX = 0.0;
			float totalY = 0.0;

			//loop over kernel values
			for (int m = -1; m < 2; m++) {
				for (int n = -1; n < 2; n++) {

					//multiply intensity of pixel by kernel value
					intensityX = input.at<Vec3b>(x - m, y - n)[0];
					totalX += intensityX*xKernel[m + 1][n + 1];

					intensityY = input.at<Vec3b>(x - m, y - n)[0];
					totalY += intensityY*yKernel[m + 1][n + 1];

				}
			}

			dX.at<uchar>(x, y) = totalX;
			dY.at<uchar>(x, y) = totalY;

			gradientMagnitude.at<uchar>(x, y) = sqrt(totalX*totalX + totalY*totalY);
			gradientDirection.at<uchar>(x, y) = atan(totalY / totalX);
		}
	}

	//cv::imwrite("dX.png", dX);
	//cv::imwrite("dY.png", dY);
	//cv::imwrite("gradientMagnitude.png", gradientMagnitude);
	//cv::imwrite("gradientDirection.png", gradientDirection);

	dX.release();
	dY.release();
	gradientMagnitude.release();
	gradientDirection.release();

}

void flips(Mat &input, bool display, bool save) {

	Mat flipped(input.size(), input.type());

	for (int x = 0; x < input.rows; x++) {
		for (int y = 0; y < input.cols; y++) {			
			flipped.at<Vec3b>(x, y) = input.at<Vec3b>(x, input.cols - y);
		}
	}

	if (save) {
		cv::imwrite("Flipped.png", flipped);
	}

	if (display) {
		cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
		cv::imshow("Original Image", input);
		cv::namedWindow("Flipped Image", CV_WINDOW_AUTOSIZE);
		cv::imshow("Flipped Image", flipped);		
		cv::waitKey(0);
	}	

	flipped.release();
}

void blur(Mat &input, int blurSize, bool display, bool save) {

	Mat blurredImage(input.size(), input.type());
	float total;
	int maskDim = (1 + 2 * blurSize);
	
	for (int x = blurSize; x < input.rows - blurSize; x++) {
		for (int y = blurSize; y < input.cols - blurSize; y++) {
			total = 0;
			
			for (int i = -blurSize; i <= blurSize; i++) {
				for (int j = -blurSize; j <= blurSize; j++) {
					total += (float)input.at<uchar>(x+i, y+j);
				}
			}
			
			blurredImage.at<uchar>(x, y) = (int)( total / (maskDim*maskDim) );
		}
	}

	if (save) {
		cv::imwrite("Blurred.png", blurredImage);
	}

	if (display) {
		cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
		cv::imshow("Original Image", input);

		cv::namedWindow("Blurred", CV_WINDOW_AUTOSIZE);
		cv::imshow("Blurred", blurredImage);

		cv::waitKey(0);
	}	

	blurredImage.release();

}