#pragma once
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>

#include "utils.hpp"

namespace Lab3
{
	void ex1()
	{
		const std::vector<std::string> imageNames = { "input\\imagine test.bmp", "input\\horizontal.bmp", "input\\vertical.bmp" , "input\\squares.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));
			auto myDCT = DCT(greyLoadedMat);
			auto openCVDCT = DCTOpenCV(greyLoadedMat);

			auto sideBySide = chainImages(std::vector<cv::Mat>{ greyLoadedMat, myDCT, openCVDCT }, CV_32F, cv::Size(256, 256));

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}

	void ex2()
	{
		const std::vector<std::string> imageNames = { "input\\imagine test.bmp", "input\\horizontal.bmp", "input\\vertical.bmp" , "input\\squares.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));
			auto myDCT = ICT(DCT(greyLoadedMat));
			auto openCVDCT = ICTOpenCV(DCTOpenCV(greyLoadedMat));

			auto sideBySide = chainImages(std::vector<cv::Mat>{ greyLoadedMat, myDCT, openCVDCT }, CV_32F, cv::Size(256, 256));

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}
}
