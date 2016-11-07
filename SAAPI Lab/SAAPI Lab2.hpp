#pragma once
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>

#include "utils.hpp"

//static const std::string WINDOW_NAME = "Fereastra Lab2";

namespace Lab2
{

	void ex12()
	{

		const std::vector<std::string> imageNames = { "input\\imagine test.bmp", "input\\horizontal.bmp", "input\\vertical.bmp" , "input\\squares.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));
			auto myFourier = DFTLogNorm(greyLoadedMat);
			auto openCVFourier = DFTOpenCVLogNorm(greyLoadedMat);

			auto sideBySide = chainImages(std::vector<cv::Mat>{ greyLoadedMat, myFourier, openCVFourier }, CV_32F, cv::Size(256, 256));

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}

	void ex34()
	{

		const std::vector<std::string> imageNames = { "input\\imagine test.bmp", "input\\horizontal.bmp", "input\\vertical.bmp" , "input\\squares.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));
			auto myFourier = IFT(DFT(greyLoadedMat));
			auto openCVFourier = IFTOpenCV(DFTOpenCV(greyLoadedMat));

			auto sideBySide = chainImages(std::vector<cv::Mat>{ greyLoadedMat, myFourier, openCVFourier }, CV_32F, cv::Size(256, 256));

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}
}