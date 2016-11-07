#pragma once
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>

#include "utils.hpp"

namespace Lab5
{
	void ex1()
	{
		const std::vector<std::string> imageNames = { "input\\circ4.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));

			cv::Mat meanFiltered, medianFiltered;

			cv::blur(greyLoadedMat, meanFiltered, cv::Size(3, 3));
			cv::medianBlur(greyLoadedMat, medianFiltered, 3);

			auto sideBySide = chainImages(
				std::vector<cv::Mat> {
					putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)),
					putTextBottom(meanFiltered, std::string("Mean"), cv::Scalar(0, 0, 0)),
					putTextBottom(medianFiltered, std::string("Median"), cv::Scalar(0, 0, 0))
			}, CV_32F, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}

	void ex2()
	{
		const std::vector<std::string> imageNames = { "input\\circ4.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));

			cv::Mat meanFiltered, kuwaharaFiltered = applyKuwahara(greyLoadedMat);

			cv::blur(greyLoadedMat, meanFiltered, cv::Size(3, 3));

			auto sideBySide = chainImages(
				std::vector<cv::Mat> {
					putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)),
					putTextBottom(meanFiltered, std::string("Mean"), cv::Scalar(0, 0, 0)),
					putTextBottom(kuwaharaFiltered, std::string("Kuwahara"), cv::Scalar(0, 0, 0))
			}, CV_32F, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}

	void ex3()
	{
		const std::vector<std::string> imageNames = { "input\\test.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));
			
			std::vector<cv::Mat> lowPasses;
			
			lowPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 5, false), std::string("Low Pass - 5"), cv::Scalar(0, 0, 0)));
			lowPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 15, false), std::string("Low Pass - 15"), cv::Scalar(0, 0, 0)));
			lowPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 30, false), std::string("Low Pass - 30"), cv::Scalar(0, 0, 0)));
			lowPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 80, false), std::string("Low Pass - 80"), cv::Scalar(0, 0, 0)));
			lowPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 230, false), std::string("Low Pass - 230"), cv::Scalar(0, 0, 0)));

			std::vector<cv::Mat> highPasses;
			highPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 5, true), std::string("High Pass - 5"), cv::Scalar(255, 255, 255)));
			highPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 15, true), std::string("High Pass - 15"), cv::Scalar(255, 255, 255)));
			highPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 30, true), std::string("High Pass - 30"), cv::Scalar(255, 255, 255)));
			highPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 80, true), std::string("High Pass - 80"), cv::Scalar(255, 255, 255)));
			highPasses.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 230, true), std::string("High Pass - 230"), cv::Scalar(255, 255, 255)));

			auto sideBySide1 = chainImages(lowPasses, CV_32F, sz);
			auto sideBySide2 = chainImages(highPasses, CV_32F, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i] + " - low passes");
			cv::imshow(WINDOW_NAME + imageNames[i] + " - low passes", sideBySide1);
			cv::namedWindow(WINDOW_NAME + imageNames[i] + " - high passes");
			cv::imshow(WINDOW_NAME + imageNames[i] + " - high passes", sideBySide2);
		}
	}
}
