#pragma once
#include <string>
#include <vector>
#include "utils.hpp"
#include <unordered_map>

namespace Lab6
{
	void ex1()
	{
		const std::string baseImageName("input\\test1");
		const std::vector<std::string> imageNames = { baseImageName + "a.jpg", baseImageName + "b.jpg", baseImageName + "c.jpg", baseImageName + "d.jpg", baseImageName + "e.jpg", baseImageName + "f.jpg" };

		const std::unordered_map<int, std::string> noiseMap = { {0 , "Gaussian"}, {1, "Rayleigh"}, {2, "Gamma"}, {3, "Exponential"}, {4, "Uniform"}, {5, "Salt & Pepper"} };

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto initialImage = loadDiskImageSafe(imageNames[i]);
			auto greyLoadedMat = toGreyOpenCV(initialImage);
			auto hist = computeGrayHistogram(greyLoadedMat);
			auto histImage = computeHistogramImage(hist);

			auto sideBySide = chainImages(std::vector<cv::Mat>{ putTextBottom(greyLoadedMat, noiseMap.at(i), cv::Scalar(255, 255, 255)), histImage }, CV_8UC1, cv::Size(256, 256));

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);			
		}
	}

	void ex2()
	{

		const std::vector<std::string> imageNames = { "input\\circ1.jpg"};
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto initialImage = loadDiskImageSafe(imageNames[i]);
			auto greyLoadedMat = toGreyOpenCV(initialImage);

			auto sideBySide = chainImages(std::vector<cv::Mat>{ putTextBottom(greyLoadedMat, "Original", cv::Scalar(255, 255, 255)),
				putTextBottom(applyGenericAverageFilter(greyLoadedMat, getMean), "Mean", cv::Scalar(255, 255, 255)),
				putTextBottom(applyGenericAverageFilter(greyLoadedMat, getGeomMean), "Geometric", cv::Scalar(255, 255, 255))}, CV_8UC1, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}

	void ex3()
	{
		const std::vector<std::string> imageNames = { "input\\circ2.jpg", "input\\circ3.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto initialImage = loadDiskImageSafe(imageNames[i]);
			auto greyLoadedMat = toGreyOpenCV(initialImage);

			auto sideBySide = chainImages(std::vector<cv::Mat>{ putTextBottom(greyLoadedMat, "Original", cv::Scalar(255, 255, 255)),
				putTextBottom(applyGenericAverageFilter(greyLoadedMat, getContraHarmPos), "Q = 1.5", cv::Scalar(255, 255, 255)),
				putTextBottom(applyGenericAverageFilter(greyLoadedMat, getContraHarmNeg), "Q = -1.5", cv::Scalar(255, 255, 255))}, CV_8UC1, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);

			auto x = sideBySide.size().width * (i % (1920 / sideBySide.size().width));
			auto y = (sideBySide.size().height + 32) * (i / (1920 / sideBySide.size().width));
			cv::moveWindow(WINDOW_NAME + imageNames[i], x, y);
		}
	}
	
	void ex4()
	{
		const std::vector<std::string> imageNames = { "input\\circ4.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = toGreyOpenCV(loadDiskImageSafe(imageNames[i]));

			cv::Mat meanFiltered, medianFiltered;

			cv::blur(greyLoadedMat, meanFiltered, cv::Size(3, 3));
			cv::medianBlur(greyLoadedMat, medianFiltered, 3);

			auto sideBySide = chainImages(
				std::vector<cv::Mat> {
				putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)),
					putTextBottom(meanFiltered, std::string("Mean"), cv::Scalar(0, 0, 0)),
					putTextBottom(medianFiltered, std::string("Median"), cv::Scalar(0, 0, 0))
			}, CV_8UC1, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}
}