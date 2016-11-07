#pragma once
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>

#include "utils.hpp"

namespace Lab4
{
	void ex1()
	{
		const std::vector<std::string> imageNames = { "input\\rand.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = toGreyOpenCV(loadDiskImageSafe(imageNames[i]));		
			auto histImage = computeHistogramImage(computeGrayHistogram(greyLoadedMat));

			cv::imshow(WINDOW_NAME + imageNames[i], greyLoadedMat);
			cv::imshow(WINDOW_NAME + imageNames[i] + " - histogram", histImage);
		}
	}

	void ex2()
	{
		const std::vector<std::string> imageNames = { "input\\rand.bmp" };
		const cv::Size sz(256, 256);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = toGreyOpenCV(loadDiskImageSafe(imageNames[i]));
			auto histImage = computeHistogramImage(computeGrayHistogram(greyLoadedMat));

			cv::Mat equalizedMat;
			cv::equalizeHist(greyLoadedMat, equalizedMat);
			auto equalizedHistImage = computeHistogramImage(computeGrayHistogram(equalizedMat));
			cv::imshow(WINDOW_NAME + imageNames[i], greyLoadedMat);
			cv::imshow(WINDOW_NAME + imageNames[i] + " - equalized", equalizedMat);
			cv::imshow(WINDOW_NAME + imageNames[i] + " - histogram", histImage);
			cv::imshow(WINDOW_NAME + imageNames[i] + " - equalized histogram", equalizedHistImage);
		}
	}

	void ex3()
	{
		auto inputGreyLoadedMat = toGreyOpenCV(loadDiskImageSafe("input\\trump.bmp"));
		auto desiredGreyLoadedMat = toGreyOpenCV(loadDiskImageSafe("input\\rand.bmp"));

		auto inputHist = computeGrayHistogram(inputGreyLoadedMat);
		auto desiredHist = computeGrayHistogram(desiredGreyLoadedMat);

		
		cv::normalize(inputHist, inputHist, 1.0);
		cv::normalize(desiredHist, desiredHist, 1.0);

		cv::Mat cumDesired = computeCumHist(desiredHist);
		cv::Mat cumInput = computeCumHist(inputHist);
		cv::normalize(cumDesired, cumDesired, 1.0);
		cv::normalize(cumInput, cumInput, 1.0);
		//cv::imshow(WINDOW_NAME + " - asd histogram", computeHistogramImage(cumInput));
		//cv::imshow(WINDOW_NAME + " - dsa histogram", computeHistogramImage(cumDesired));

		std::vector<int> pixelMap(256, 0);

		for (int i = 0; i < 256; ++i)
		{
			pixelMap[i] = searchNearest(cumDesired, (float)cumInput.at<float>(i));
		}

		cv::Mat outimg(inputGreyLoadedMat.rows, inputGreyLoadedMat.cols, CV_8UC1);

		for (int x = 0; x < inputGreyLoadedMat.rows; ++x)
		{
			for (int y = 0; y < inputGreyLoadedMat.cols; ++y)
			{
				outimg.at<uchar>(x, y) = ((uchar)pixelMap[(int)inputGreyLoadedMat.at<uchar>(x, y)]);
			}
		}

		cv::imshow(WINDOW_NAME + " - original", inputGreyLoadedMat);
		cv::imshow(WINDOW_NAME + " - output", outimg);
		cv::imshow(WINDOW_NAME + " - initial histogram", computeHistogramImage(inputHist));
		cv::imshow(WINDOW_NAME + " - target histogram", computeHistogramImage(desiredHist));
		cv::imshow(WINDOW_NAME + " - output histogram", computeHistogramImage(computeGrayHistogram(outimg)));
	}
}