#pragma once
#include "utils.hpp"

namespace Lab7
{
	void ex1()
	{
		const std::vector<std::string> imageNames = { "input\\lena.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = toGreyOpenCV(loadDiskImageSafe(imageNames[i]));

			
			std::vector<cv::Mat> imgVec;

			imgVec.push_back(putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 80, false), std::string("Low Pass - 80"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(applyPassFilter(greyLoadedMat, 80, true), std::string("High Pass - 80"), cv::Scalar(255, 255, 255)));
			imgVec.push_back(putTextBottom(applyBandFilter(greyLoadedMat, 25, 10, true), std::string("Band Pass - W = 25, D0 = 10"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(applyBandFilter(greyLoadedMat, 50, 50, false), std::string("Band Reject - W = 50, D0 = 50"), cv::Scalar(255, 255, 255)));

			auto sideBySide = chainImages(imgVec, CV_8UC1, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}

	void ex2()
	{
		const std::vector<std::string> imageNames = { "input\\nasa.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));

			auto spectrum = DFTOpenCVLogNorm(greyLoadedMat);

			
			std::vector<cv::Mat> imgVec;
			imgVec.push_back(putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(spectrum, std::string("Spectrum"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(applyBandFilter(greyLoadedMat, 126, 160, false), std::string("Band Reject"), cv::Scalar(0, 0, 0)));

			auto sideBySide = chainImages(imgVec, CV_32F, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}

	void ex3()
	{
		const std::vector<std::string> imageNames = { "input\\coperta1.jpg" };
		const cv::Size sz(512, 512);

		for (int i = 0; i < imageNames.size(); ++i)
		{
			auto greyLoadedMat = convertToGrey(loadDiskImageSafe(imageNames[i]));

			auto fourier = DFTOpenCV(greyLoadedMat);

			for (int i = 0; i < fourier.first.rows; ++i)
			{
				for (int j = 0; j < fourier.first.cols; ++j)
				{
					auto hUV = H(i + 1, j + 1);

					//std::cout << "before first: " << fourier.first.at<float>(i, j) << ", second: " << fourier.second.at<float>(i, j) << ", h = " << hUV.first << " " << hUV.second << std::endl;
					fourier.first.at<float>(i, j) /= hUV.first;
					fourier.second.at<float>(i, j) /= hUV.second;
					//std::cout << "after first: " << fourier.first.at<float>(i, j) << ", second: " << fourier.second.at<float>(i, j) << ", h = " << hUV.first << " " << hUV.second << std::endl;
					
				}
			}

			auto iftMat = IFTOpenCV(fourier);
			std::vector<cv::Mat> imgVec;
			imgVec.push_back(putTextBottom(greyLoadedMat, std::string("Original"), cv::Scalar(0, 0, 0)));
			imgVec.push_back(putTextBottom(iftMat, std::string("Reconstructed"), cv::Scalar(0, 0, 0)));

			auto sideBySide = chainImages(imgVec, CV_32F, sz);

			cv::namedWindow(WINDOW_NAME + imageNames[i]);
			cv::imshow(WINDOW_NAME + imageNames[i], sideBySide);
		}
	}
}