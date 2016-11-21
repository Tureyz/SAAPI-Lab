#pragma once
#include <string>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<cmath>

static const std::string WINDOW_NAME = "SAAPI ";

static const float EULER = 2.71828;
static const float PI = 3.14159;

void printPixelInfo(const cv::Mat &image)
{
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			auto val = image.at<cv::Vec3b>(i, j).val;
			std::cout << "Image[" << i << "][" << j << "]: (" << (int)val[0] << ", " << (int)val[1] << ", " << (int)val[2] << ")" << std::endl;
		}
	}
}

cv::Mat loadDiskImageSafe(std::string filePath)
{
	cv::Mat diskImage = cv::imread(filePath);

	if (!diskImage.data)
	{
		std::cout << "Eroare la deschidere imagine" << std::endl;
		return cv::Mat();
	}

	return diskImage;
}

cv::Mat showDiskImageSafe(std::string filePath, std::string windowSuffix)
{
	auto result = loadDiskImageSafe(filePath);

	if (result.size().width == 0)
		return cv::Mat();

	cv::imshow(WINDOW_NAME + windowSuffix, result);

	return result;
}

cv::Mat convertToGrey(const cv::Mat &coloredMat)
{
	cv::Mat result(coloredMat.rows, coloredMat.cols, CV_8UC1, cv::Scalar(0, 0, 0));

	for (int i = 0; i < coloredMat.rows; ++i)
	{
		for (int j = 0; j < coloredMat.cols; ++j)
		{
			auto coloredPixel = coloredMat.at<cv::Vec3b>(i, j).val;

			result.at<uchar>(i, j) = 0.3 * coloredPixel[2] + 0.59 * coloredPixel[1] + 0.11 * coloredPixel[0];
		}
	}

	result.convertTo(result, CV_32F, 1.0 / 255.0);
	return result;
}

cv::Mat toGreyOpenCV(const cv::Mat &coloredMat)
{
	cv::Mat result;
	cv::cvtColor(coloredMat, result, cv::COLOR_RGB2GRAY);
	return result;
}

cv::Mat computeGrayHistogram(const cv::Mat &input)
{
	cv::Mat result;
	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	/// Compute the histograms:
	calcHist(&input, 1, 0, cv::Mat(), result, 1, &histSize, &histRange, uniform, accumulate);

	return result;
}

cv::Mat computeHistogramImage(const cv::Mat &hist)
{
	// Draw the histograms for B, G and R
	int hist_w = 800;
	int hist_h = 600;
	int bin_w = cvRound((double)hist_w / 256);

	cv::Mat result(hist_h, hist_w, CV_8UC1, cv::Scalar(0, 0, 0));
	cv::Mat normalizedHist;
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, normalizedHist, 0, result.rows, cv::NORM_MINMAX);
	
	/// Draw for each channel
	for (int i = 0; i < 256; i++)
	{
		cv::line(result,
			cv::Point(bin_w * i, hist_h),
			cv::Point(bin_w * i, hist_h - cvRound(normalizedHist.at<float>(i))),
			cv::Scalar(255, 255, 255), 2, 8, 0);
	}

	return result;
}



std::pair<cv::Mat, cv::Mat> DFT(const cv::Mat &input)
{
	std::pair<cv::Mat, cv::Mat> result;
	cv::Mat realResult(input.rows, input.cols, CV_32F, cv::Scalar(0, 0, 0));
	cv::Mat imaginaryResult(input.rows, input.cols, CV_32F, cv::Scalar(0, 0, 0));

	const float M = input.rows;
	const float N = input.cols;
	const float factor = 1.f / (M * N);

	for (int u = 0; u < input.rows; ++u)
	{
		for (int v = 0; v < input.cols; ++v)
		{
			std::pair<float, float> partialSum(0.0, 0.0);

			for (int x = 0; x < M; ++x)
			{
				for (int y = 0; y < N; ++y)
				{
					float eulerX = PI * 2 * ((u * x / M) + (v * y / N));
					float rawPixelValue = input.at<float>(x, y) * 255.0;

					partialSum.first += rawPixelValue * cos(eulerX);
					partialSum.second -= rawPixelValue * sin(eulerX);
				}
			}

			partialSum.first *= factor;
			partialSum.second *= factor;

			realResult.at<float>(u, v) = partialSum.first;
			imaginaryResult.at<float>(u, v) = partialSum.second;
		}
	}

	result.first = realResult;
	result.second = imaginaryResult;

	return result;
}

float ck(const int k)
{
	return k == 0 ? (1.0 / sqrtf(2)) : 1;
}

cv::Mat DCT(const cv::Mat &input)
{
	cv::Mat result(input.rows, input.cols, CV_32F, cv::Scalar(0, 0, 0));;

	const float M = input.rows;
	const float N = input.cols;

	for (int u = 0; u < input.rows; ++u)
	{
		for (int v = 0; v < input.cols; ++v)
		{
			const float factor = (2 * ck(u) * ck(v)) / sqrtf(M * N);
			float partialSum = 0.0;

			for (int r = 0; r < M; ++r)
			{
				for (int s = 0; s < N; ++s)
				{
					float cosProduct = cos(((2 * r + 1) / (2 * M)) * u * PI) * cos(((2 * s + 1) / (2 * N)) * v * PI);
					float rawPixelValue = input.at<float>(r, s);

					partialSum += rawPixelValue * cosProduct;
				}
			}

			partialSum *= factor;

			result.at<float>(u, v) = partialSum;
		}
	}

	return result;
}

cv::Mat DCTOpenCV(const cv::Mat &input)
{
	cv::Mat result;
	cv::dct(input, result);
	return result;
}

cv::Mat ICT(const cv::Mat &input)
{
	cv::Mat result(input.rows, input.cols, CV_32F, cv::Scalar(0, 0, 0));;

	const float M = input.rows;
	const float N = input.cols;

	const float factor = 2.0 / (sqrtf(M * N));

	for (int r = 0; r < input.rows; ++r)
	{
		for (int s = 0; s < input.cols; ++s)
		{
			float partialSum = 0.0;

			for (int u = 0; u < M; ++u)
			{
				for (int v = 0; v < N; ++v)
				{
					float cosProduct = cos(((2 * r + 1) / (2 * M)) * u * PI) * cos(((2 * s + 1) / (2 * N)) * v * PI);
					float cProduct = ck(u) * ck(v);
					float rawPixelValue = input.at<float>(u, v);

					partialSum += cProduct * rawPixelValue * cosProduct;
				}
			}

			partialSum *= factor;

			result.at<float>(r, s) = partialSum;
		}
	}

	return result;
}

cv::Mat ICTOpenCV(const cv::Mat &input)
{
	cv::Mat result;
	cv::dct(input, result, cv::DCT_INVERSE);
	return result;
}

cv::Mat DFTLogNorm(const cv::Mat &input)
{
	auto fourier = DFT(input);

	cv::Mat result;
	magnitude(fourier.first, fourier.second, result);

	result += cv::Scalar::all(1);
	log(result, result);

	int cx = result.cols / 2;
	int cy = result.rows / 2;

	cv::Mat q0(result, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(result, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(result, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(result, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(result, result, 0, 1, CV_MINMAX);
	return result;
}

std::pair<cv::Mat, cv::Mat> DFTOpenCV(const cv::Mat &input)
{	
	cv::Mat asd;
	input.convertTo(asd, CV_32F);
	cv::Mat planes[] = { cv::Mat_<float>(asd), cv::Mat::zeros(asd.size(), CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);

	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	return std::make_pair(planes[0], planes[1]);
}

cv::Mat DFTOpenCVLogNorm(const cv::Mat &input)
{
	auto fourier = DFTOpenCV(input);

	
	magnitude(fourier.first, fourier.second, fourier.first);


	cv::Mat magI = fourier.first;
	magI += cv::Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);

	return magI;
}

cv::Mat IFT(const std::pair<cv::Mat, cv::Mat> &input)
{
	cv::Mat result(input.first.rows, input.first.cols, CV_32F, cv::Scalar(0, 0, 0));

	auto M = input.first.rows;
	auto N = input.second.cols;

	for (int x = 0; x < M; ++x)
	{
		for (int y = 0; y < N; ++y)
		{
			std::pair<float, float> partialSum(0.0, 0.0);
			for (int u = 0; u < M; ++u)
			{
				for (int v = 0; v < N; ++v)
				{
					auto eulerX = PI * 2 * ((1.0 * u * x / M) + (1.0 * v * y / N));

					partialSum.first += input.first.at<float>(u, v) * cos(eulerX);
					partialSum.second += input.second.at<float>(u, v) * sin(eulerX);
				}
			}

			result.at<float>(x, y) = partialSum.first - partialSum.second;
		}
	}

	cv::normalize(result, result, 0, 1, CV_MINMAX);
	return result;
}

cv::Mat IFTOpenCV(const std::pair<cv::Mat, cv::Mat> &input)
{
	cv::Mat result;
	cv::Mat combined;
	cv::Mat arr[2] = { input.first, input.second };
	merge(arr, 2, combined);

	cv::dft(combined, combined, cv::DFT_INVERSE | cv::DFT_SCALE);

	cv::split(combined, arr);

	arr[0].copyTo(result);
	return result;
}

cv::Mat chainImages(const std::vector<cv::Mat> &imgVec, const int matType, const cv::Size individualResize)
{
	if (imgVec.empty())
		return cv::Mat();

	std::vector<cv::Mat> resizedVec;
	for (auto img : imgVec)
	{
		cv::Mat resizedImg;
		cv::resize(img, resizedImg, individualResize);
		resizedVec.push_back(resizedImg);
	}

	cv::Mat result(resizedVec[0].size().height, resizedVec[0].size().width * resizedVec.size(), matType);

	for (int i = 0; i < resizedVec.size(); ++i)
	{
		resizedVec[i].copyTo(result(cv::Rect(i * resizedVec[i].size().width, 0, resizedVec[i].size().width, resizedVec[i].size().height)));
	}

	return result;
}

cv::Mat computeCumHist(const cv::Mat &hist)
{
	cv::Mat result = hist.clone();

	for (int i = 1; i < 256; ++i)
	{
		result.at<float>(i) = hist.at<float>(i) + result.at<float>(i - 1);
	}

	return result;
}

int searchNearest(const cv::Mat &arr, const float key)
{
	float value = fabsf(key - arr.at<float>(0));
	int num = 0;

	for (int x = 0; x < arr.rows; x++)
	{
		if (value > fabsf(key - arr.at<float>(x)))
		{
			value = fabsf(key - arr.at<float>(x));
			num = x;
		}
	}

	return num;

}

std::vector<cv::Mat> splitIntoRegions(cv::Mat input)
{
	cv::Mat region1(input, cv::Rect(0, 0, 3, 3));
	cv::Mat region2(input, cv::Rect(0, 2, 3, 3));
	cv::Mat region3(input, cv::Rect(2, 0, 3, 3));
	cv::Mat region4(input, cv::Rect(2, 2, 3, 3));

	return std::vector<cv::Mat> { region1, region2, region3, region4 };
}

cv::Mat getRegionAround(cv::Mat input, std::pair<int, int> centerPixel, int regionSize)
{
	return cv::Mat(input, cv::Rect(centerPixel.first - regionSize, centerPixel.second - regionSize, regionSize * 2 + 1, regionSize * 2 + 1));
}

int getMean(const cv::Mat &input)
{
	int result = 0;
	int cnt = 0;

	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			result += input.at<uchar>(i, j);
			cnt++;
		}
	}

	return result / cnt;
}

int getGeomMean(const cv::Mat &input)
{
	uint64_t result = 1;
	int cnt = 0;
	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			result *= input.at<uchar>(i, j);
			cnt++;
		}
	}

	return std::pow(result, 1.f / float(cnt));
}

int getContraHarm(const cv::Mat &input, const float Q)
{
	float result1 = 0;
	float result2 = 0;

	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			result1 += std::pow((float) input.at<uchar>(i, j), Q + 1.f);
			result2 += std::pow((float) input.at<uchar>(i, j), Q);
		}
	}
	
	return result1 / result2;
}

int getContraHarmPos(const cv::Mat &input)
{
	return getContraHarm(input, 1.5f);
}

int getContraHarmNeg(const cv::Mat &input)
{
	return getContraHarm(input, -1.5f);
}

cv::Mat applyGenericAverageFilter(const cv::Mat &input, int(filterFunction)(const cv::Mat &))
{
	cv::Mat result = input.clone();

	for (int i = 2; i < input.cols - 2; ++i)
	{
		for (int j = 2; j < input.rows - 2; ++j)
		{
			auto subMat = getRegionAround(input, std::make_pair(i, j), 1);
			subMat.convertTo(subMat, CV_8UC1);
			result.at<uchar>(j, i) = filterFunction(subMat);
		}
	}

	return result;
}



cv::Mat applyKuwahara(const cv::Mat &input)
{

	cv::Mat result = input.clone();

	for (int i = 2; i < input.cols - 2; ++i)
	{
		for (int j = 2; j < input.rows - 2; ++j)
		{
			auto subMat = getRegionAround(input, std::make_pair(i, j), 2);
			auto spl = splitIntoRegions(subMat);

			auto pixelValue = input.at<float>(j, i);
			auto minStdDev = cv::Scalar(10000000);
			for (auto region : spl)
			{
				cv::Scalar mean, stdDev;

				cv::meanStdDev(region, mean, stdDev);

				if (*stdDev.val < *minStdDev.val)
				{
					minStdDev = stdDev;
					pixelValue = *mean.val;
				}
			}

			result.at<float>(j, i) = pixelValue;
		}
	}

	return result;
}

cv::Mat applyPassFilter(const cv::Mat &input, const int filterCutoff, bool which)
{
	
	auto fourier = DFTOpenCV(input);

	for (int i = 0; i < fourier.first.rows; ++i)
	{
		for (int j = 0; j < fourier.first.cols; ++j)
		{
			auto dUV = sqrtf(i * i + j * j);

			auto realValue = fourier.first.at<float>(i, j);
			auto imaginaryValue = fourier.second.at<float>(i, j);

			fourier.first.at<float>(i, j) = which ? (dUV > filterCutoff ? realValue : 0) : (dUV < filterCutoff ? realValue : 0);
			fourier.second.at<float>(i, j) = which ? (dUV > filterCutoff ? imaginaryValue : 0) : (dUV < filterCutoff ? imaginaryValue : 0);
		}
	}

	return IFTOpenCV(fourier);
}

// false = band reject, true = band pass
cv::Mat applyBandFilter(const cv::Mat &input, const float W, const float D0, bool which)
{
	auto fourier = DFTOpenCV(input);

	for (int i = 0; i < fourier.first.rows; ++i)
	{
		for (int j = 0; j < fourier.first.cols; ++j)
		{
			auto dUV = sqrtf(i * i + j * j);

			auto realValue = fourier.first.at<float>(i, j);
			auto imaginaryValue = fourier.second.at<float>(i, j);
			auto hUV = (dUV < D0 - W / 2.f || dUV > D0 + W / 2.f) ? 1 : 0;

			hUV = which ? 1 - hUV : hUV;

			fourier.first.at<float>(i, j) = realValue * hUV;
			fourier.second.at<float>(i, j) = imaginaryValue * hUV;
		}
	}

	return IFTOpenCV(fourier);
}

cv::Mat putTextBottom(const cv::Mat &input, const std::string text, const cv::Scalar color)
{
	cv::Mat result = input.clone();

	cv::putText(result, text, cv::Point(30, input.cols - 30), cv::FONT_HERSHEY_PLAIN, 1.4, color, 2);

	return result;
}

std::pair<double, double> H(float u, float v, float a, float b, float T)
{

	const float piFact = PI * (u * a + v * b);
	const float factor = ((T / (piFact)) * sinf(piFact));
		
	return std::make_pair(factor * cosf(piFact), factor * (-sinf(piFact)));
}

std::pair<double, double>  H(float u, float v)
{
	return H(u, v, 0.1, 0.1, 1);
}
