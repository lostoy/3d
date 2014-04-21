#ifndef MATCHRESULT_H
#define MATCHRESULT_H
#include <opencv2/opencv.hpp>]
#include <pcl/point_types.h>
class MatchResult
{
public:
	//mse, the mean squared error of the visual fusion
	MatchResult()
	{
		mse = 1e10;
		transform = Eigen::Matrix4f::Identity();
	}
	std::vector<cv::DMatch> inliers;
	
	float mse;
	Eigen::Matrix4f transform;
};
#endif