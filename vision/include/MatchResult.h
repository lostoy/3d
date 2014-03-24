#ifndef MATCHRESULT_H
#define MATCHRESULT_H
#include <opencv2/opencv.hpp>]
#include <pcl/point_types.h>
class MatchResult
{
public:
	std::vector<cv::DMatch> inliers;
	
	float mse;
	Eigen::Matrix4f transform;
};
#endif