#ifndef SUPERFRAME_H
#define SUPERFRAME_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

#include <MatchResult.h>

#define _DEBUG true
using namespace pcl;
class SuperFrame
{
public:
	SuperFrame()
	{
		img = cv::Mat(height_, width_, CV_8UC3);
		mask = cv::Mat::zeros(height_, width_, CV_8UC1);
	}
	bool getFeatures(PointCloud<PointXYZRGBA>::Ptr cloud, std::vector<cv::KeyPoint> &keypoints,cv::Mat &features);

	bool featuresMatch(SuperFrame &other, std::vector<cv::DMatch> &vis_matches);

	bool getImageFromCloud(PointCloud<PointXYZRGBA>::Ptr cloud, cv::Mat &img,cv::Mat &mask);

	inline bool correctMatch(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, cv::KeyPoint &keypoint);

	bool ransacFusion(SuperFrame &other, std::vector<cv::DMatch> vis_matches, MatchResult& res);

	bool computeInlierAndError(SuperFrame &other, Eigen::Matrix4f transform, std::vector<cv::DMatch> vis_matches, std::vector<cv::DMatch> &inliers, float &mse);

	Eigen::Matrix4f getTransformFromSamples(SuperFrame &other, std::vector<cv::DMatch>);

	bool match2SuperFrames(SuperFrame &other, MatchResult& res);

	PointCloud<PointXYZRGBA> cloud_;
	cv::Mat features_;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat img;
	cv::Mat mask;


	size_t frameid_;
	static const int width_ = 1920, height_ = 1080;
	static const float min_ratio;
	static const int max_match_num = 1000;
	static const int max_sample_iter = 10;
	static const int max_ransac_iter = 3000;
	static const float max_inlier_dis;
	static const int min_inlier_num = 40;

};



#endif