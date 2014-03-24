#include <SuperFrame.h>

bool SuperFrame::getImageFromCloud(PointCloud<PointXYZRGBA>::Ptr cloud, cv::Mat &img,cv::Mat &mask)
{
	if (img.data&&mask.data)
		return true;
	img = cv::Mat(height_, width_, CV_8UC3);
	mask = cv::Mat::zeros(height_, width_, CV_8UC1);
	int ind=0;
	for (int i = 0; i < width_; i++)
	{

		for (int j = 0; j < height_; j++)
		{
			img.at<uchar>(i, j, 0) = cloud->points[ind].g;
			img.at<uchar>(i, j, 1) = cloud->points[ind].b;
			img.at<uchar>(i, j, 2) = cloud->points[ind].r;

			mask.at<uchar>(i, j) = pcl_isfinite(cloud->points[ind].x);
			ind++;
		}
	}
	return true;
}
bool SuperFrame::getFeatures(PointCloud<PointXYZRGBA>::Ptr cloud, std::vector<cv::KeyPoint> &keypoints, cv::Mat &features)
{
	getImageFromCloud(cloud, img,mask);

	//convert to gray image and upload {mask,img_gray} to gpu
	cv::gpu::GpuMat img_gpu,mask_gpu;
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_RGB2GRAY);
	img_gpu.upload(img_gray);
	mask_gpu.upload(mask);

	//compute SURF features for img_gray
	cv::gpu::SURF_GPU surf;
	cv::gpu::GpuMat keypoints_gpu,img_desc_gpu;
	surf.upright = 1;
	surf(img_gpu, mask_gpu, keypoints_gpu, img_desc_gpu);
	surf.downloadKeypoints(keypoints_gpu, keypoints);
	img_desc_gpu.download(features);
	return true;
}

bool SuperFrame::correctMatch(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, cv::KeyPoint &keypoint)
{
	int px = keypoint.pt.x, py = keypoint.pt.y;
	if (pcl_isfinite(cloud->points[py*width_ + px].x))
		return true;
	for (int tx = std::max(px - 1, 0); tx < std::min(px + 1, width_); tx++)
	{
		for (int ty = std::max(py - 1, 0); ty < std::min(py + 1, height_); ty++)
		{
			if (pcl_isfinite(cloud->points[ty*width_ + tx].x))
			{
				keypoint.pt.x = tx;
				keypoint.pt.y = ty;
				return true;
			}
		}
	}
	return false;
}
bool SuperFrame::featuresMatch(SuperFrame &other, std::vector<cv::DMatch> vis_matches)
{
	//initial matches with 2nn_flann_matcher
	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> initial_matches;
	matcher.knnMatch(features_, other.features_, initial_matches,2);

	//prune mathces by valid correction and ratio test
	vis_matches.clear();
	for (int i = 0; i < initial_matches.size(); i++)
	{
		if (initial_matches[i][1].distance / initial_matches[i][0].distance < min_ratio)
		{
			continue;
		}
		
		//valid correction
		if (correctMatch(cloud_.makeShared(), keypoints[initial_matches[i][0].queryIdx]) && \
			correctMatch(other.cloud_.makeShared(), other.keypoints[initial_matches[i][0].trainIdx]))
		{
			vis_matches.push_back(initial_matches[i][0]);
		}
	}

	//return only the first max_match_num matches
	sort(vis_matches.begin(), vis_matches.end());
	vis_matches = std::vector<cv::DMatch>(vis_matches.begin(), vis_matches.begin() + max_match_num);
	
	return true;
}

bool get3Samples(std::vector<cv::DMatch> cand_matches, std::vector<cv::DMatch> &sample_matches)
{
	if (cand_matches.size() < 3)
		return false;
	
	
	int idx1, idx2, idx3;
	
	idx1 = std::min(rand() % cand_matches.size(), rand() % cand_matches.size());

	int iter = 0;
	do
	{
		idx2 = std::min(rand() % cand_matches.size(), rand() % cand_matches.size());
		iter++;
	} while (iter<SuperFrame::max_sample_iter&&idx2 == idx1);

	iter = 0;
	do
	{
		idx3 = std::min(rand() % cand_matches.size(), rand() % cand_matches.size());
	} while (iter<SuperFrame::max_sample_iter&&(idx3 == idx2 || idx3 == idx1));

	if (idx1 != idx2&&idx1 != idx3&&idx2 != idx3)
	{

		sample_matches.push_back(cand_matches[idx1]);
		sample_matches.push_back(cand_matches[idx2]);
		sample_matches.push_back(cand_matches[idx3]);
		return true;
	}
	else
		return false;
}

Eigen::Matrix4f SuperFrame::getTransformFromSamples(SuperFrame &other, std::vector<cv::DMatch> samples)
{
	Eigen::Matrix<float, 3, Eigen::Dynamic> points_query(3, samples.size()), points_train(3, samples.size());
	for (int i = 0; i < samples.size(); i++)
	{
		//map from matches->keypoints->cloud
		cv::KeyPoint keypoint_query;
		points_query.col(i) = cloud_.points[keypoint_query.pt.y*width_ + keypoint_query.pt.x].getVector3fMap();
		cv::KeyPoint keypoint_train;
		points_train.col(i) = other.cloud_.points[keypoint_train.pt.y*width_ + keypoint_train.pt.x].getVector3fMap();
	}
	Eigen::Matrix4f transform=Eigen::umeyama(points_query, points_train, false);
	
	return transform;
}

bool SuperFrame::computeInlierAndError(SuperFrame &other, Eigen::Matrix4f transform, std::vector<cv::DMatch> vis_matches, std::vector<cv::DMatch> &inliers, float &mse)
{
	inliers.clear();
	
	float mean = 0.0;
	int n = 0;
	float M2 = 0.0;

	for (int i = 0; i < vis_matches.size(); i++)
	{
		//map from mathces->keypoint->cloud
		cv::KeyPoint keypoint_query = keypoints[vis_matches[i].queryIdx];
		Eigen::Vector4f p_query = cloud_.points[keypoint_query.pt.y*width_ + keypoint_query.pt.x].getVector4fMap();

		cv::KeyPoint keypoint_train = other.keypoints[vis_matches[i].trainIdx];
		Eigen::Vector4f p_train = other.cloud_.points[keypoint_train.pt.y*width_ + keypoint_train.pt.x].getVector4fMap();

		float err = (transform*p_query - p_train).norm();
		if (err < max_inlier_dis*max_inlier_dis)
		{
			n++;
			inliers.push_back(vis_matches[i]);
			float delta = err - mean;
			mean += delta / n;
			M2 += delta*(err - mean);
		}
	}
	mse = M2 / n;
	return true;

}
bool SuperFrame::ransacFusion(SuperFrame &other, std::vector<cv::DMatch> vis_matches, MatchResult& res)
{
	if (vis_matches.size() == 0)
		return false;

	//get samples from candadite visual matches
	
	for (int iter = 0; iter < max_ransac_iter; iter++)
	{
		MatchResult tmp_res;
		std::vector<cv::DMatch> sample_matches;
		if (get3Samples(vis_matches, sample_matches))
		{
			//initial result
			tmp_res.transform = getTransformFromSamples(other, sample_matches);
			computeInlierAndError(other, tmp_res.transform, vis_matches,tmp_res.inliers, tmp_res.mse);
			
			//refined result
			for (int inner_iter = 0; inner_iter < 10; inner_iter++)
			{
				tmp_res.transform=getTransformFromSamples(other, tmp_res.inliers);
				computeInlierAndError(other, tmp_res.transform, vis_matches,tmp_res.inliers, tmp_res.mse);
			}
			
		}
		//update new result
		if (tmp_res.inliers.size()>res.inliers.size() || (tmp_res.inliers.size() == res.inliers.size() && tmp_res.mse < res.mse))
			res = tmp_res;
		
	}
}
