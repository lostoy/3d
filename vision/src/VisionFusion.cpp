#include "PointCloudStreamer.hpp"

// this is the main registration code, a little clusterd and unorganized.
bool getSamples(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pre_cloud_, std::vector<int> &samples, int maxiter, float threshold)
{
	// We're assuming that indices_ have already been set in the constructor
	if (cloud_->size() < 3 || cloud_->size() != pre_cloud_->size())
		return false;

	samples.resize(3);
	double trand = cloud_->size() / (RAND_MAX + 1.0);


	// Get a random number between 1 and max_indices
	int idx = (int)(rand() * trand);
	// Get the index
	samples[0] = idx;

	// Get a second point which is different than the first
	Eigen::Vector4f p1p0, p2p0, p2p1;

	int iter = 0;
	do
	{
		int iter2 = 0;
		do
		{
			idx = (int)(rand() * trand);
			samples[1] = idx;
			++iter2;
			if (iter2 > maxiter)
				break;
		} while (samples[1] == samples[0]);

		// Get the values at the two points
		pcl::Vector4fMap p0 = cloud_->points[samples[0]].getVector4fMap();
		pcl::Vector4fMap p1 = cloud_->points[samples[1]].getVector4fMap();

		// Compute the segment values (in 3d) between p1 and p0
		p1p0 = p1 - p0;
		++iter;
		if (iter > maxiter)
		{
			return false;
		}
	} while (p1p0.squaredNorm() <= threshold*threshold);

	int iter1 = 0;
	do
	{
		int iter2 = 0;
		do
		{
			// Get the third point, different from the first two
			int iter3 = 0;
			do
			{
				idx = (int)(rand() * trand);
				samples[2] = idx;
				++iter3;
				if (iter3 > maxiter)
					return false;
			} while ((samples[2] == samples[1]) || (samples[2] == samples[0]));

			pcl::Vector4fMap p0 = cloud_->points[samples[0]].getVector4fMap();
			pcl::Vector4fMap p1 = cloud_->points[samples[1]].getVector4fMap();
			pcl::Vector4fMap p2 = cloud_->points[samples[2]].getVector4fMap();

			// Compute the segment values (in 3d) between p2 and p0
			p2p0 = p2 - p0;
			p2p1 = p2 - p1;


			++iter2;
			if (iter2 > maxiter)
			{

				return false;
			}
		} while (p2p0.dot(p1p0) / p2p0.norm() / p1p0.norm()>0.86);

		++iter1;
		if (iter1 > maxiter)
		{

			return false;
		}
	} while (p2p0.squaredNorm() < threshold*threshold || p2p1.squaredNorm() < threshold*threshold);
	return true;
}


void countInliers(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pre_cloud_, Eigen::Matrix4f transform, float threshold, std::vector<int> &inliers)
{
	for (int i = 0; i < cloud_->size(); i++)
	{
		pcl::PointXYZRGBA p1, p2;
		p1 = cloud_->points[i];
		p2 = pre_cloud_->points[i];

		p1 = pcl::transformPoint<pcl::PointXYZRGBA>(p1, Eigen::Affine3f(transform));
		if ((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z) < threshold*threshold)
		{
			inliers.push_back(i);
		}

	}
}
bool isCollinear(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_, std::vector<int> indices)
{
	Eigen::Vector3f mean;
	mean.setZero();
	for (int i = 0; i < indices.size(); i++)
	{
		mean[0] += cloud_->points[indices[i]].x;
		mean[1] += cloud_->points[indices[i]].y;
		mean[2] += cloud_->points[indices[i]].z;
	}
	mean /= indices.size();


	Eigen::Matrix3f D;
	D.setZero();
	Eigen::Vector3f u;
	for (int i = 0; i < indices.size(); i++)
	{
		u[0] = cloud_->points[indices[i]].x - mean[0];
		u[1] = cloud_->points[indices[i]].y - mean[1];
		u[2] = cloud_->points[indices[i]].z - mean[2];

		D += u*u.transpose();
	}

	Eigen::Vector3cf eigens = D.eigenvalues();

	float e0 = eigens[0].real(), e1 = eigens[1].real(), e2 = eigens[2].real();
	float max1 = std::max(std::max(e0, e1), e2);
	std::cout << "linearity: " << max1 / (e0 + e1 + e2 - max1) << std::endl;
	if (max1 > (e0 + e1 + e2 - max1) * 30)
	{
		return true;
	}
	else
		return false;
}


bool PointCloudStreamer::ransacFusion(Eigen::Matrix4f &transform_refined)
{
	img_mat_rgb=cv::Mat(height, width, CV_8UC3, image_);
	cv::gpu::GpuMat img_gpu_mat;

	boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());

	//cv::cvtColor(pre_img_mat, pre_img_mat, CV_RGB2GRAY);
	cv::cvtColor(img_mat_rgb, img_mat, CV_RGB2GRAY);
	img_gpu_mat.upload(img_mat);
	//img_gpu_mat_pre.upload(pre_img_mat);

	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	
	cv::gpu::GpuMat mask_gpu;
	cv::Mat mask;

	//pre_mask = cv::Mat::zeros(img_mat.size(), CV_8U);
	mask = cv::Mat::zeros(img_mat.size(), CV_8U);
	//cv::DenseFeatureDetector dense;

	for (int i = 0; i < height; i++)
	{

		for (int j = 0; j < width; j++)
		{
			int ind = i*width + j;
			if (pcl_isfinite(cloud_.points[ind].x))
			{
				for (int ci = std::max(i - 6, 0); ci < std::min(height, i + 6); ci++)
				for (int cj = std::max(j - 6, 0); cj < std::min(width, j + 6); cj++)
					mask.at<uchar>(ci, cj) = 255;
			}
			/*if (pcl_isfinite(pre_cloud_.points[ind].x))
			{
			for (int ci = std::max(i - 6, 0); ci < std::min(height , i+6); ci++)
			for (int cj = std::max(j - 6, 0); cj < std::min(width , j+6); cj++)
			pre_mask.at<uchar>(ci, cj) = 255;
			}*/
		}
	}


	mask_gpu.upload(mask);

	cv::gpu::SURF_GPU surf;
	surf.upright = 1;

	
	surf(img_gpu_mat, mask_gpu, keypoints_gpu, img_desc_gpu);
	surf.downloadKeypoints(keypoints_gpu, keypoints);

	if (pre_cloud_.empty())
		return true;

	
	
	cv::gpu::BFMatcher_GPU matcher_gpu(cv::NORM_L2);
	cv::gpu::GpuMat trainIdx, distance;
	std::vector<cv::DMatch> matches;

	matcher_gpu.matchSingle(img_desc_gpu_pre, img_desc_gpu, trainIdx, distance);
	

	

	
	


	cv::gpu::BFMatcher_GPU::matchDownload(trainIdx, distance, matches);

	

	std::sort(matches.begin(), matches.end());
	pcl::PointCloud<pcl::PointXYZRGBA> pcl_keypoints, pcl_keypoints_pre;

	float mindist = (matches.begin())->distance;
	
	for (std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end() && pcl_keypoints.size()<50;)
	{
		int sid = it->queryIdx;
		int tid = it->trainIdx;

		int pre_px = keypoints_pre[sid].pt.x, pre_py = keypoints_pre[sid].pt.y;
		int px = keypoints[tid].pt.x, py = keypoints[tid].pt.y;
		int sind = pre_py*width + pre_px;
		int tind = py*width + px;

		bool valid_match = false;
		if (!pcl_isfinite(cloud_.points[tind].x))
		{

			for (int tx = std::max(px - 1, 0); tx < std::min(px + 1, width); tx++)
			for (int ty = std::max(py - 1, 0); ty < std::min(py + 1, height); ty++)
			{
				if (pcl_isfinite(cloud_.points[ty*width + tx].x))
				{
					keypoints[tid].pt.x = tx;
					keypoints[tid].pt.y = ty;
					valid_match = true;
					break;
				}
			}
		}
		else
			valid_match = true;
		if (!valid_match)
		{
			it = matches.erase(it); continue;
		}

		valid_match = false;
		if (!pcl_isfinite(pre_cloud_.points[sind].x))
		{

			for (int tx = std::max(pre_px - 1, 0); tx < std::min(pre_px + 1, width); tx++)
			for (int ty = std::max(pre_py - 1, 0); ty < std::min(pre_py + 1, height); ty++)
			{
				if (pcl_isfinite(pre_cloud_.points[ty*width + tx].x))
				{
					keypoints_pre[sid].pt.x = tx;
					keypoints_pre[sid].pt.y = ty;
					valid_match = true;
					break;
				}
			}
		}
		else
			valid_match = true;
		if (!valid_match)
			it = matches.erase(it);
		else
		{
			pcl_keypoints.push_back(cloud_[int(keypoints[tid].pt.y)*width + int(keypoints[tid].pt.x)]);
			pcl_keypoints_pre.push_back(pre_cloud_[int(keypoints_pre[sid].pt.y)*width+int(keypoints_pre[sid].pt.x)]);
			it++;
		}

	}

	float k = 30.0;
	int iter = 0;
	std::vector<int>best_inliers;
	int N = pcl_keypoints.size();
	int best_inliers_n = 0;
	while (iter<k&&iter<5000)
	{

		std::vector<int> samples;
		Eigen::Matrix4f transform;

		if (getSamples(pcl_keypoints.makeShared(), pcl_keypoints_pre.makeShared(), samples,500,0.01))
		{
			pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGBA, pcl::PointXYZRGBA> est;
			std::vector<int> now_inliers;

			est.estimateRigidTransformation(pcl_keypoints, samples, pcl_keypoints_pre, samples, transform);
			countInliers(pcl_keypoints.makeShared(), pcl_keypoints_pre.makeShared(),transform, 0.05,now_inliers);
			int inlier_n = now_inliers.size();
			

			if (inlier_n > best_inliers_n)
			{
				best_inliers = now_inliers;
				best_inliers_n = inlier_n;
				double w = static_cast<double> (best_inliers_n)*1.0 / N;
				double p_no_outliers = 1.0 - pow(w, static_cast<double> (3));
				p_no_outliers = (std::max) (std::numeric_limits<double>::epsilon(), p_no_outliers);       // Avoid division by -Inf
				p_no_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers);   // Avoid division by 0.
				k = log(1 - 0.9f) / log(p_no_outliers);
			}

		}
		else
		{
			std::cout << "sample error\n";
			return false;
		}
		iter++;
		
	}

	
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < best_inliers.size(); i++)
		good_matches.push_back(matches[best_inliers[i]]);
		
	drawMatches(pre_img_mat, keypoints_pre, img_mat, keypoints, good_matches, img_matches);
	imshow("matches", img_matches);
	cvWaitKey(2);
	
	std::stringstream ss;
	ss << frameid;
	cv::imwrite("data/_img/" + ss.str() + ".jpg", img_matches);

	if (/*best_inlier_n < 0.2*N||*/best_inliers.size()<5)
	{
		std::cout << "!!![1]failed: iter: " << iter << " inlier:" << best_inliers.size() << std::endl;
		return false;
	}
	
	/*if (isCollinear(pcl_keypoints.makeShared(), best_inliers))
	{
		std::cout << "!!![1]failed: collinear!!" << std::endl;
		return false;
	}*/
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGBA, pcl::PointXYZRGBA> est;
	est.estimateRigidTransformation(pcl_keypoints, best_inliers, pcl_keypoints_pre, best_inliers, transform_refined);
	
	//getTranslationFromWahaba(pcl_keypoints_pre.makeShared(), pcl_keypoints.makeShared(), frameid, 10, filenames, sensorMatrixHost_, transform_refined);
	//draw keypoints and correspondence for visualization purpose
	
	std::cout << "[1] ransac finished iter: " << iter << "with inlier#: " << best_inliers.size() << " of feature# " << N << " finish!!" << std::endl;


	boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
	boost::posix_time::time_duration dt = t2 - t1;
	std::cout << "[2]match a frame in: " << dt.total_milliseconds() / 1000.0 << std::endl;
	//estimate transform with refined inliers		
	
	




	return true;

}
