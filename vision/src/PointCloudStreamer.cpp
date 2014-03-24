
#include "PointCloudStreamer.hpp"


void PointCloudStreamer::initVis()
{
	cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer);
	cloud_viewer_->setBackgroundColor(0, 0, 0);
	cloud_viewer_->registerKeyboardCallback(boost::bind(&PointCloudStreamer::keyboard_callback, this, _1));
}


void PointCloudStreamer::grabRGBAframe(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
{
	boost::mutex::scoped_try_lock lock(data_ready);
	if (!lock || exit_)
		return;
	boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());

	pcl::copyPointCloud(*cloud, cloud_);
	size_t j = 0;
	for (size_t i = 0; i < cloud->size(); i++)
	{

		image_[j++] = cloud->points[i].b;
		image_[j++] = cloud->points[i].g;
		image_[j++] = cloud->points[i].r;
	}
	boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
	boost::posix_time::time_duration dt = t2 - t1;
	std::cout << "grab a frame in: " << dt.total_milliseconds() / 1000.0 << std::endl;

	data_ready_cond_.notify_one();
}


bool PointCloudStreamer::checkValidTransform(Eigen::Matrix4f &transform)
{
	Eigen::Matrix3f transform_ = worldTransform.block(0, 0, 3, 3);
	transform_.transposeInPlace();
	transform_ = transform.block(0,0,3,3)*transform_;
	std::cout << (transform_.trace() - 1) / 2 << std::endl;
	return (((transform_.trace() - 1) / 2) > 0.20);
	
}
void filterCloud(pcl::PointCloud<pcl::PointXYZRGBA> &cloud, float size)
{
	for (int i = 0; i < cloud.size(); i++)
	{
		if (pcl_isfinite(cloud.points[i].x))
		{
			cloud.points[i].x = floor(cloud.points[i].x / size)*size;
			cloud.points[i].y = floor(cloud.points[i].y / size)*size;
			cloud.points[i].z = floor(cloud.points[i].z / size)*size;
		}
	}
}

void PointCloudStreamer::mainLoop()
{
	using namespace openni_wrapper;
	typedef boost::shared_ptr<Image> ImagePtr;
	boost::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f_cloud = boost::bind(&PointCloudStreamer::grabRGBAframe, this, _1);

	capture_.registerCallback(f_cloud);

	capture_.start();
	boost::unique_lock<boost::mutex> lock(data_ready);

	while (!exit_&&!cloud_viewer_->wasStopped())
	{
		if (stepFrame_)
		{
			char ch;
			ch = getch();
			if (ch == 'q')
				exit_ = true;
		}

		bool has_data = data_ready_cond_.timed_wait(lock, boost::posix_time::millisec(500));

		if (has_data)
		{
			std::cout << "\n\nframeid: " << frameid <<" "<<cloud_.header.frame_id<< std::endl;
			std::cout << "[0]START to fusion!!" << std::endl;

			

			Eigen::Matrix4f transform1;
			Eigen::Matrix4f transform2;

			bool succ1;
			transform1.setIdentity();
			transform2.setIdentity();
			
			succ1=ransacFusion(transform1);
							
			saveCloud("_beforeicp/","_beforeicp.ply");
			if (succ1/*&&checkValidTransform(transform1)*/)
				onlyicp(transform1,transform2);
			else
			{
				transform1 = worldTransform;
				pcl::transformPointCloud(cloud_, cloud_, worldTransform);
				onlyicp(transform1,transform2);
			}

			
			Eigen::Matrix4f transform=transform1*transform2;
			bool succ = pre_cloud_.empty() || (!cloud_.empty() && checkValidTransform(transform));
			if (succ )
			{
				
				pcl::transformPointCloud(cloud_, cloud_, transform);
				pcl::copyPointCloud(cloud_, pre_cloud_);
				//memcpy(pre_image_, image_, width*height * 3);
				
				worldTransform = transform;
				cv::gpu::GpuMat tmp;
				tmp = keypoints_gpu_pre;

				keypoints_gpu_pre = keypoints_gpu;
				keypoints_gpu = tmp;
				tmp = img_desc_gpu_pre;
				img_desc_gpu_pre = img_desc_gpu; 
				img_desc_gpu = tmp;
				
				std::vector<cv::KeyPoint> tmp2;
				tmp2 = keypoints_pre;
				keypoints_pre = keypoints;
				keypoints = tmp2;

				cv::Mat tmp3;
				tmp3 = pre_img_mat;
				pre_img_mat = img_mat;
				img_mat = tmp3;
				
				std::stringstream ss;
				ss << frameid;
				cv::imwrite("data/_txt/" + ss.str() + "_txt.bmp", img_mat_rgb);
				std::ofstream file(("data/_mat/" + ss.str() + "_mat.txt").c_str());
				file << worldTransform;
				file.close();

				//std::cout << worldTransform<<std::endl;
			}


			if (enableVis_)
			{


				
				cloud_viewer_->removeAllPointClouds();
				cloud_viewer_->addPointCloud(cloud_.makeShared(), "world_");
				cloud_viewer_->spinOnce(10);

			}

			if (succ&&saveFrame_)
			{

				saveCloud("_cloud/","_cloud.ply");
			}
			
			frameid++;
		}


	}
	capture_.stop();

}

void PointCloudStreamer::mainLoopFile()
{
	
	
	while (!exit_/*&&!image_viewer_.wasStopped()*/)
	{
		if (stepFrame_)
		{
			char ch;
			ch = getch();
			if (ch == 'q')
				exit_ = true;
		}

	
		if (frameid < filenames.size())
		{
			pcl::io::loadPCDFile(filenames[frameid],cloud_);
			size_t j = 0;
			for (size_t i = 0; i < cloud_.size(); i++)
			{

				image_[j++] = cloud_.points[i].b;
				image_[j++] = cloud_.points[i].g;
				image_[j++] = cloud_.points[i].r;
			}
			//saveCloud("_ori/","_ori.ply");
			std::cout << "\n\nframeid: " << frameid << " " << cloud_.header.frame_id << std::endl;
			std::cout << "[0]START to fusion!!" << std::endl;

			
			Eigen::Matrix4f transform1;
			Eigen::Matrix4f transform2;

			bool succ1;
			transform1.setIdentity();
			transform2.setIdentity();

			succ1 = ransacFusion(transform1);
			
			//saveCloud("_beforeicp/","_beforeicp.ply");
			if (succ1/*&&checkValidTransform(transform1)*/)
				onlyicp(transform1,transform2);
			else
			{
				
				transform1 = worldTransform;
				
				//pcl::transformPointCloud(cloud_, cloud_, worldTransform);
				onlyicp(transform1,transform2);
			}


			
			bool succ = pre_cloud_.empty() || (!cloud_.empty() && checkValidTransform(transform2));
			if (succ)
			{
				
				pcl::transformPointCloud(cloud_, cloud_, transform2);
				
				//projectcloud2plane(pre_cloud_,cloud_, 0.05f);
				pcl::copyPointCloud(cloud_, pre_cloud_);
				//memcpy(pre_image_, image_, width*height * 3);

				worldTransform = transform2;
				cv::gpu::GpuMat tmp;
				tmp = keypoints_gpu_pre;

				keypoints_gpu_pre = keypoints_gpu;
				keypoints_gpu = tmp;
				tmp = img_desc_gpu_pre;
				img_desc_gpu_pre = img_desc_gpu;
				img_desc_gpu = tmp;

				std::vector<cv::KeyPoint> tmp2;
				tmp2 = keypoints_pre;
				keypoints_pre = keypoints;
				keypoints = tmp2;

				cv::Mat tmp3;
				tmp3 = pre_img_mat;
				pre_img_mat = img_mat;
				img_mat = tmp3;

				std::stringstream ss;
				ss << frameid;
				cv::imwrite("data/_txt/" + ss.str() + "_txt.bmp", img_mat_rgb);
				std::ofstream file(("data/_mat/" + ss.str() + "_mat.txt").c_str());
				file << worldTransform;
				file.close();

				//std::cout << worldTransform<<std::endl;
			}


			if (enableVis_)
			{


				//image_viewer_.showRGBImage(image_, width, height);
				cloud_viewer_->removeAllPointClouds();
				cloud_viewer_->addPointCloud(cloud_.makeShared(), "world_");
				cloud_viewer_->spinOnce(10);

			}

			if (succ&&saveFrame_)
			{

				saveCloud("_cloud/","_cloud.ply");
			}
			
			frameid+=1;
		}


	}
	

}


