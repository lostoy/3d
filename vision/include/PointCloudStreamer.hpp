//
//#ifndef POINTCLOUDSTREAMER_H
//#define POINTCLOUDSTREAMER_H
//
//
//#include <boost/timer.hpp>
//#include <boost/chrono.hpp>
//#include <boost/filesystem.hpp>
//
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//
//#include <pcl/io/openni_grabber.h>
//#include <pcl/io/pcd_grabber.h>
//
//#include <pcl/io/io.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/io/ply_io.h>
//
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/image_viewer.h>
//
//#include <pcl/registration/icp.h>
//#include <pcl/registration/gicp.h>
//#include <pcl/filters/approximate_voxel_grid.h>
//#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
//#include <pcl/features/normal_3d.h>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/nonfree/gpu.hpp>
//
//
//#include <iostream>
//#include <string>
//#include <conio.h>  
//
//
//
//class PointCloudStreamer
//{
//public:
//	//constructor, several switches for the streamer is self-explaning.
//	PointCloudStreamer(pcl::Grabber &capture, bool enableVis = false) :capture_(capture),  enableVis_(enableVis), frameid(0), stepFrame_(false), saveFrame_(true)
//	{
//		exit_ = false;
//		worldTransform.setIdentity();
//		if (enableVis_)
//			initVis();
//		
//		boost::filesystem::create_directory("data");
//		boost::filesystem::create_directory("data/_img");
//		boost::filesystem::create_directory("data/_txt");
//		boost::filesystem::create_directory("data/_mat");
//	}
//
//	PointCloudStreamer(pcl::Grabber &capture, std::string dirname, bool enableVis = false) :capture_(capture), enableVis_(enableVis), frameid(0), stepFrame_(false), saveFrame_(true)
//	{
//		exit_ = false;
//		worldTransform.setIdentity();
//		if (enableVis_)
//			initVis();
//		
//		boost::filesystem::create_directory("data");
//		boost::filesystem::create_directory("data/_img");
//		boost::filesystem::create_directory("data/_txt");
//		boost::filesystem::create_directory("data/_mat");
//		boost::filesystem::create_directory("data/_cloud");
//		boost::filesystem::create_directory("data/_ori");
//		boost::filesystem::create_directory("data/_beforeicp");
//		filenames=setFilenames(dirname);
//	}
//
//	~PointCloudStreamer()
//	{
//
//	
//	}
//
//	void initVis();
//
//	
//	// kinect callback, copy the pointcloud and image generated by kinect to streamer.
//	void grabRGBAframe(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud);
//
//	std::vector<std::string> setFilenames(std::string dirname)
//	{
//
//		boost::filesystem::path dir(dirname);
//		std::vector<std::string> pcdFiles;
//		boost::filesystem::directory_iterator pos(dir);
//		boost::filesystem::directory_iterator end;
//
//		for (; pos != end; pos++)
//		if (boost::filesystem::is_regular_file(pos->status()))
//		if (boost::filesystem::extension(*pos) == ".pcd")
//			pcdFiles.push_back(pos->path().string());
//		std::sort(pcdFiles.begin(), pcdFiles.end());
//		return pcdFiles;
//	}
//	void mainLoopFile();
//	// the loop: feature extracted and transformation estimated
//	void mainLoop();
//
//	
//private:
//
//	
//	// experimental function, ignore this. trying to refine registration with icp.
//	void gicp(Eigen::Matrix4f &init_, Eigen::Matrix4f &transform)
//	{
//
//		pcl::IterativeClosestPoint< pcl::PointXYZRGBA, pcl::PointXYZRGBA > gicp_;
//		pcl::PointCloud<pcl::PointXYZRGBA> cloud_plus1, cloud_plus2;
//		//pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> gicp_;
//		std::vector<int> ind;
//		pcl::removeNaNFromPointCloud(pre_cloud_, cloud_plus1, ind);
//		pcl::removeNaNFromPointCloud(cloud_, cloud_plus2, ind);
//		//gridFilter(cloud_plus1, 0.05f);
//		//gridFilter(cloud_plus2, 0.05f);
//		gicp_.setInputSource(cloud_plus2.makeShared());
//		gicp_.setInputTarget(cloud_plus1.makeShared());
//		
//		//Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
//		gicp_.align(cloud_plus1,init_);
//		transform = gicp_.getFinalTransformation();
//		
//		//pcl::transformPointCloud(cloud_, cloud_, transform);
//
//	}
//
//	// the ransacFusion which is used by the streamer currently.
//	bool ransacFusion(Eigen::Matrix4f &transform_refined);
//	
//	bool checkValidTransform(Eigen::Matrix4f &transform);
//	// previous fusion method based on feature matching, without ransac
//	
//
//	// onlyicp calls gicp, ignore this.
//	void onlyicp(Eigen::Matrix4f &init_,Eigen::Matrix4f &transform)
//	{
//		//transform.setIdentity();
//		if (pre_cloud_.empty())
//			return;
//		boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
//		gicp(init_,transform);
//		boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
//		boost::posix_time::time_duration dt = t2 - t1;
//		
//
//		std::cout << "[2.5]fusion finished in " << dt.total_milliseconds() / 1000.0<<std::endl;
//	}
//
//	// voxel grid filter, helper function, ignore this.
//	void gridFilter(pcl::PointCloud<pcl::PointXYZRGBA> &cloud, float scale)
//	{
//		pcl::ApproximateVoxelGrid<pcl::PointXYZRGBA> sog;
//		sog.setInputCloud(cloud.makeShared());
//		sog.setLeafSize(scale, scale, scale);
//		sog.filter(cloud);
//
//	}
//
//	// saveCloud, used by streamer to save every REGISTRATED frame.
//	void saveCloud(std::string loc, std::string suffix)
//	{
//		if (cloud_.empty())
//			return;
//		std::stringstream ss;
//		ss << frameid;
//		pcl::PointCloud<pcl::PointXYZRGBA> cloud_f;
//		std::vector<int> ind;
//		pcl::removeNaNFromPointCloud(cloud_, cloud_f, ind);
//		pcl::io::savePLYFileBinary("data/"+loc + ss.str() + suffix, cloud_f);
//
//	}
//	
//	
//	// keyboard callback funtion, capture exit and save event
//	void keyboard_callback(const pcl::visualization::KeyboardEvent &e)
//	{
//		if (e.keyUp())
//		{
//			int key = e.getKeyCode();
//			if (key == (int)'q')
//				exit_ = true;
//			if (key == (int)'s')
//			{
//
//				saveCloud("_cloud/","_cloud.ply");
//			}
//
//		}
//	}
//
//	//careful!!!!!! for kinect1 640*480, for kinect2 512*424
//	static const int width = 1920, height = 1080;
//	
//	pcl::Grabber& capture_;
//
//	cv::Mat img_matches;
//
//	pcl::visualization::PCLVisualizer::Ptr cloud_viewer_;
//	
//	pcl::PointCloud<pcl::PointXYZRGBA> cloud_;
//	pcl::PointCloud<pcl::PointXYZRGBA> pre_cloud_;
//	pcl::PointCloud<pcl::PointXYZRGBA> world_;
//	
//	Eigen::Matrix4f worldTransform;
//
//	unsigned char image_[width*height * 3];
//	
//
//	
//	cv::gpu::GpuMat keypoints_gpu, keypoints_gpu_pre;
//	cv::gpu::GpuMat img_desc_gpu, img_desc_gpu_pre;
//	
//	cv::Mat img_mat, pre_img_mat,img_mat_rgb;
//	std::vector<cv::KeyPoint> keypoints, keypoints_pre;
//	
//
//
//
//	bool enableVis_;
//	bool exit_;
//	bool stepFrame_;
//	bool saveFrame_;
//	
//	boost::mutex data_ready;
//	boost::condition_variable data_ready_cond_;
//
//	std::vector<std::string> filenames;
//	size_t frameid;
//};
//
//#endif

//
#ifndef POINTCLOUDSTREAMER_H
#define POINTCLOUDSTREAMER_H


#include <boost/timer.hpp>
#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


//#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_grabber.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/image_viewer.h>

//#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/filters/filter.h>

#include <iostream>
#include <string>
#include <conio.h>  
#include <vector>
#include <algorithm>

#include <SuperFrame.h>
#include <MatchResult.h>

class PointCloudStreamer
{
public:
	//constructor, several switches for the streamer is self-explaning.
	PointCloudStreamer(std::string pcd_dir) :cur_frameid(0), stepFrame_(false), saveFrame_(true), enable_vis_(false)
	{
		exit_ = false;
		
		boost::filesystem::create_directory("data");
		boost::filesystem::create_directory("data/_img");
		boost::filesystem::create_directory("data/_txt");
		boost::filesystem::create_directory("data/_cloud");
		boost::filesystem::create_directory("data/_wcloud");
		boost::filesystem::create_directory("data/_mat");

		//list pcd files
		filenames = setFilenames(pcd_dir);

		//initialize world transform
		worldTransform_.setIdentity();

		//intialize superframes
		superFrames[0] = new SuperFrame;
		superFrames[1] = new SuperFrame;

		//intialize visualization
		if (enable_vis_)
		{
			cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer);
			cloud_viewer_->registerKeyboardCallback(boost::bind(&PointCloudStreamer::keyboard_callback, this, _1));

		}

	}

	void keyboard_callback(const pcl::visualization::KeyboardEvent &e)
	{
		if (e.keyUp())
			{
				
				int key = e.getKeyCode();
				if (key == (int)'q')
					exit_ = true;
				
				
			}
	}

	~PointCloudStreamer()
	{

	
	}

	static int getFrameIDFromPath(std::string path)
	{
		boost::filesystem::path s(path);
		std::string filename = s.filename().string();


		int se = filename.find_last_of(".ply");
		return (atoi(filename.substr(0, se).c_str()));
	}

	static bool isFilenameSmaller(std::string s1, std::string s2)
	{
		int i1 = getFrameIDFromPath(s1);
		int i2 = getFrameIDFromPath(s2);
		return i1<i2;
	}


	//get all pcd fiels in $dirname, and sort them by frameid
	std::vector<std::string> setFilenames(std::string dirname)
	{

		boost::filesystem::path dir(dirname);
		std::vector<std::string> pcdFiles;
		boost::filesystem::directory_iterator pos(dir);
		boost::filesystem::directory_iterator end;

		for (; pos != end; pos++)
		if (boost::filesystem::is_regular_file(pos->status()))
		if (boost::filesystem::extension(*pos) == ".pcd")
			pcdFiles.push_back(pos->path().string());
		std::sort(pcdFiles.begin(), pcdFiles.end(), isFilenameSmaller);
		return pcdFiles;
	}
	void mainLoopFile();
	void gicp(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr query_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr train_cloud, Eigen::Matrix4f &transform)
	{
		pcl::IterativeClosestPoint< pcl::PointXYZRGBA, pcl::PointXYZRGBA > gicp_;
		pcl::PointCloud<pcl::PointXYZRGBA> cloud_plus1, cloud_plus2;
		
		std::vector<int> ind;
		gridFilter(query_cloud,cloud_plus1, 0.001f);
		gridFilter(train_cloud,cloud_plus2, 0.001f);
		gicp_.setInputSource(cloud_plus1.makeShared());
		gicp_.setInputTarget(cloud_plus2.makeShared());
				
		
		gicp_.align(cloud_plus1, transform);
		transform = gicp_.getFinalTransformation();
		
				
		
	}
	
private:

	//to check if the new frame is within a small step of the previous one
	bool isSmallStep(MatchResult &res)
	{
		
		Eigen::Quaternion <float> q(res.transform.block<3, 3>(0, 0));
		
		//if almost lose track
		if (res.inliers.size() <= 20)
			return false;

		//if the new frame is within 0.4m of the previous one
		if ((res.transform.col(3).squaredNorm() - 1) > 0.4f*0.4f)
			return false;
		//if the new frame is within 10 degrees of the previous one
		if (abs(acos(q.w())) > M_PI / 36)
			return false;

		return true;
	}

	//filter invalid point
	void gridFilter(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA> &out_cloud, float leaf_size)
	{
		pcl::ApproximateVoxelGrid<pcl::PointXYZRGBA> sog;
		sog.setInputCloud(cloud);
		sog.setLeafSize(leaf_size, leaf_size, leaf_size);
		sog.filter(out_cloud);
	}


	// saveCloud. if world is true, it will save a cloud in the worl coordinate
	void saveCloud(pcl::PointCloud<pcl::PointXYZRGBA> &cloud_, std::string loc, size_t frameid,std::string suffix,bool world)
	{
		if (cloud_.empty())
			return;
		std::stringstream ss;
		ss << frameid;
		pcl::PointCloud<pcl::PointXYZRGBA> cloud_f;
		std::vector<int> ind;
		pcl::removeNaNFromPointCloud(cloud_, cloud_f, ind);
		if (world)
			pcl::transformPointCloud(cloud_f, cloud_f, worldTransform_);
		pcl::io::savePLYFileBinary("data/"+loc + ss.str() + suffix, cloud_f);

	}

	//save the pose matrix for global optimization
	void saveMatrix(MatchResult res, std::string loc, size_t frameid, std::string suffix)
	{
		std::stringstream ss;
		ss << frameid;
		std::fstream file("data/" + loc + ss.str() + suffix, std::ios::out);
		file << res.transform << std::endl;
		file << res.inliers.size() << std:: endl;
		file.close();
	}
	
	SuperFrame* superFrames[2];
	Eigen::Matrix4f worldTransform_;

	pcl::visualization::PCLVisualizer::Ptr cloud_viewer_;

	bool exit_;
	bool stepFrame_;
	bool saveFrame_;
	bool enable_vis_;
	
	std::vector<std::string> filenames;
	size_t cur_frameid;
};

#endif