//#include "PointCloudStreamer.hpp"
//
//std::vector<std::string> getPCDFiles(std::string dirname)
//{
//
//	boost::filesystem::path dir(dirname);
//	std::vector<std::string> pcdFiles;
//	boost::filesystem::directory_iterator pos(dir);
//	boost::filesystem::directory_iterator end;
//
//	for (; pos != end; pos++)
//	if (boost::filesystem::is_regular_file(pos->status()))
//	if (boost::filesystem::extension(*pos) == ".pcd")
//		pcdFiles.push_back(pos->path().string());
//	std::sort(pcdFiles.begin(), pcdFiles.end());
//	return pcdFiles;
//}
//
//int main(int argc, char* argv[])
//{
//	boost::shared_ptr<pcl::Grabber> capture;
//	float fps = 0.5;
//	boost::shared_ptr<PointCloudStreamer> app;
//	try
//	{
//		if (argc < 2)
//		{
//			capture.reset(new pcl::OpenNIGrabber());
//			app= boost::shared_ptr<PointCloudStreamer>(new PointCloudStreamer(*capture));
//		}
//		else
//		{
//			
//			capture.reset(new pcl::PCDGrabber<pcl::PointXYZRGBA>(getPCDFiles(argv[1]), fps, false));
//			app = boost::shared_ptr<PointCloudStreamer>(new PointCloudStreamer(*capture, std::string(argv[1])));
//			
//		}
//		
//
//	}
//	catch (const pcl::PCLException&){ std::cout << "wrong open device!" << std::endl; exit(-1); }
//	
//	
//
//	
//	app->mainLoopFile();
//	return 0;
//}

#include <PointCloudStreamer.hpp>
#include <cstdio>;

int main(int argc,char *argv[])
{
	PointCloudStreamer app(argv[1]);
	//freopen("log.log", "w", stdout);
	app.mainLoopFile();
	std::cout << "ok" << std:: endl;
	getch();
	return 0;

}