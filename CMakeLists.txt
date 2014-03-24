cmake_minimum_required(VERSION 2.8)
project(PCL_Streamer)
set(BOOST_LIBRARYDIR C:/local/boost_1_55_0/lib64-msvc-11.0)
find_package(PCL 1.2 REQUIRED COMPONENTS common features filters io io_ply kdtree registration sample_consensus search visualization segmentation)
set(OpenCV_STATIC ON)
set(Qt_DIR c:/Qt/4.8.0)
find_package( OpenCV REQUIRED COMPONENTS core features2d highgui nonfree imgproc gpu)

set(VISION_INCLUDE_DIRS "vision/include")

include_directories(${PCL_INCLUDE_DIRS} ${VISION_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(
srcs vision/src/StreamerApp.cpp 
     vision/src/PointCloudStreamer.cpp 
	 vision/src/VisionFusion.cpp 
	 vision/src/SuperFrame.cpp 
	 
)

set(
incs vision/include/SuperFrame.h
	 vision/include/PointCloudStreamer.hpp)
add_executable(PointCloudStreamer ${srcs} ${incs})


target_link_libraries( PointCloudStreamer ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})


