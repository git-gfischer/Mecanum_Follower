/**
* This file is part of Mecanum-follower.
* Copyright (C) 2021 Gabriel Fischer Abati - Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: João Carlos Virgolino Soares, Marcelo Gattass and Marco Antonio Meggiolaro
* For more information see https://github.com/Master-Fischer/Mecanum_Follower.
* Please report suggestions and comments to fischerabati@gmail.com
**/
//Usage: ./yoloDepth 


#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>


#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

//tracking libraries SORT 
#include <opencv2/tracking.hpp>
#include "Tracking/Hungarian.h"
#include "Tracking/KalmanTracker.h"

//control library
#include "Mecanum_Controller/Mecanum_controller.h"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include<algorithm>
#include<chrono>

#define CONTROLLER 
#define RECORDER

using namespace std;
using namespace cv;
using namespace dnn;

//yolo variables and parameters
cv::dnn::Net net;

// Initialize the parameters
float confThreshold = 0.5; // 0.5 Confidence threshold
float nmsThreshold = 0.4;  // 0.4 Non-maximum suppression threshold

// yoloV3 416/416
//CyTi 160/160
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image

vector<string> classes;

bool protonect_shutdown = false; // Whether the running application should shut down.

//functions prototypes
// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(Mat& frame, const vector<Mat>& out, vector<int>& classIds);
void postprocess(Mat& frame, const vector<Mat>& outs, vector<int>& classIds, vector<int>& centersX, vector<int>& centersY, vector<Rect>& boxes);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);


enum
{
	Processor_cl,
	Processor_gl,
	Processor_cpu
};
//=================Serial Comm====================
void end_comm(int s)
{
	std::cout<<"turn off Serial Comm" <<std::endl;
	close(s);
}
//================================================
void send_cmd(int s, std::string inp)
{
	int n=write(s,inp.c_str(),inp.length());
	if (n < 0) {fputs("write()to microcontroller failed!\n", stderr);}
	//else {cout<<"msg sent"<<endl;}
}
//================================================
typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;
//================================================
// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
//================================================
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, vector<int>& classIds, vector<int> &centersX, vector<int> &centersY, vector<Rect>& boxes)
{
    //vector<int> classIds;
    vector<float> confidences;
    //vector<Rect> boxes;
	//vector<int> centersX;
	//vector<int> centersY;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
				if(classIdPoint.x==0){ // filter to show only people 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
				centersX.push_back(centerX);
				centersY.push_back(centerY);
				}
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];	
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, frame);
		
    }
}
//========================================================
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}
//========================================================
// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
//========================================================
void sigint_handler(int s){protonect_shutdown = true;}
//========================================================
//=========================MAIN===========================
//========================================================
int main(int argc, char** argv)
{
	std::cout << "Staring mecanum follower" << std::endl;

	// Load names of classes
	cout << "loading classes...";    	
	//string classesFile = "models/yoloV3/coco.names";
	string classesFile = "models/CYTi/CYTI.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
	cout << "done\n";
    
    // Give the configuration and weight files for the model
    cout << "loading weight files...";

	//yoloV3
    //string modelConfiguration = "models/yoloV3/yolov3.cfg";
    //string modelWeights = "models/yoloV3/yolov3.weights";
    
	//CYTi
	string modelConfiguration = "models/CYTi/CYTI.cfg";
    string modelWeights = "models/CYTi/CYTI.weights";
	
	cout << "done\n";


	cout<< "loading Tracker..." <<endl;
	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3; //0.3
	vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
	
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;


	cout<< "done" <<endl;


    // Load the network
	cout << "loading network...";
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

	//run in CPU
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 

	//run in GPU
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); 
	cout << "done\n";  

	cout << "--------------------------------------------"  << endl;

	cout << "strating Kinect Camera" << endl;

	libfreenect2::Freenect2 freenect2;
	libfreenect2::Freenect2Device *dev = NULL;
	libfreenect2::PacketPipeline  *pipeline = NULL;

	//check if the Kinect Microsoft is connected
	if(freenect2.enumerateDevices() == 0)
	{
		std::cout << "no device connected!" << std::endl;
		return -1;
	}

	string serial = freenect2.getDefaultDeviceSerialNumber(); // Serial communication between pc and Camera

	std::cout << "SERIAL: " << serial << std::endl;

#if 1 // sean
	int depthProcessor = Processor_cl;

	if(depthProcessor == Processor_cpu)
	{
		if(!pipeline)
			//! [pipeline]
			pipeline = new libfreenect2::CpuPacketPipeline();
		//! [pipeline]
	}
	else if (depthProcessor == Processor_gl) // if support gl
	{
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
		if(!pipeline)
		{
			pipeline = new libfreenect2::OpenGLPacketPipeline();
		}
#else
		std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
	}
	else if (depthProcessor == Processor_cl) // if support cl
	{
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
		if(!pipeline)
			pipeline = new libfreenect2::OpenCLPacketPipeline();
#else
		std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
	}

	if(pipeline) {dev = freenect2.openDevice(serial, pipeline);}
	else {dev = freenect2.openDevice(serial);}

	if(dev == 0) 
	{
		std::cout << "failure opening device!" << std::endl;
		return -1;
	}

	signal(SIGINT, sigint_handler);
	protonect_shutdown = false;

	libfreenect2::SyncMultiFrameListener listener(
			libfreenect2::Frame::Color |
			libfreenect2::Frame::Depth |
			libfreenect2::Frame::Ir);
	libfreenect2::FrameMap frames;

	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);

	dev->start();

	std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
	std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

	libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
	libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4);
	
	cv::Mat rgbmat, depthmat, depthmatUndistorted, irmat, rgbd, rgbd2; //opencv objects

	//Creating opencv display windows
	cv::namedWindow("rgb", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("ir", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("depth", WND_PROP_ASPECT_RATIO);
	//cv::namedWindow("undistorted", WND_PROP_ASPECT_RATIO);
	//cv::namedWindow("registered", WND_PROP_ASPECT_RATIO);
	//cv::namedWindow("depth2RGB", WND_PROP_ASPECT_RATIO);

	#ifdef RECORDER	 //record video
		std::cout<<"Starting recorder..."<<std::endl;
		int frame_width = 1920;
		int frame_height = 1080;
		VideoWriter video("../recordings/output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));
		VideoWriter video_raw("../recordings/out_raw.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));
		//VideoWriter video_depth("../recordings/out_depth.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));
		unsigned int depth_map_counter=0;
		std::cout<<"OK"<<std::endl;
	#endif

	#ifdef CONTROLLER
		//Setup Serial Com (Low level arduino motor control)
		cout<<"Starting Serial Communication..."<< endl;
		
		
		struct termios tio;	
		memset(&tio,0,sizeof(tio));
		tio.c_iflag=0;
		tio.c_oflag=0;
		tio.c_cflag=CS8|CREAD|CLOCAL; // 8n1, see termios.h for more information
		tio.c_lflag=0;
		tio.c_cc[VMIN]=1;
		tio.c_cc[VTIME]=5;
		

		//Might have to change depending on what port your microcontroller is plugged in
		int serial_port = open("/dev/ttyUSB1", O_RDWR); 

		cfsetospeed(&tio,B115200); // 115200 baud
		cfsetispeed(&tio,B115200); // 115200 baud
		tcsetattr(serial_port,TCSANOW,&tio);

		cout<<"Serial communication activated" <<endl;

		cout<< "Starting Controller"<<endl;
		Mecanum_controller controller;
		cout<< "Controller activated" <<endl;
	#endif


	while(!protonect_shutdown)
	{
		//get frames
		listener.waitForNewFrame(frames);
		libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
		libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
		libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
		
		//convert to opencv
		cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbmat);
		cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irmat);
		cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);

		//convert channel of rgbmat
		cv::cvtColor(rgbmat,rgbmat,CV_BGRA2BGR);
		cv::Mat raw_frame=rgbmat.clone();


		//draw a point in the middle of the frame
		int center_frame_X=rgbmat.size().width/2;
		int center_frame_Y=rgbmat.size().height/2;
		cv::Point center_point(center_frame_X,center_frame_Y);
		cv::circle(rgbmat,center_point,5,cv::Scalar(0,0,100),CV_FILLED);
	
		//show images
		//cv::imshow("rgb", rgbmat);
		//cv::imshow("ir", irmat / 4500.0f);
		//cv::imshow("depth", depthmat / 4500.0f);
		
		registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
		

		//Object detection
		Mat blob; 
		Mat frame = rgbmat; // object dection output

		// Create a 4D blob from a frame.
		blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
		
		//Sets the input to the network
		net.setInput(blob);
		
		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		vector<int> classId;
		vector<int> centersX;
		vector<int> centersY;
		vector<Rect> boxes; //[left, top, width, height]
		vector<Rect> people_boxes;

		
		// Remove the bounding boxes with low confidence
		postprocess(frame, outs, classId, centersX, centersY, boxes);
		
		//cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(depthmatUndistorted);
		//cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd);
		cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(rgbd2);

		//tracking----------------------------------------------
		if(trackers.size()==0) //the first frame met
		{
			for (int i=0; i<centersX.size();i++) // for every bbox in the frame
			{
				Rect box = boxes[i];
				KalmanTracker trk = KalmanTracker(box);
				trackers.push_back(trk);
			}
		}
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		trkNum = predictedBoxes.size();
		detNum = centersX.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], boxes[j]);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		// 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(boxes[detIdx]);
		}
		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(boxes[umd]);
			trackers.push_back(tracker);
		}

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}
		//display
		for (auto tb : frameTrackingResult)cv::rectangle(frame, tb.box, Scalar(0,0,255), 2, 8, 0);

		//getting depth from centroid of the bbox
		float* depth_val=(float*)rgbd2.data;
		for (int i=0; i<centersX.size();i++)
		{		
			cv::Rect box = boxes[i];
			float val=depth_val[centersX[i]+rgbd2.size().width*centersY[i]];
			std::string label=format("Depth: %.3f mm", val);
			int baseLine;
    		cv::Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			cv::rectangle(frame, cv::Point(box.x + 150, box.y - round(1.5*labelSize.height)), Point(box.x + round(2.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
			cv::putText(frame,label, cv::Point(box.x + 150, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),2);

			//Draw a point at the centroid
			cv::Point centroid(centersX[i],centersY[i]);
			cv::circle(frame,centroid,5,cv::Scalar(0,0,100),CV_FILLED);

			//get id
			TrackingBox res=frameTrackingResult[i];
			int id=res.id;
			std::string label_id=format("ID: %d", id);
			int baseLine_id;
    		labelSize = getTextSize(label_id, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine_id);
			//cv::rectangle(frame, cv::Point(box.x+50, box.y - round(1.5*labelSize.height)-10), Point(box.x + round(2.5*labelSize.width), box.y + baseLine_id), Scalar(255, 255, 255), FILLED);
			cv::putText(frame,label_id, cv::Point(box.x, box.y+40), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);

			//control
			#ifdef CONTROLLER
			if (id==1){ //follow person with id=1
				cv::Point setpoint(center_frame_X,center_frame_Y);
				std::string msg=controller.calculate(setpoint,centroid,val); //calculate control output
				send_cmd(serial_port,msg); //send cmd to the robot via USB
			}
			else {send_cmd(serial_port,"S 0");} // robot stops
			#endif
		}

		//show images
		cv::imshow("rgb", frame);
		cv::imshow("ir", irmat / 4500.0f);
		cv::imshow("depth", depthmat / 4500.0f);

		#ifdef RECORDER
			cv::Mat depthmap= depthmat / 4500.0f;
			depth_map_counter++;
			std::string depthpath= "../recordings/depths/depth" + std::to_string(depth_map_counter) + ".jpg";
			video.write(frame);
			video_raw.write(raw_frame);
			cv::imwrite(depthpath,depthmap);
			//video_depth.write(depthmap);
		#endif
		//cv::imshow("undistorted", depthmatUndistorted / 4500.0f);
		//cv::imshow("registered", rgbd);
		//cv::imshow("depth2RGB", rgbd2 / 4500.0f);

		int key = cv::waitKey(1);
		protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

		listener.release(frames);
		
	}

	#ifdef CONTROLLER
		end_comm(serial_port);
	#endif

	dev->stop();
	dev->close();
	
	#ifdef RECORDER
		video.release();
		video_raw.release();
		//video_depth.release();	
	#endif

	delete registration;

#endif

	std::cout << "Goodbye World!" << std::endl;
	return 0;
}


/* YOLO detection and processing functions are based on the OpenCV project. 
They are subject to the license terms at http://opencv.org/license.html */

