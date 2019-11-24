#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <bitset>

static int maxHistVal = 0;

void histogram(cv::Mat& img,cv::Mat& hist){
	static constexpr int N = 256;
	int data[3*N]{0};
	int* h_data = data;
	int* l_data = data + N;
	int* s_data = data + 2*N;


	for (int r = 0; r < img.rows; ++r) {
		auto begin = img.ptr<uchar>(r);
		for (int c = 0; c < img.cols; ++c) {
			auto& h = *begin++;
			auto& l = *begin++;
			auto& s = *begin++;
			
			++h_data[h];
			++l_data[l];
			++s_data[s];
		}
	}

	maxHistVal = std::max(maxHistVal,*std::max_element(data,data+3*N-1));
	double factor = static_cast<double>(N)/static_cast<double>(maxHistVal);	

	hist = cv::Mat::zeros(3*N,2*N,CV_8UC3);

	cv::Point2f prev_points[] = {{0,3*N-1},{0,2*N-1},{0,N-1}};

	for (int c = 0; c < N; ++c) {
		cv::Point2f points[] = {
			cv::Point2f(2*c, (3*N-factor*(h_data[c]))-1 ),
			cv::Point2f(2*c, (2*N-factor*(l_data[c]))-1 ),
			cv::Point2f(2*c, (1*N-factor*(s_data[c]))-1 )
		};
		cv::line(hist,prev_points[0],points[0],cv::viz::Color::cyan(),1,cv::LINE_AA);
		cv::line(hist,prev_points[1],points[1],cv::viz::Color::yellow(),1,cv::LINE_AA);
		cv::line(hist,prev_points[2],points[2],cv::viz::Color::magenta(),1,cv::LINE_AA);

		prev_points[0] = points[0];
		prev_points[1] = points[1];
		prev_points[2] = points[2];
	}
}




int main(){
	cv::VideoCapture cap("ObjectAbandonmentAndRemoval/ObjectAbandonmentAndRemoval1.avi");
	cv::Mat frame,cmp_frame,cmp_output, hls, histImg, frame_gray, prev_frame_gray, running_avg, disp;
	
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float lranges[] = { 0, 255 };
	float sranges[] = { 0, 255 };
	const float* ranges[] = {hranges,lranges,sranges};
	int histSize[] = {30,32};
	int channels[] = {0,1,2};

	int scale = 10;
	double maxVal;

	//Feature based optical flow setup;
	std::vector<cv::Point2f> previous_features, current_features;
	const int MAX_CORNERS = 500;
	int win_size = 10;
	std::vector<uchar> features_found;

	cv::TermCriteria criteria(cv::TermCriteria::Type::MAX_ITER | cv::TermCriteria::Type::EPS, 20,0.03); 

	cap >> frame;
	cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);
	cv::goodFeaturesToTrack(frame_gray, current_features, MAX_CORNERS, 0.05, 5, cv::noArray(),3,false,0.04);
	previous_features = current_features;
	prev_frame_gray = frame_gray.clone();

	cmp_frame = cv::Mat::zeros(frame.rows,frame.cols,CV_8UC1);

	unsigned int rolling_avg_sum=0;
	const int N_FRAMES = 16;

	unsigned short int shift_reg = 0;

	bool isMoving = false; bool prevIsMoving = false;

	
	while(true) {

		cap >> frame;
		if (frame.empty()) {
			cap.set(cv::CAP_PROP_POS_MSEC,0); // Repeat until quit
			cap >> frame;
		}

		//Grayscale
		cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);


		//Histogram on HLS
		cv::cvtColor(frame,hls,cv::COLOR_BGR2HLS);
		histogram(hls,histImg);



		/*
		//Feature based optical flow
		disp = frame.clone();
		cv::goodFeaturesToTrack(prev_frame_gray, previous_features, MAX_CORNERS, 0.05, 5, cv::noArray(),3,false,0.04);
		cv::cornerSubPix(prev_frame_gray, previous_features, cv::Size(win_size,win_size),cv::Size(-1,-1), criteria);
		cv::calcOpticalFlowPyrLK ( prev_frame_gray, frame_gray, previous_features, current_features, features_found, cv::noArray(), cv::Size(win_size*4+1,win_size*4+1),5, criteria);
		for (int i = 0; i < (int)previous_features.size(); ++i) {
			if (features_found[i]) {
				cv::line(disp,previous_features[i],current_features[i],cv::viz::Color::red(),1,cv::LINE_AA);
			}
		}
		*/

		// How you look with beer
		cv::GaussianBlur(frame_gray,frame_gray,cv::Size(17,17),32);

		//absdiff
		cv::absdiff(frame_gray,cmp_frame,cmp_output);
		auto sum = cv::sum(cmp_output)[0];
		rolling_avg_sum = sum/N_FRAMES + (N_FRAMES-1)*rolling_avg_sum/N_FRAMES;

		// Do the bitwise dance
		shift_reg <<= 1;
		cmp_frame = frame_gray.clone();
		if (80000 < sum) {
			shift_reg |= 0x01;
		} else {
			shift_reg &= ~0x01;
		}
		if (shift_reg == 0xffff) {
			isMoving = true;
		} else if (shift_reg == 0x0000) {
			isMoving = false;
		}
		if ((isMoving) && (!prevIsMoving)) {
			std::cout << "Movement detected" << std::endl;
		} else if ((!isMoving) && (prevIsMoving)) {
			std::cout << "Movement stopped" << std::endl;
		}
		prevIsMoving = isMoving;

		//cv::imshow("float_me Optical feature flow", disp);
		cv::imshow("float_me ABSDIFF",cmp_output);
		cv::imshow("float_me3",frame_gray);
		cv::imshow("float_me1",hls);
		cv::imshow("float_me",histImg);

		auto key = cv::waitKey(30);
		if (key == 'q' || key == 27) {
			break;
		}

		prev_frame_gray = frame_gray.clone();
	}
	return 0;
}
