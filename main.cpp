#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>

int main(){
	cv::VideoCapture cap("ObjectAbandonmentAndRemoval/ObjectAbandonmentAndRemoval1.avi");
	cv::Mat frame, hls, hist;
	std::vector<cv::Mat> hls_planes;
	
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 180 };
	const float* ranges[] = {hranges,sranges};
	int histSize = 64;

	while(true) {
		cap >> frame;
		if (frame.empty()) {
			cap.set(cv::CAP_PROP_POS_MSEC,0); // Repeat until quit
			cap >> frame;
		}
		cv::cvtColor(frame,hls,cv::COLOR_BGR2HLS);
		cv::split(hls,hls_planes);

		cv::calcHist(&hls,3,0,cv::Mat(),hist,1,&histSize,0);
		
//		cv::imshow("float_me",hls);
		cv::imshow("float_me",hist);

		auto key = cv::waitKey(30);
		if (key == 'q' || key == 27) {
			break;
		}
	}
	return 0;
}
