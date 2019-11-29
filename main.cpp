#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <bitset>

int main(){
	cv::VideoCapture cap("ObjectAbandonmentAndRemoval/ObjectAbandonmentAndRemoval1.avi");
	
	int key = 0;
	while(!(key == 'q' || key == 27)) {
		
		// MOG2 setup
		auto pSub = cv::createBackgroundSubtractorMOG2();
		pSub->setVarThreshold(50);

		cap.set(cv::CAP_PROP_POS_MSEC,0); // Repeat until quit
		cv::Mat MaskMOG2,frame, frame_blur,frame_masked;
		cap >> frame;

		do {
			cv::GaussianBlur(frame,frame_blur,{5,5},20);

			//MOG2
			pSub->apply(frame,MaskMOG2,0.0000001);

			// Rmove fireflies with morph operations
			auto mask_morph = MaskMOG2.clone();
			cv::Mat morph (6,6,CV_8UC1,cv::Scalar(1));
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_ERODE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_ERODE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);
			cv::morphologyEx(mask_morph,mask_morph,cv::MORPH_DILATE,morph);


			//apply mask
			cv::Mat frame_channels[3];
			cv::Mat masked_frame_channels[3];
			cv::split(frame,frame_channels);
			for (int i = 0; i < 3; ++i) {
				masked_frame_channels[i] = frame_channels[i] & mask_morph;
			}
			cv::merge(masked_frame_channels,3,frame_masked);

			// Contours
			cv::Mat boundImg = frame.clone();
			cv::Mat contourImg = frame.clone();
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(mask_morph,contours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
			if (contours.size()) {
				int idx = 0;
				for( ; idx >= 0; idx = hierarchy[idx][0] )
				{
					cv::Scalar color( rand()&255, rand()&255, rand()&255 );
					cv::drawContours( contourImg, contours, idx, color, cv::FILLED, 8, hierarchy );
				}
				
				//Bounding box
				for (const auto& i : contours) {
					auto box = cv::boundingRect(i);
					cv::rectangle(boundImg,box,cv::viz::Color::red());
				}
			}

			

			cv::imshow("float_me Boundings", boundImg);
			cv::imshow("flaot_me Contours",contourImg);
			cv::imshow("float_me Morphed Mask",mask_morph);
			cv::imshow("float_me MOG mask",MaskMOG2);
			cv::imshow("float_me frame_masked",frame_masked);
			cv::imshow("float_me frame",frame);


			key = cv::waitKey(30);
			if (key == 'q' || key == 27) {
				break;
			}

			cap >> frame;
		} while(!frame.empty()) ;
	}
	return 0;
}
