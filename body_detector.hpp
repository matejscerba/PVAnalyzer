#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class body_detector {

    cv::HOGDescriptor _hog;

public:

    body_detector() {
        _hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    }

    // Detects athlete in frame.
    void detect(cv::Mat &frame) {
        std::vector<cv::Rect> detections;
        _hog.detectMultiScale(frame, detections, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2);

        for (const auto &detection : detections) {
            cv::rectangle(frame, detection.tl(), detection.br(), cv::Scalar(0, 255, 0), 2);
        }

        // Show current frame.
        cv::imshow("frame", frame);
        cv::waitKey(0);
    }

};