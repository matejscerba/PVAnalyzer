#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#include "body_detector.hpp"

class video_processor {

public:

    // Processes video from path `filename`.
    void process(const std::string &filename) {
        cv::VideoCapture video;
        
        // Try to open video.
        if (!video.open(filename))
            return;

        cv::Mat frame;
        body_detector detector;

        // Video is opened, processing begins.
        for (;;) {
            video >> frame;

            // Video ended.
            if (frame.empty())
                break;

            // Detect body.
            detector.detect(frame);
        }

        // Free resources.
        video.release();
        cv::destroyAllWindows();
    }

};