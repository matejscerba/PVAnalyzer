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

        body_detector detector(0, cv::Point(805, 385));

        // Try to open video.
        if (!video.open(filename)){
            cv::Mat im = cv::imread(filename);
            detector.detect(im);
            return;
        }

        cv::Mat frame;

        // Video is opened, processing begins.
        for (;;) {
            video >> frame;

            // Video ended.
            if (frame.empty())
                break;

            // Detect body.
            detector.detect(frame);
        }

        // detector.write("tracked.avi");

        // Free resources.
        video.release();
        cv::destroyAllWindows();
    }

};