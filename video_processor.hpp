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

        std::size_t frame_start = 0;
        cv::Point start;
        if (filename.find("kolin2.MOV") != std::string::npos) {
            frame_start = 0;
            start = cv::Point(805, 385);
        } else if (filename.find("kolin.mp4") != std::string::npos) {
            frame_start = 96;
            start = cv::Point(85, 275);
        }

        body_detector detector(frame_start, start);

        // Video could not be opened, try photo.
        if (!video.open(filename)){
            cv::Mat im = cv::imread(filename);
            detector.detect(im);
            return;
        }

        cv::Mat frame;
        std::size_t c = 0;

        // Video is opened, processing begins.
        for (;;) {
            video >> frame;

            // Video ended.
            if (frame.empty())
                break;

            std::cout << frame.cols << " " << frame.rows << " " << c << std::endl;
            c++;

            // Detect body.
            detector.detect(frame);
        }

        // detector.write("tracked.avi");

        // Free resources.
        video.release();
        cv::destroyAllWindows();
    }

};