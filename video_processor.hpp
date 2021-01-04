#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

#include "body_detector.hpp"

class video_processor {

    std::vector<cv::Mat> frames;

    void write(const std::string &&filename) {
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('D','I','V','X'), 30, cv::Size(frames.back().cols, frames.back().rows));
        for (auto &f : frames)
            writer.write(f);
        writer.release();
    }

public:

    // Processes video from path `filename`.
    void process(const std::string &filename) {
        cv::VideoCapture video;

        std::size_t fps = 30;
        std::size_t frame_start = 0;
        cv::Point position;
        if (filename.find("kolin2.MOV") != std::string::npos) {
            fps = 120;
            frame_start = 0;
            position = cv::Point(805, 385);
        } else if (filename.find("kolin.mp4") != std::string::npos) {
            frame_start = 96;
            position = cv::Point(85, 275);
        }

        body_detector detector(frame_start, position, fps);

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
            body_detector::result res = detector.detect(frame);
            if (res == body_detector::result::error)
                break;
            else if (res == body_detector::result::skip)
                continue;
            

            frames.push_back(frame.clone());

            // Display current person in frame.
            cv::imshow("frame", frame);
            cv::waitKey();
        }

        // write("tracked_rotation.avi");

        // Free resources.
        video.release();
        cv::destroyAllWindows();
    }

};