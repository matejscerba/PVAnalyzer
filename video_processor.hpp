#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

#include "body_detector.hpp"


/**
 * @brief Wrapper of whole program's functionality.
 * 
 * This class is intented to be used for default functionality. It processes
 * the video and passes individual frames to other parts of the program.
 */
class video_processor {

    /// @brief Holds frames modified by processor.
    std::vector<cv::Mat> frames;

    /**
     * @brief Write modified frames as a video to given file.
     * 
     * @param filename Path to file, where modified video should be saved.
     */
    void write(const std::string &&filename) const {
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('D','I','V','X'), 30, cv::Size(frames.back().cols, frames.back().rows));
        for (const auto &f : frames)
            writer.write(f);
        writer.release();
    }

public:

    /**
     * @brief Process video at certain path frame by frame.
     * 
     * @param filename Path to video file to be processed.
     */
    void process(const std::string &filename) {
        // Try to open video.
        cv::VideoCapture video;
        if (!video.open(filename)) {
            std::cout << "Error opening video " << filename << std::endl;
            return;
        }

        // Set default parameters.
        std::size_t fps = 30;
        std::size_t frame_start = 0;
        cv::Point position;
        if (filename.find("kolin2.MOV") != std::string::npos) {
            fps = 30;
            frame_start = 0;
            position = cv::Point(805, 385);
        } else if (filename.find("kolin.mp4") != std::string::npos) {
            frame_start = 96;
            position = cv::Point(85, 275);
        }

        // Prepare body detector.
        body_detector detector(frame_start, position, fps);

        cv::Mat frame;
        // Video is opened, processing frame by frame begins.
        for (;;) {
            video >> frame;

            // Video ended.
            if (frame.empty())
                break;

            // Detect body.
            body_detector::result res = detector.detect(frame);
            if (res == body_detector::result::error)
                break; // Error has occured while detecting body.
            else if (res == body_detector::result::skip)
                continue; // This frame is supposed to be skipped.
            
            frames.push_back(frame.clone());

            // Display current frame.
            cv::imshow("frame", frame);
            cv::waitKey();

            if (filename.find("kolin2.MOV") != std::string::npos) {
                video >> frame; video >> frame; video >> frame;
            }
        }

        // write("scaling-2.avi");

        // Free resources.
        video.release();
        cv::destroyAllWindows();
    }

};