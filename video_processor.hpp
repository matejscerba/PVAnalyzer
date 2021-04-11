#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <optional>

#include "forward.hpp"
#include "body_detector.hpp"
#include "person.hpp"
#include "vault_analyzer.hpp"
#include "viewer.hpp"


/**
 * @brief Wrapper of whole program's functionality.
 * 
 * This class is intented to be used for default functionality. It processes
 * the video and passes individual frames to other parts of the program.
 */
class video_processor {
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
        double fps = 30;
        std::size_t frame_start = 0;
        cv::Point position;
        if (filename.find("kolin2.MOV") != std::string::npos) {
            frame_start = 0;
            position = cv::Point(805, 385);
        } else if (filename.find("kolin.mp4") != std::string::npos) {
            frame_start = 96;
            position = cv::Point(85, 275);
        }

        // Prepare body detector.
        body_detector detector(frame_start, position, fps);

        cv::Mat frame, raw_frame;
        body_detector::result res = body_detector::result::unknown;
        // Video is opened, processing frame by frame begins.
        for (std::size_t frame_no = 0; ; frame_no++) {
            video >> frame;
            raw_frame = frame.clone();

            // Video ended.
            if (frame.empty())
                break;

            std::cout << "Processing frame " << frame_no << std::endl;

            if (res != body_detector::result::error) {
                // No error occured yet, process video further.

                // Detect body.
                res = detector.detect(frame, frame_no);
                if (res == body_detector::result::ok) {
                    // Detection on given frame was valid.

                    // Draw detections in frame.
                    detector.draw(frame, frame_no);
                }
            }

            raw_frames.push_back(raw_frame);
            frames.push_back(frame.clone());

            // To be removed.
            if (filename.find("kolin2.MOV") != std::string::npos) {
                video >> frame; video >> frame; video >> frame;
            }
        }

        // Free resources.
        video.release();
        cv::destroyAllWindows();

        vault_analyzer analyzer;
        std::optional<person> athlete = detector.get_athlete();
        if (athlete) {
            analyzer.analyze(*athlete, filename, frames.size(), fps);
        } else {
            std::cout << "Athlete could not be detected in given video" << std::endl;
        }

        viewer v(frames, raw_frames, analyzer);
        v.show();

        // write("no_last_box.avi");
    }

private:

    /// @brief Holds frames modified by processor.
    std::vector<cv::Mat> frames;

    /// @brief Holds frames unmodified by processor.
    std::vector<cv::Mat> raw_frames;

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

};