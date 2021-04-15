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
        // Prepare body detector.
        double fps = 30;
        body_detector detector(fps);

        // Set default parameters.
        // std::size_t frame_start = 0;
        // cv::Point position;
        // if (filename.find("kolin2.MOV") != std::string::npos) {
        //     frame_start = 0;
        //     position = cv::Point(805, 385);
        // } else if (filename.find("kolin.mp4") != std::string::npos) {
        //     frame_start = 96;
        //     position = cv::Point(85, 275);
        // }

        if (!find_athlete(filename, detector)) {
            std::cout << "Athete could not be found in video " << filename << std::endl;
            return;
        }
        detect_athlete(filename, detector, fps);

        vault_analyzer analyzer;
        analyzer.analyze(detector.get_athlete(), filename, frames.size(), fps);

        viewer v(frames, raw_frames, analyzer);
        v.show();

        // write("no_last_box.avi");
    }

private:

    /// @brief Holds frames modified by processor.
    std::vector<cv::Mat> frames;

    /// @brief Holds frames unmodified by processor.
    std::vector<cv::Mat> raw_frames;

    bool find_athlete(const std::string &filename, body_detector &detector) noexcept {
        // Try to open video.
        cv::VideoCapture video;
        if (!video.open(filename)) {
            std::cout << "Error opening video " << filename << std::endl;
            return false;
        }

        cv::Mat frame;
        for (std::size_t frame_no = 0; ; frame_no++) {
            video >> frame;

            // Video ended.
            if (frame.empty())
                break;

            detector.find(frame, frame_no);

            raw_frames.push_back(frame.clone());

            // To be removed.
            if (filename.find("kolin2.MOV") != std::string::npos) {
                video >> frame; video >> frame; video >> frame;
            }
        }

        video.release();

        return detector.is_found();
    }

    void detect_athlete(const std::string &filename, body_detector &detector, double fps) noexcept {
        detector.setup();

        cv::Mat frame;
        body_detector::result res = body_detector::result::unknown;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Processing frame " << frame_no << std::endl;

            frame = raw_frames[frame_no].clone();

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

            frames.push_back(frame);
        }
    }

    /**
     * @brief Write modified frames as a video to given file.
     * 
     * @param filename Path to file, where modified video should be saved.
     */
    void write(const std::string &filename) const {
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('D','I','V','X'), 30, cv::Size(frames.back().cols, frames.back().rows));
        for (const auto &f : frames)
            writer.write(f);
        writer.release();
    }

};