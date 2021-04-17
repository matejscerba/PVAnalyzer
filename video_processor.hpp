#pragma once

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

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
    void process(const std::string &filename) const noexcept {
        std::vector<cv::Mat> raw_frames = extract_frames(filename);

        double fps = 30;
        body_detector detector(fps);

        std::vector<cv::Mat> found_frames;
        if (!find_athlete(raw_frames, detector, found_frames)) {
            std::cout << "Athete could not be found in video " << filename << std::endl;
            return;
        }
        std::vector<cv::Mat> frames = detect_athlete(raw_frames, detector, fps);

        std::string output_filename = "outputs/videos/" + create_output_filename();
        std::string ext = ".avi";
        write(output_filename + "_raw_frames" + ext, raw_frames);
        write(output_filename + "_found_frames" + ext, found_frames);
        write(output_filename + "_frames" + ext, frames);

        // Analyze detected athlete.
        vault_analyzer analyzer;
        analyzer.analyze(detector.get_athlete(), filename, frames.size(), fps);

        // Show result.
        viewer v;
        v.show(frames, raw_frames, analyzer);
        cv::destroyAllWindows();
    }

private:

    /**
     * @brief Extract frames from video given by its path.
     * 
     * @param filename Path to video.
     * 
     * @returns frames of video.
     */
    std::vector<cv::Mat> extract_frames(const std::string &filename) const noexcept {
        std::vector<cv::Mat> frames;

        // Open video.
        cv::VideoCapture video;
        if (!video.open(filename)) {
            std::cout << "Error opening video " << filename << std::endl;
            return frames;
        }

        // Loop through video and save frames.
        cv::Mat frame;
        for (;;) {
            video >> frame;

            // Video ended.
            if (frame.empty()) break;

            // Save current frame.
            frames.push_back(frame.clone());

            // Skip frames of tested 120-fps video for efficiency reasons.
            if (filename.find("kolin2.MOV") != std::string::npos) {
                video >> frame; video >> frame; video >> frame;
            }
        }
        video.release();

        return frames;
    }

    /**
     * @brief Find athlete in video given by frames.
     * 
     * @param raw_frames Frames of video to be processed.
     * @param detector Detector used for finding athlete in video.
     * @param[out] found_frames Frames with detections' drawings.
     * 
     * @returns whether athlete was found.
     */
    bool find_athlete(const std::vector<cv::Mat> &raw_frames, body_detector &detector, std::vector<cv::Mat> &found_frames) const noexcept {
        std::vector<cv::Mat> frames;
        cv::Mat raw_frame, found_frame;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            raw_frame = raw_frames[frame_no].clone();
            found_frame = raw_frames[frame_no].clone();

            // Try to find athlete in current frame and draw detections.
            detector.find(raw_frame, frame_no);
            detector.draw(found_frame, frame_no);

            // Save unmodified and modified frames.
            frames.push_back(raw_frame);
            found_frames.push_back(found_frame);
        }
        return detector.is_found();
    }

    /**
     * @brief Detect athlete and his body parts in video given by frames.
     * 
     * @param raw_frames Frames of video to be processed.
     * @param detector Detector used for finding athlete in video.
     * @param fps Frame rate of processed video.
     * 
     * @returns frames with detections' drawings.
     */
    std::vector<cv::Mat> detect_athlete(const std::vector<cv::Mat> &raw_frames, body_detector &detector, double fps) const noexcept {
        std::vector<cv::Mat> frames;

        cv::Mat frame;
        body_detector::result res = body_detector::result::unknown;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Processing frame " << frame_no << std::endl;

            frame = raw_frames[frame_no].clone();

            if (res != body_detector::result::error) {
                // No error occured yet, process video further.

                // Detect athlete's body in current frame.
                res = detector.detect(frame, frame_no);
                if (res == body_detector::result::ok) {
                    // Detection on given frame was valid.

                    // Draw detections into frame.
                    detector.draw(frame, frame_no);
                }
            }

            // cv::imshow("frame", frame);
            // cv::waitKey();

            // Save current modified frame.
            frames.push_back(frame);
        }
        cv::destroyAllWindows();
        return frames;
    }

    /**
     * @brief Write frames as a video to given file.
     * 
     * @param filename Path to file, where modified video should be saved.
     * @param frames Frames which will be used to create video.
     */
    void write(const std::string &filename, const std::vector<cv::Mat> &frames) const noexcept {
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('D','I','V','X'), 30, cv::Size(frames.front().cols, frames.front().rows));
        if (writer.isOpened()) {
            for (const auto &f : frames)
                writer.write(f);
            writer.release();
        } else {
            std::cout << "Video could not be written to file " << filename << std::endl;
        }
    }

};