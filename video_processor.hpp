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
     * Extract frames, find athlete, detect athlete's body
     * and analyze movements. Write output.
     * 
     * @param filename Path to video file to be processed.
     * @param save Whether to save model of athlete's movements.
     */
    void process_video(const std::string &filename, bool save = true) const noexcept {
        double fps;
        std::vector<cv::Mat> raw_frames = extract_frames(filename, fps);

        body_detector detector(fps);

        std::vector<cv::Mat> found_frames;
        std::optional<person> athlete = detector.find_athlete(raw_frames, found_frames);
        std::string output_filename = "outputs/videos/" + create_output_filename();
        std::string ext = ".avi";
        // write(output_filename + "_raw_frames" + ext, raw_frames);
        // write(output_filename + "_found_frames" + ext, found_frames);
        if (!athlete) {
            std::cout << "Athlete could not be found in video " << filename << std::endl;
            return;
        }
        std::vector<cv::Mat> frames = athlete->detect(raw_frames);

        write(output_filename + "_frames" + ext, frames);

        model m(*athlete, filename);
        // if (save) m.save();

        // analyze(m, filename, fps, raw_frames, frames);
    }

    /**
     * @brief Process model of athlete's movement given by path.
     * 
     * Load model, extract frames of video which was used to create
     * model, draw athlete in frames, analyze athlete's movements.
     * 
     * @param filename Path to model to be processed.
     */
    void process_model(const std::string &filename) const noexcept {
        model m(filename);
        std::string video_filename;
        if (m.load(video_filename)) {
            double fps;
            std::vector<cv::Mat> raw_frames = extract_frames(video_filename, fps);
            std::vector<cv::Mat> frames = m.draw(raw_frames);
            analyze(m, video_filename, fps, raw_frames, frames);
        }
    }

private:

    /**
     * @brief Analyze model of athlete's movement.
     * 
     * @param m Model of athlete's movement.
     * @param filename Path to processed video.
     * @param fps Frame rate of processed video.
     * @param raw_frames Unmodified frames of video.
     * @param frames Frames of video with detections drawings.
     */
    void analyze(   const model &m,
                    const std::string filename,
                    double fps,
                    const std::vector<cv::Mat> &raw_frames,
                    const std::vector<cv::Mat> &frames) const noexcept {

        // Analyze detected athlete.
        vault_analyzer analyzer;
        analyzer.analyze(m, filename, fps);

        // Show result.
        viewer v;
        v.show(frames, raw_frames, analyzer);
        cv::destroyAllWindows();
    }

    /**
     * @brief Extract frames from video given by its path.
     * 
     * @param filename Path to video.
     * @param[out] fps Frame rate of processed video.
     * 
     * @returns frames of video.
     */
    std::vector<cv::Mat> extract_frames(const std::string &filename, double &fps) const noexcept {
        std::vector<cv::Mat> frames;

        // Open video.
        cv::VideoCapture video;
        if (!video.open(filename)) {
            std::cout << "Error opening video " << filename << std::endl;
            return frames;
        }

        fps = video.get(cv::CAP_PROP_FPS);

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
     * @brief Write frames as a video to given file.
     * 
     * @param filename Path to file, where video should be saved.
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