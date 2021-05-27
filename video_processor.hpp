#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "forward.hpp"
#include "athlete_finder.hpp"
#include "person.hpp"
#include "vault_analyzer.hpp"
#include "viewer.hpp"


/**
 * @brief Wrapper of whole program's functionality.
 * 
 * This class is intented to be used for default functionality. It processes
 * the video or model and passes individual frames to other parts of the program.
 */
class video_processor {
public:

    /**
     * @brief Process video at given path frame by frame.
     * 
     * Extract frames, find athlete, detect athlete's body
     * and analyze movements. Write output.
     * 
     * @param filename Path to video file to be processed.
     * @param find Find athlete automatically if true, let user select athlete
     *      manually if false.
     */
    void process_video(const std::string &filename, bool find) const {
        // Extract frames.
        double fps;
        std::vector<cv::Mat> raw_frames = extract_frames(filename, fps);
        if (!raw_frames.size()) return;

        athlete_finder finder(fps);

        // Find athlete.
        std::vector<cv::Mat> found_frames;
        std::optional<person> athlete = std::nullopt;
        if (find)
            athlete = finder.find_athlete(raw_frames, found_frames);
        else
            athlete = finder.select_athlete(raw_frames, fps);
        if (!athlete) {
            std::cout << "Athlete could not be found in video " << filename << std::endl;
            return;
        }

        // Detect athlete's body.
        std::vector<cv::Mat> frames = athlete->detect(raw_frames);

        // Create model from detections.
        model m(*athlete, filename);

        // Save output.
        write(raw_frames, "raw", filename, fps);
        write(found_frames, "found", filename, fps);
        write(frames, "detections", filename, fps);

        // Analyze movement.
        analyze(m, filename, fps, raw_frames, frames, true);
    }

    /**
     * @brief Process model of athlete's movement given by path.
     * 
     * Load model, extract frames of video which was used to create
     * model, draw athlete in frames, analyze athlete's movements.
     * 
     * @param filename Path to model to be processed.
     */
    void process_model(const std::string &filename) const {
        model m(filename);
        std::string video_filename;
        if (m.load(video_filename)) {
            double fps;
            std::vector<cv::Mat> raw_frames = extract_frames(video_filename, fps);
            std::vector<cv::Mat> frames = m.draw(raw_frames);
            analyze(m, video_filename, fps, raw_frames, frames, false);
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
    void analyze(   model &m,
                    const std::string filename,
                    double fps,
                    const std::vector<cv::Mat> &raw_frames,
                    const std::vector<cv::Mat> &frames,
                    bool save) const {

        // Analyze detected athlete.
        vault_analyzer analyzer;
        analyzer.analyze(m, filename, fps);
        if (save)
            m.save();

        // Show result.
        viewer v;
        v.show(frames, raw_frames, analyzer);
    }

    /**
     * @brief Extract frames from video given by its path.
     * 
     * @param filename Path to video.
     * @param[out] fps Frame rate of processed video.
     * 
     * @returns frames of video.
     */
    std::vector<cv::Mat> extract_frames(const std::string &filename, double &fps) const {
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

            // Resize and save current frame.
            frames.push_back(resize(frame));
        }
        video.release();

        return frames;
    }

    /**
     * @brief Write frames as a video to given file.
     * 
     * @param frames Frames which will be used to create video.
     * @param output_filename Name of output video without path and extension.
     * @param input_filename Filename of processed video.
     * @param fps Frame rate of processed and output video.
     */
    void write(const std::vector<cv::Mat> &frames, const std::string &output_filename, const std::string &input_filename, double fps) const {
        if (!frames.size()) return;
        std::string output_dir = get_output_dir(input_filename);
        std::string path = output_dir + output_filename + ".avi";
        cv::VideoWriter writer(
            path,
            cv::VideoWriter::fourcc('D','I','V','X'),
            fps,
            cv::Size(frames.front().cols, frames.front().rows)
        );
        if (writer.isOpened()) {
            for (const auto &f : frames)
                writer.write(f);
            writer.release();
        } else {
            std::cout << "Video could not be written to file \"" << path << "\"" << std::endl;
        }
    }

};