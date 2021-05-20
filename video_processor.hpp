#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

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
    void process_video(const std::string &filename, bool find) const noexcept {
        // Extract frames.
        double fps;
        std::vector<cv::Mat> raw_frames = extract_frames(filename, fps);
        if (!raw_frames.size()) return;

        body_detector detector(fps);

        // Find athlete.
        std::vector<cv::Mat> found_frames;
        std::optional<person> athlete = std::nullopt;
        if (find)
            athlete = detector.find_athlete(raw_frames, found_frames);
        else
            athlete = select_athlete(raw_frames, fps, filename);
        if (!athlete) {
            std::cout << "Athlete could not be found in video " << filename << std::endl;
            return;
        }

        // Detect athlete's body.
        std::vector<cv::Mat> frames = athlete->detect(raw_frames);

        // Create model from detections.
        model m(*athlete, filename);

        // Save output.
        m.save();
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
    void process_model(const std::string &filename) const noexcept {
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
     * @brief Let user select athlete's bounding box in video.
     * 
     * @param frames Frames of video to select athlete from.
     * @param fps Frame rate of processed video.
     * @returns athlete if user selected valid bounding box.
     */
    std::optional<person> select_athlete(const std::vector<cv::Mat> &frames, double fps, const std::string &filename) const noexcept {
        if (filename == "data/BP_videos/1.MOV") {
            return person(0, cv::Rect(695, 329, 49, 79), fps);
        } else if (filename == "data/BP_videos/2.MOV") {
            return person(0, cv::Rect(757, 331, 46, 97), fps);
        } else if (filename == "data/BP_videos/3.MOV") {
            return person(0, cv::Rect(855, 334, 173, 232), fps);
        } else if (filename == "data/BP_videos/4.MOV") {
            return person(0, cv::Rect(741, 201, 226, 315), fps);
        } else if (filename == "data/BP_videos/5.MOV") {
            return person(0, cv::Rect(830, 345, 99, 147), fps);
        } else if (filename == "data/BP_videos/6.MOV") {
            return person(0, cv::Rect(804, 245, 231, 325), fps);
        } else if (filename == "data/BP_videos/7.MOV") {
            return person(0, cv::Rect(770, 340, 68, 139), fps);
        } else if (filename == "data/BP_videos/8.MOV") {
            return person(0, cv::Rect(646, 315, 99, 158), fps);
        } else if (filename == "data/BP_videos/9.MOV") {
            return person(0, cv::Rect(641, 274, 231, 296), fps);
        } else if (filename == "data/BP_videos/10.MOV") {
            return person(0, cv::Rect(468, 333, 97, 149), fps);
        } else if (filename == "data/BP_videos/11.MOV") {
            return person(0, cv::Rect(48, 642, 83, 154), fps);
        } else if (filename == "data/BP_videos/12.MOV") {
            return person(0, cv::Rect(434, 309, 145, 256), fps);
        } else if (filename == "data/BP_videos/13.MOV") {
            return person(0, cv::Rect(227, 616, 67, 143), fps);
        } else if (filename == "data/BP_videos/14.MOV") {
            return person(0, cv::Rect(472, 280, 146, 280), fps);
        } else if (filename == "data/BP_videos/15.MOV") {
            return person(0, cv::Rect(675, 353, 76, 144), fps);
        } else if (filename == "data/BP_videos/16.MOV") {
            return person(0, cv::Rect(610, 297, 204, 289), fps);
        } else if (filename == "data/BP_videos/17.MOV") {
            return person(0, cv::Rect(679, 342, 81, 157), fps);
        } else if (filename == "data/BP_videos/18.MOV") {
            return person(0, cv::Rect(684, 332, 105, 151), fps);
        } else if (filename == "data/BP_videos/19.MOV") {
            return person(133, cv::Rect(690, 371, 116, 150), fps);
        } else if (filename == "data/BP_videos/20.MOV") {
            return person(0, cv::Rect(616, 347, 93, 155), fps);
        } else if (filename == "data/BP_videos/21.MOV") {
            return person(0, cv::Rect(669, 291, 118, 189), fps);
        } else if (filename == "data/BP_videos/22.MOV") {
            return person(0, cv::Rect(613, 344, 136, 197), fps);
        } else if (filename == "data/BP_videos/23.MOV") {
            return person(0, cv::Rect(646, 303, 104, 189), fps);
        } else if (filename == "data/BP_videos/24.MOV") {
            return person(0, cv::Rect(581, 249, 76, 163), fps);
        } else if (filename == "data/BP_videos/25.MOV") {
            return person(0, cv::Rect(501, 211, 60, 87), fps);
        } else if (filename == "data/BP_videos/26.MOV") {
            return person(0, cv::Rect(770, 220, 60, 93), fps);
        } else if (filename == "data/BP_videos/27_L.mp4") {
            return person(0, cv::Rect(733, 249, 46, 79), fps);
        } else if (filename == "data/BP_videos/27_P.MOV") {
            return person(0, cv::Rect(202, 356, 57, 90), fps);
        } else if (filename == "data/BP_videos/28_odraz.mp4") {
            return person(0, cv::Rect(472, 258, 45, 67), fps);
        } else if (filename == "data/BP_videos/28_stojany.MOV") {
            return person(0, cv::Rect(259, 340, 45, 67), fps);
        } else if (filename == "data/BP_videos/29.MOV") {
            return person(0, cv::Rect(687, 326, 43, 75), fps);
        } else if (filename == "data/BP_videos/30.mp4") {
            return person(0, cv::Rect(524, 224, 34, 59), fps);
        } else if (filename == "data/BP_videos/31.MP4") {
            return person(0, cv::Rect(555, 271, 90, 141), fps);
        } else if (filename == "data/kolin_short.mp4") {
            return person(0, cv::Rect(74, 329, 108, 188), fps);
        }
        return std::nullopt;
        // return person(0, cv::Rect(785, 347, 38, 79), fps);
        // cv::Rect bbox;
        // for (std::size_t frame_no = 0; frame_no < frames.size(); ++frame_no) {
        //     std::cout << "Skip to next frame by pressing the c button!" << std::endl;
        //     bbox = cv::selectROI("Select athlete", frames[frame_no], false, false);
        //     if (bbox.width > 0 && bbox.height > 0) {
        //         std::cout << filename << bbox << frame_no << std::endl;
        //         cv::waitKey(1);
        //         cv::destroyAllWindows();
        //         cv::waitKey(1);
        //         return person(frame_no, bbox, fps);
        //     }
        // }
        // cv::waitKey(1);
        // cv::destroyAllWindows();
        // cv::waitKey(1);
        // return std::nullopt;
    }

    /**
     * @brief Analyze model of athlete's movement.
     * 
     * @param m Model of athlete's movement.
     * @param filename Path to processed video.
     * @param fps Frame rate of processed video.
     * @param raw_frames Unmodified frames of video.
     * @param frames Frames of video with detections drawings.
     * @param save Whether to save output.
     */
    void analyze(   const model &m,
                    const std::string filename,
                    double fps,
                    const std::vector<cv::Mat> &raw_frames,
                    const std::vector<cv::Mat> &frames,
                    bool save) const noexcept {

        // Analyze detected athlete.
        vault_analyzer analyzer;
        analyzer.analyze(m, filename, fps, save);

        // Show result.
        // viewer v;
        // v.show(frames, raw_frames, analyzer);
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

            // Resize and save current frame.
            frames.push_back(resize(frame));

            // Skip frames of tested 120-fps video for efficiency reasons.
            // TODO: remove
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
     * @param frames Frames which will be used to create video.
     * @param output_filename
     * @param input_filename
     * @param fps
     */
    void write(const std::vector<cv::Mat> &frames, const std::string &output_filename, const std::string &input_filename, double fps) const noexcept {
        if (!frames.size()) return;
        std::string output_dir = get_output_dir(input_filename);
        std::string path = output_dir + "/" + output_filename + ".avi";
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