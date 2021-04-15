#pragma once

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <ios>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <ostream>
#include <vector>

#include "forward.hpp"
#include "parameters.hpp"

/**
 * @brief Shows user frames of video and parameters' values for each frame.
 */
class viewer {
public:

    /**
     * @brief Default constructor.
     */
    viewer() noexcept {
        reset();
    }

    /**
     * @brief Show current frame of video.
     * 
     * Handle user inputs: left arrow shows previous frame, right arrow shows the next one,
     * space bar toggles whether to show detections drawings, escape key closes window and ends.
     * 
     * @param frames Frames with detections' drawings.
     * @param raw_frames Unmodified frames of processed video.
     * @param analyzer Analyzer after analyzing athlete's movement.
     */
    void show(const std::vector<cv::Mat> &frames, const std::vector<cv::Mat> &raw_frames, const vault_analyzer &analyzer) noexcept {
        reset(analyzer);
        
        std::cout << std::fixed << std::setprecision(2);
        for (;;) {
            std::stringstream label;
            label << "Frame " << frame_no << "/" << frames.size() - 1 << (drawing ? " with drawing" : "");
            cv::destroyAllWindows();
            cv::imshow(label.str(), drawing ? frames[frame_no] : raw_frames[frame_no]);

            write_parameters();

            switch (cv::waitKey()) {
                case 2:
                    // Left arrow pressed.
                    if (frame_no) {
                        --frame_no;
                    } else {
                        std::cout << "No frame before this one is available" << std::endl;
                    }
                    break;
                case 3:
                    // Right arrow pressed.
                    if (frame_no < frames.size() - 1) {
                        ++frame_no;
                    } else {
                        std::cout << "No frame after this one is available" << std::endl;
                    }
                    break;
                case 27:
                    // Esc key pressed.
                    cv::destroyAllWindows();
                    return;
                case 32:
                    // Space key pressed.
                    drawing = !drawing;
                    break;
            }
        }
    }

private:

    /**
     * @brief Current number of frame.
     */
    std::size_t frame_no;

    /**
     * @brief Whether to show detections' drawings.
     */
    bool drawing;

    /**
     * @brief Parameters and their values.
     */
    std::vector<std::shared_ptr<parameter>> parameters;

    /**
     * @brief Number of frame in which attempt begins.
     */
    std::size_t start;

    /**
     * @brief Number of frame in which athlete takes off.
     */
    std::size_t takeoff;

    /**
     * @brief Number of frame in which athlete's hips are highest.
     */
    std::size_t culmination;

    /**
     * @brief Set initial values.
     */
    void reset() noexcept {
        frame_no = 0;
        drawing = true;

    }

    /**
     * @brief Set initial values and values associated with analyzer.
     * 
     * @param analyzer Analyzer after analyzing athlete's movement.
     * 
     * @see analyzer.
     */
    void reset(const vault_analyzer &analyzer) noexcept {
        reset();
        parameters = analyzer.get_parameters();
        start = analyzer.get_start();
        takeoff = analyzer.get_takeoff();
        culmination = analyzer.get_culmination();
    }

    /**
     * @brief Get which vault parts are valid for current frame number.
     * 
     * @returns all vault parts valid for current frame number.
     */
    std::vector<vault_part> get_current_parts() const noexcept {
        std::vector<vault_part> res;
        // No frame part is valid.
        if (!start && !takeoff && !culmination) return res;

        if (frame_no < start) {
            // Before start.
            res.push_back(vault_part::invalid);
        }
        if (start <= frame_no && frame_no <= takeoff) {
            // Since start up to takeoff.
            res.push_back(vault_part::runup);
        }
        if (frame_no == takeoff) {
            // Takeoff.
            res.push_back(vault_part::takeoff);
        }
        if (takeoff <= frame_no && frame_no <= culmination) {
            // Since takeoff up to culmination.
            res.push_back(vault_part::vault);
        }
        if (frame_no > culmination) {
            // After culmination.
            res.push_back(vault_part::invalid);
        }
        return res;
    }

    /**
     * @brief Write parameters of current frame to standard output.
     * 
     * Write only parameters valid for current vault part.
     */
    void write_parameters() const noexcept {
        // Filter parameters valid for current frame.
        std::vector<vault_part> parts = get_current_parts();
        std::vector<std::shared_ptr<parameter>> to_write;
        for (auto &param : parameters) {
            for (auto part : parts) {
                if (part == param->part) {
                    to_write.push_back(param);
                }
            }
        }
        // Write parameters and their values.
        for (const auto &p : to_write) {
            std::cout << p->name << " : ";
            p->write_value(std::cout, frame_no, true);
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};