#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <ios>
#include <iomanip>
#include <iostream>
#include <memory>

#include "forward.hpp"
#include "parameters.hpp"

class visual {
public:

    /**
     * @brief Default constructor.
     */
    visual( const std::vector<cv::Mat> &frames,
            const std::vector<cv::Mat> &raw_frames,
            vault_analyzer &analyzer) :
                frames(frames),
                raw_frames(raw_frames) {
                    parameters = analyzer.get_parameters();
                    start = analyzer.get_start();
                    takeoff = analyzer.get_takeoff();
                    culmination = analyzer.get_culmination();
                }

    /**
     * @brief Show current frame of video.
     * 
     * Handle user inputs: left arrow shows previous frame, right arrow shows the next one,
     * space bar toggles whether to show detected rectangles and points, escape key closes window.
     */
    void show() {
        std::cout << std::fixed << std::setprecision(2);
        for (;;) {
            std::stringstream label;
            label << "Frame " << frame_no << "/" << frames.size() - 1 << (drawing ? " with drawing" : "");
            cv::destroyAllWindows();
            cv::imshow(label.str(), drawing ? frames[frame_no] : raw_frames[frame_no]);
            write_parameters();
            switch (cv::waitKey()) {
                case 2:
                    // Left arrow.
                    if (frame_no) {
                        frame_no--;
                    } else {
                        std::cout << "No frame before this one is available" << std::endl;
                    }
                    break;
                case 3:
                    // Right arrow.
                    if (frame_no < frames.size() - 1) {
                        frame_no++;
                    } else {
                        std::cout << "No frame after this one is available" << std::endl;
                    }
                    break;
                case 27:
                    // Esc.
                    cv::destroyAllWindows();
                    return;
                case 32:
                    // Space.
                    drawing = !drawing;
                    break;
            }
        }
    }

private:

    std::size_t frame_no = 0;
    bool drawing = true;
    const std::vector<cv::Mat> frames;
    const std::vector<cv::Mat> raw_frames;
    std::vector<std::shared_ptr<parameter>> parameters;

    std::optional<std::size_t> start;

    std::optional<std::size_t> takeoff;

    std::optional<std::size_t> culmination;

    std::vector<vault_part> get_current_parts() const noexcept {
        std::vector<vault_part> res;
        if (start && frame_no < *start) {
            res.push_back(vault_part::invalid);
        }
        if (start && takeoff && *start <= frame_no && frame_no <= *takeoff) {
            res.push_back(vault_part::runup);
        }
        if (takeoff && frame_no == *takeoff) {
            res.push_back(vault_part::takeoff);
        }
        if (takeoff && culmination && takeoff <= frame_no && frame_no <= culmination) {
            res.push_back(vault_part::vault);
        }
        if (culmination && frame_no > culmination) {
            res.push_back(vault_part::invalid);
        }
        return res;
    }

    /**
     * @brief Write parameters of current frame to standard output.
     */
    void write_parameters() const {
        std::vector<vault_part> parts = get_current_parts();
        std::vector<std::shared_ptr<parameter>> to_write;
        for (auto &param : parameters) {
            for (auto part : parts) {
                if (part == param->part) {
                    to_write.push_back(param);
                }
            }
        }
        for (auto part : parts) {
            if (part == vault_part::invalid) {
                std::cout << "invalid" << std::endl;
            }
            if (part == vault_part::runup) {
                std::cout << "runup" << std::endl;
            }
            if (part == vault_part::takeoff) {
                std::cout << "takeoff" << std::endl;
            }
            if (part == vault_part::vault) {
                std::cout << "vault" << std::endl;
            }
        }
        for (const auto &p : to_write) {
            std::cout << p->name << " : ";
            p->write_value(std::cout, frame_no, true);
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};