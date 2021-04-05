#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <iostream>

#include "forward.hpp"


class visual {
public:

    /**
     * @brief Default constructor.
     */
    visual(const std::vector<cv::Mat> &frames, const std::vector<cv::Mat> &raw_frames, const std::vector<parameter> &parameters)
        : frames(frames), raw_frames(raw_frames), parameters(parameters) {}

    /**
     * @brief Show current frame of video.
     * 
     * Handle user inputs: left arrow shows previous frame, right arrow shows the next one,
     * space bar toggles whether to show detected rectangles and points, escape key closes window.
     */
    void show() {
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
    const std::vector<parameter> parameters;

    /**
     * @brief Write parameters of current frame to standard output.
     */
    void write_parameters() const {
        for (const auto &p : parameters) {
            if (frame_no < get_values(p).size()) {
                std::cout << get_name(p) << " : " <<  get_values(p)[frame_no] << std::endl;
            }
        }
        std::cout << std::endl;
    }

};