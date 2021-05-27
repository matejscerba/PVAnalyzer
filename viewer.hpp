#pragma once

#include <opencv2/opencv.hpp>
#include <Python.h>

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
    viewer() {
        reset();
    }

    /**
     * @brief Show frames of video.
     * 
     * Handle user inputs: left arrow shows previous frame, right arrow shows the next one,
     * space bar toggles whether to show detections drawings, escape key closes window and ends.
     * Show important moments of vault, write parameters' values to standard output.
     * 
     * @param frames Frames with detections' drawings.
     * @param raw_frames Unmodified frames of processed video.
     * @param analyzer Analyzer after analyzing athlete's movement.
     */
    void show(const std::vector<cv::Mat> &frames, const std::vector<cv::Mat> &raw_frames, const vault_analyzer &analyzer) {
        if (!frames.size() || frames.size() != raw_frames.size()) {
            std::cout << "Unable to show frames" << std::endl;
            return;
        }

        reset(analyzer);

        cv::imshow("start", raw_frames[start]);
        cv::imshow("takeoff", raw_frames[takeoff]);
        cv::imshow("culmination", raw_frames[culmination]);
        
        std::cout << std::fixed << std::setprecision(2);
        for (;;) {
            cv::imshow("window", drawing ? frames[frame_no] : raw_frames[frame_no]);

            std::cout << "Frame " << frame_no + 1 << "/" << frames.size() << ":" << std::endl;
            write_parameters();

            switch (cv::waitKey()) {
                case 2:
                    // Left arrow pressed.
                    frame_no = std::max(--frame_no, 0);
                    break;
                case 3:
                    // Right arrow pressed.
                    frame_no = std::min(++frame_no, (int)frames.size() - 1);
                    break;
                case 27:
                    // Esc key pressed.
                    cv::destroyAllWindows();
                    cv::waitKey(1); // Forces windows to close immediately.
                    show_parameters();
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
    int frame_no;

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
     * @brief Path to file containing values of all analyzed parameters.
     */
    std::string params_filename;

    /**
     * @brief Set initial values.
     */
    void reset() {
        frame_no = 0;
        drawing = true;
        parameters.clear();
        start = 0;
        takeoff = 0;
        culmination = 0;
        params_filename = "";
    }

    /**
     * @brief Set initial values and values associated with analyzer.
     * 
     * @param analyzer Analyzer after analyzing athlete's movement.
     * 
     * @see analyzer.
     */
    void reset(const vault_analyzer &analyzer) {
        reset();
        parameters = analyzer.get_parameters();
        start = analyzer.get_start();
        takeoff = analyzer.get_takeoff();
        culmination = analyzer.get_culmination();
        params_filename = analyzer.get_params_filename();
    }

    /**
     * @brief Get which vault parts are valid for current frame number.
     * 
     * @returns all vault parts valid for current frame number.
     */
    std::vector<vault_part> get_current_parts() const {
        std::vector<vault_part> res{vault_part::all};
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
    void write_parameters() const {
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

    /**
     * @brief Run python script to plot parameters from generated file.
     */
    void show_parameters() const {
        Py_Initialize();

        FILE *file = fopen("plot_parameters.py", "r");
        if(file) {
            wchar_t *argv[3] = {
                Py_DecodeLocale("plot_parameters.py", NULL),
                Py_DecodeLocale("--file", NULL),
                Py_DecodeLocale(params_filename.c_str(), NULL)
            };
            PySys_SetArgv(3, argv);
            PyRun_SimpleFile(file, "plot_parameters.py");
            fclose(file);
        }

        Py_Finalize();
    }

};