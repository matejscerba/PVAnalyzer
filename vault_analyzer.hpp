#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <optional>
#include <tuple>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <memory>

#include "parameters.hpp"
#include "person.hpp"

/**
 * @brief Analyzes detections made when processing video.
 * 
 * Holds detected athlete's body parts and values of corresponding parameters.
 */
class vault_analyzer {
public:

    /**
     * @brief Analyze athlete's movements in processed video.
     * 
     * Get detected body parts of athlete in whole video (not only in the part where athlete was detected).
     * 
     * @param athlete Instance of person to be analyzed (representing athlete).
     * @param filename Path to analyzed video.
     * @param frames Number of frames in video.
     */
    void analyze(const std::optional<person> &athlete, const std::string &filename, std::size_t frames, double fps) noexcept {
        if (athlete) {
            std::size_t before = athlete->first_frame_no;
            std::size_t after = frames - (athlete->first_frame_no + athlete->get_points().size());
            points_frame =
                video_body(before, frame_body(npoints, std::nullopt)) +
                athlete->get_points() +
                video_body(after, frame_body(npoints, std::nullopt));
            points_real =
                video_body(before, frame_body(npoints, std::nullopt)) +
                athlete->get_points(true) +
                video_body(after, frame_body(npoints, std::nullopt));
        } else {
            points_frame = video_body(frames, frame_body(npoints, std::nullopt));
            points_real = video_body(frames, frame_body(npoints, std::nullopt));
        }

        dir = athlete->move_analyzer.get_direction();
        this->filename = filename;
        this->fps = fps;

        compute_parameters(points_real);
        write_parameters();
    }

    /**
     * @brief Get parameters including their values.
     * 
     * @returns analyzed parameters.
     */
    std::vector<std::shared_ptr<parameter>> get_parameters() const noexcept {
        return parameters;
    }

    std::optional<std::size_t> get_start() const noexcept {
        return start;
    }

    std::optional<std::size_t> get_takeoff() const noexcept {
        return takeoff;
    }

    std::optional<std::size_t> get_culmination() const noexcept {
        return culmination;
    }

private:

    /**
     * @brief Body parts of person in each frame (not transformed).
     */
    video_body points_frame;

    /**
     * @brief Body parts of person in each frame transformed into real life coordinates.
     */
    video_body points_real;
    
    /**
     * @brief Path to analyzed video.
     */
    std::string filename;

    /**
     * @brief Athlete's movement direction.
     */
    direction dir;

    double fps;

    /**
     * @brief Analyzed parameters.
     */
    std::vector<std::shared_ptr<parameter>> parameters;

    std::optional<std::size_t> start;

    std::optional<std::size_t> takeoff;

    std::optional<std::size_t> culmination;

    /**
     * @brief Create name for output file from current date.
     * 
     * @returns name for output file.
     */
    std::string create_output_filename() const noexcept {
        std::time_t now = std::time(nullptr);
        std::stringstream sstr;
        sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
        return "parameters/" + sstr.str() + ".csv";
    }

    /**
     * @brief Compute values of parameters to be analyzed.
     */
    void compute_parameters(const video_body &points) noexcept {
        parameters.push_back(std::make_shared<hips_height>());
        parameters.push_back(std::make_shared<body_part_height>(body_part::l_ankle));
        parameters.push_back(std::make_shared<body_part_height>(body_part::r_ankle));
        parameters.push_back(std::make_shared<vertical_tilt>(
            "Torso tilt", body_part::l_hip, body_part::r_hip, body_part::neck, body_part::neck, dir));
        parameters.push_back(std::make_shared<vertical_tilt>(
            "Shoulders tilt", body_part::l_hip, body_part::r_hip, body_part::l_shoulder, body_part::r_shoulder, dir));
        auto steps_dur = std::make_shared<steps_duration>(fps);
        parameters.push_back(steps_dur);

        // Compute corresponding values.
        for (auto &param : parameters) {
            param->compute(points);
        }

        get_moments_of_interest(steps_dur, points);
    }

    std::optional<std::size_t> get_start(const video_body &points) noexcept {
        std::size_t index = 0;
        frame_part left = std::nullopt;
        frame_part right = std::nullopt;
        for (const auto &body : points) {
            std::optional<double> dist = distance(body[body_part::l_ankle], left);
            if (dist && *dist > 1) break;
            dist = distance(body[body_part::r_ankle], right);
            if (dist && *dist > 1) break;
            left = body[body_part::l_ankle];
            right = body[body_part::r_ankle];
            index++;
        }
        if (index && index != points.size())
            return index - 1;
        return std::nullopt;
    }

    std::optional<std::size_t> get_takeoff(std::shared_ptr<steps_duration> steps_duration) noexcept {
        return steps_duration->get_takeoff();
    }

    std::optional<std::size_t> get_culmination(const video_body &points) noexcept {
        std::optional<double> highest;
        std::size_t highest_idx;
        for (std::size_t i = 0; i < points.size(); i++) {
            if (points[i][body_part::l_hip] && points[i][body_part::r_hip]) {
                double current = (points[i][body_part::l_hip]->y + points[i][body_part::r_hip]->y) / 2;
                highest = highest ? std::min(current, *highest) : current;
                highest_idx = (highest == current ? i : highest_idx);
            }
        }
        return highest_idx;
    }

    /**
     * @brief Decide which frames represent start of attempt, takeoff and culmination above bar.
     */
    void get_moments_of_interest(std::shared_ptr<steps_duration> steps_duration, const video_body &points) noexcept {
        start = get_start(points);
        takeoff = get_takeoff(steps_duration);
        culmination = get_culmination(points);
    }
    
    /**
     * @brief Write parameters and their values in file.
     * 
     * Each column represents one parameter.
     * Frame-wise parameters have each value in row corresponding to frame,
     * multiple-valued parameters have values written from the first row,
     * single-valued parameters have value written in first row.
     * 
     * @note Units of parameters' values are omitted.
     */
    void write_parameters() const noexcept {
        std::ofstream output;
        std::string o_filename = create_output_filename();
        output.open(o_filename);
        if (!output.is_open()) {
            std::cout << "Output file \"" << o_filename << "\" could not be opened" << std::endl;
        }
        output << std::fixed << std::setprecision(2);
        output << filename << std::endl;
        
        output << "Frame number";
        // Write names of parameters.
        for (const auto &param : parameters) {
            output << "," << param->name;
        }

        // Write values of parameters.
        for (std::size_t i = 0; i < points_frame.size(); i++) {
            output << std::endl << i;
            for (const auto &param : parameters) {
                output << ",";
                // if (i < param->size()) {
                //     param->write_param(output, i, false);
                // }
            }
        }
        output.close();
    }

};