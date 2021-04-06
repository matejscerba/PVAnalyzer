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
    void analyze(const person &athlete, const std::string &filename, std::size_t frames) noexcept {
        std::size_t before = athlete.first_frame_no;
        std::size_t after = frames - (athlete.first_frame_no + athlete.get_points().size());
        points_frame =
            video_body(before, frame_body(npoints, std::nullopt)) +
            athlete.get_points() +
            video_body(after, frame_body(npoints, std::nullopt));
        points_real =
            video_body(before, frame_body(npoints, std::nullopt)) +
            athlete.get_points(true) +
            video_body(after, frame_body(npoints, std::nullopt));

        dir = athlete.move_analyzer.get_direction();
        this->filename = filename;

        compute_parameters();
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
    int dir;

    /**
     * @brief Analyzed parameters.
     */
    std::vector<std::shared_ptr<parameter>> parameters;

    /**
     * @brief Create name for output file from current date.
     * 
     * @returns name for output file.
     */
    std::string create_output_filename() const noexcept {
        std::time_t now = std::time(nullptr);
        std::stringstream sstr;
        sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
        return "outputs/" + sstr.str() + ".csv";
    }

    /**
     * @brief Compute values of parameters to be analyzed.
     */
    void compute_parameters() noexcept {
        parameters.push_back(std::make_shared<hips_height_parameter>());
        parameters.push_back(std::make_shared<body_part_height_parameter>(body_part::l_ankle));
        parameters.push_back(std::make_shared<body_part_height_parameter>(body_part::r_ankle));
        parameters.push_back(std::make_shared<vertical_tilt_parameter>(
            "Torso tilt", body_part::l_hip, body_part::r_hip, body_part::neck, body_part::neck, dir));
        parameters.push_back(std::make_shared<vertical_tilt_parameter>(
            "Shoulders tilt", body_part::l_hip, body_part::r_hip, body_part::l_shoulder, body_part::r_shoulder, dir));

        // Compute corresponding values.
        for (auto &param : parameters) {
            param->compute(points_real);
        }
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
                if (i < param->size()) {
                    param->write_value(output, i, false);
                }
            }
        }
        output.close();
    }

};