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
 */
class vault_analyzer {
public:

    /**
     * @brief Analyze athlete's movements in analyzed video.
     * 
     * @param athlete Athlete to be analyzed.
     * @param filename Path to analyzed video.
     */
    void analyze(const person &athlete, const std::string &filename) {
        points_frame = video_body(athlete.first_frame_no, frame_body(npoints, std::nullopt)) + athlete.get_points();
        points_real = video_body(athlete.first_frame_no, frame_body(npoints, std::nullopt)) + athlete.get_points(true);
        dir = athlete.move_analyzer.get_direction();
        this->filename = filename;

        compute_parameters();
        //write_parameters();
    }

    /**
     * @brief Get computed parameters.
     * 
     * @returns computed parameters.
     */
    std::vector<std::shared_ptr<parameter>> get_parameters() const {
        return parameters;
    }

private:

    /**
     * @brief Body parts of person in each frame (untransformed).
     * 
     * @see forward.hpp.
     */
    video_body points_frame;

    /**
     * @brief Body parts of person in each frame transformed into real life coordinates.
     * 
     * @see forward.hpp.
     */
    video_body points_real;
    
    /// @brief Path to analyzed video.
    std::string filename;

    int dir;

    std::vector<std::shared_ptr<parameter>> parameters;

    /**
     * @brief Create name for output file from current date.
     * 
     * @returns name for output .csv file.
     */
    std::string create_output_filename() const {
        std::time_t now = std::time(nullptr);
        std::stringstream sstr;
        sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
        return "outputs/" + sstr.str() + ".csv";
    }

    void compute_parameters() {
        parameters.push_back(std::make_shared<hips_height_parameter>(points_real));
        parameters.push_back(std::make_shared<body_part_height_parameter>(points_real, body_part::l_ankle));
        parameters.push_back(std::make_shared<body_part_height_parameter>(points_real, body_part::r_ankle));
        parameters.push_back(std::make_shared<vertical_tilt_parameter>(
            points_real, "Torso tilt", body_part::l_hip, body_part::r_hip, body_part::neck, body_part::neck, dir));
        parameters.push_back(std::make_shared<vertical_tilt_parameter>(
            points_real, "Shoulders tilt", body_part::l_hip, body_part::r_hip, body_part::l_shoulder, body_part::r_shoulder, dir));
    }
    
    /**
     * @brief Write parameters in csv file.
     * 
     * Each column represents one parameter. Each frame has its value in one row.
     */
    void write_parameters(const std::vector<parameter> &parameters) const {
        // std::ofstream output;
        // output.open(create_output_filename());
        // output << filename << std::endl;
        
        // // Extract values from parameters.
        // std::vector<std::string> names;
        // std::vector<std::vector<std::optional<double>>> values;
        // for (auto &param : parameters) {
        //     names.push_back(get_name(param));
        //     values.push_back(get_values(param));
        // }
        // output << "Frame number";
        // // Write names of parameters.
        // for (const auto &name : names) {
        //     output << "," << name;
        // }
        // // Write values of parameters.
        // for (std::size_t i = 0; i < points_frame.size(); i++) {
        //     output << std::endl << i;
        //     bool written = false;
        //     for (const auto &val : values) {
        //         output << ",";
        //         if (i < val.size()) {
        //             if (val[i]) output << - *val[i];
        //             written = true;
        //         }
        //     }
        //     if (!written) {
        //         break;
        //     }
        // }
        // output.close();
    }

};