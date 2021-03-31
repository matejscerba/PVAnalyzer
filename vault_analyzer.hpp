#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <optional>
#include <tuple>
#include <ctime>
#include <iomanip>

#include "person.hpp"

class vault_analyzer {

    /**
     * @brief Body parts of person in each frame transformed into correct coordinates.
     * 
     * @see forward.hpp.
     */
    video_body points_frame;

    video_body points_real;

    std::size_t first_frame;

    std::size_t frames;
    
    /// @brief Path to analyzed video.
    std::string filename;

    cv::Point2d get_part(const frame_body &body, body_part part) const {
        if (body[part]) return *body[part];
        return cv::Point2d();
    }

    double get_part_height(const frame_body &body, body_part part) const {
        return get_part(body, part).y;
    }

    double get_left_foot_height(const frame_body &body) const {
        return get_part_height(body, body_part::l_ankle);
    }

    double get_right_foot_height(const frame_body &body) const {
        return get_part_height(body, body_part::r_ankle);
    }

    /// @brief Returns centers of gravity of person in each frame.
    cv::Point2d get_hips(const frame_body &body) const {
        // Average of left and right hip (if possible).
        if (body[body_part::r_hip] && body[body_part::l_hip]) {
            return ((*body[body_part::r_hip] + *body[body_part::l_hip]) / 2);
        } else if (body[body_part::r_hip]) {
            return *body[body_part::r_hip];
        } else if (body[body_part::l_hip]) {
            return *body[body_part::l_hip];
        } else {
            return cv::Point2d();
        }
    }

    /// @brief Returns centers of gravity of person in each frame.
    double get_hips_height(const frame_body &body) const {
        return get_hips(body).y;
    }

    parameter get_parameter(std::string name, double (vault_analyzer::*get_value)(const frame_body &) const , bool real = false) const {
        video_body points = points_frame;
        std::string full_name = name + " (frame coordinates)";
        if (real) {
            points = points_real;
            full_name = name + " (real coordinates)";
        }
        std::vector<double> res;
        for (std::size_t i = 0; i < first_frame; i++)
            res.emplace_back(); // Skip frames without athlete detected.
        std::transform(points.begin(), points.end(), std::back_inserter(res),
                       [this, get_value](const frame_body &body){ return (this->*get_value)(body); }
        );
        return parameter(full_name, res);
    }

    std::string create_output_filename() const {
        std::time_t now = std::time(nullptr);
        std::stringstream sstr;
        sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
        return "outputs/" + sstr.str() + ".csv";
    }

    /**
     * @brief Write parameters in csv file.
     * 
     * Needs to be improved...
     */
    void write_params() const {
        std::vector<parameter> parameters;
        parameters.push_back(get_parameter("Hips height", &vault_analyzer::get_hips_height));
        parameters.push_back(get_parameter("Hips height", &vault_analyzer::get_hips_height, true));
        parameters.push_back(get_parameter("Left foot height", &vault_analyzer::get_left_foot_height));
        parameters.push_back(get_parameter("Left foot height", &vault_analyzer::get_left_foot_height, true));
        parameters.push_back(get_parameter("Right foot height", &vault_analyzer::get_right_foot_height));
        parameters.push_back(get_parameter("Right foot height", &vault_analyzer::get_right_foot_height, true));

        std::ofstream output;
        output.open(create_output_filename());
        output << filename << std::endl;
        
        // Extract values from parameters.
        std::vector<std::string> names;
        std::vector<std::vector<double>> values;
        for (auto &param : parameters) {
            names.push_back(get_name(param));
            values.push_back(get_values(param));
        }
        // Write names of parameters.
        for (const auto &name : names) {
            output << name << ",";
        }
        // Write values of parameters.
        for (std::size_t i = 0; i < frames; i++) {
            output << std::endl;
            bool written = false;
            for (const auto &val : values) {
                if (i < val.size()) {
                    output << val[i];
                    written = true;
                }
                output << ",";
            }
            if (!written) {
                break;
            }
        }
        output.close();
    }

public:

    /**
     * @brief Analyze athlete's movements in analyzed video.
     * 
     * @param athlete Athlete to be analyzed.
     * @param filename Path to analyzed video.
     */
    void analyze(const person &athlete, const std::string &filename) {
        points_frame = athlete.get_points();
        points_real = athlete.get_points(true);
        first_frame = athlete.first_frame_no;
        frames = first_frame + points_frame.size();
        this->filename = filename;
        write_params();
    }

};