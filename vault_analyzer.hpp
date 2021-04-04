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

#include "person.hpp"

/**
 * @brief Analyzes detections made when processing video.
 */
class vault_analyzer {

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

    /// @brief Number of first frame in which person was detected.
    std::size_t first_frame;

    /// @brief Number of frames in video where person was detected.
    std::size_t frames;
    
    /// @brief Path to analyzed video.
    std::string filename;

    int dir;

    std::vector<parameter> parameters;

    /// @brief Returns position of person's body part.
    std::optional<cv::Point2d> get_part(const frame_body &body, body_part part) const {
        return body[part];
    }

    /// @brief Returns height of person's body part.
    std::optional<double> get_part_height(const frame_body &body, body_part part) const {
        auto p = get_part(body, part);
        if (p) return p->y;
        return std::nullopt;
    }

    /// @brief Returns height of person's left foot.
    std::optional<double> get_left_foot_height(const frame_body &body) const {
        return get_part_height(body, body_part::l_ankle);
    }

    /// @brief Returns height of person's right foot.
    std::optional<double> get_right_foot_height(const frame_body &body) const {
        return get_part_height(body, body_part::r_ankle);
    }

    /// @brief Returns person's hips position.
    std::optional<cv::Point2d> get_hips(const frame_body &body) const {
        // Average of left and right hip (if possible).
        if (body[body_part::r_hip] && body[body_part::l_hip]) {
            return ((*body[body_part::r_hip] + *body[body_part::l_hip]) / 2);
        } else if (body[body_part::r_hip]) {
            return *body[body_part::r_hip];
        } else if (body[body_part::l_hip]) {
            return *body[body_part::l_hip];
        } else {
            return std::nullopt;
        }
    }

    /**
     * @brief Returns hips height of person based on position of its body parts.
     * 
     * @param body Body parts of person to be processed.
     * 
     * @returns height of person's hips.
     */
    std::optional<double> get_hips_height(const frame_body &body) const {
        auto hips = get_hips(body);
        if (hips) return hips->y;
        return std::nullopt;
    }

    /**
     * 
     */
    std::optional<double> get_chest_tilt(const frame_body &body) const {
        auto hips = get_hips(body);
        auto chest = get_part(body, body_part::chest);
        if (hips && chest) {
            double y = hips->y - chest->y;
            // Check that athlete is not upside down.
            if (y > 0) {
                double x = (double)dir * (hips->x - chest->x);
                return std::atan(x / y) * 180.0 / M_PI;
            }
        }
        return std::nullopt;
    }

    /**
     * 
     */
    std::optional<double> get_shoulders_tilt(const frame_body &body) const {
        auto hips = get_hips(body);
        auto l_shoulder = get_part(body, body_part::l_shoulder);
        auto r_shoulder = get_part(body, body_part::r_shoulder);
        auto shoulders = l_shoulder + r_shoulder;
        if (hips && shoulders) {
            shoulders = *shoulders / 2;
            double y = hips->y - shoulders->y;
            // Check that athlete is not upside down.
            if (y > 0) {
                double x = (double)dir * (hips->x - shoulders->x);
                return std::atan(x / y) * 180.0 / M_PI;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Get parameter and its values from detected body parts.
     * 
     * @param name Name of parameter to be analyzed.
     * @param get_value Function extracting parameter's value from detected body parts in single frame.
     * @param real Uses transformed coordinates (based on background tracking) if true.
     * 
     * @returns parameter containing its name and values.
     */
    parameter get_parameter(std::string name, std::optional<double> (vault_analyzer::*get_value)(const frame_body &) const , bool real = false) const {
        video_body points = points_frame;
        std::string full_name = name + " (frame coordinates)";
        if (real) {
            points = points_real;
            full_name = name + " (real coordinates)";
        }
        std::vector<std::optional<double>> res;
        for (std::size_t i = 0; i < first_frame; i++)
            res.push_back(std::nullopt); // Skip frames without athlete detected.
        std::transform(points.begin(), points.end(), std::back_inserter(res),
                       [this, get_value](const frame_body &body){ return (this->*get_value)(body); }
        );
        return parameter(full_name, res);
    }

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

    // TODO: parameters not for each frame
    void compute_parameters() {
        parameters.push_back(get_parameter("Hips height", &vault_analyzer::get_hips_height));
        parameters.push_back(get_parameter("Hips height", &vault_analyzer::get_hips_height, true));
        parameters.push_back(get_parameter("Left foot height", &vault_analyzer::get_left_foot_height));
        parameters.push_back(get_parameter("Left foot height", &vault_analyzer::get_left_foot_height, true));
        parameters.push_back(get_parameter("Right foot height", &vault_analyzer::get_right_foot_height));
        parameters.push_back(get_parameter("Right foot height", &vault_analyzer::get_right_foot_height, true));
        parameters.push_back(get_parameter("Torso tilt (chest)", &vault_analyzer::get_chest_tilt));
        parameters.push_back(get_parameter("Torso tilt (shoulders)", &vault_analyzer::get_shoulders_tilt));
    }
    
    /**
     * @brief Write parameters in csv file.
     * 
     * Each column represents one parameter. Each frame has its value in one row.
     */
    void write_params(const std::vector<parameter> &parameters) const {
        std::ofstream output;
        output.open(create_output_filename());
        output << filename << std::endl;
        
        // Extract values from parameters.
        std::vector<std::string> names;
        std::vector<std::vector<std::optional<double>>> values;
        for (auto &param : parameters) {
            names.push_back(get_name(param));
            values.push_back(get_values(param));
        }
        output << "Frame number";
        // Write names of parameters.
        for (const auto &name : names) {
            output << "," << name;
        }
        // Write values of parameters.
        for (std::size_t i = 0; i < frames; i++) {
            output << std::endl << i;
            bool written = false;
            for (const auto &val : values) {
                output << ",";
                if (i < val.size()) {
                    if (val[i]) output << - *val[i];
                    written = true;
                }
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
        dir = athlete.move_analyzer.get_direction();
        frames = first_frame + points_frame.size();
        this->filename = filename;

        compute_parameters();
        //write_params();
    }

    std::vector<parameter> get_parameters() const {
        return parameters;
    }

};