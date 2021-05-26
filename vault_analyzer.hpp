#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "parameters.hpp"
#include "model.hpp"

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
     * Get detected body parts of athlete in whole video (not only in the part where athlete was detected)
     * and use these detections to compute parameters' values.
     * 
     * @param athlete Athlete to be analyzed.
     * @param filename Path to analyzed video.
     * @param fps Frame rate of processed video.
     */
    void analyze(model &athlete, const std::string &filename, double fps) {
        points_frame = athlete.get_frame_points();
        points_real = athlete.get_real_points();

        dir = athlete.get_direction();
        this->filename = filename;
        this->fps = fps;

        if (compute_parameters(points_real, athlete)) {
            write_parameters();
        }
    }

    /**
     * @brief Get parameters including their values.
     */
    std::vector<std::shared_ptr<parameter>> get_parameters() const {
        return parameters;
    }

    /**
     * @brief Get number of frame in which attempt begins.
     */
    std::size_t get_start() const {
        return start;
    }

    /**
     * @brief Get number of frame in which athlete takes off.
     */
    std::size_t get_takeoff() const {
        return takeoff;
    }

    /**
     * @brief Get number of frame in which athlete's hips are highest.
     */
    std::size_t get_culmination() const {
        return culmination;
    }

    /**
     * @brief Get path to file where parameters will be saved.
     */
    std::string get_params_filename() const {
        return get_output_dir(filename) + PARAMETERS_FILE;
    }

private:

    /**
     * @brief Body parts of person in each frame (not transformed).
     */
    model_video_points points_frame;

    /**
     * @brief Body parts of person in each frame transformed into real life coordinates.
     */
    model_video_points points_real;
    
    /**
     * @brief Path to analyzed video.
     */
    std::string filename;

    /**
     * @brief Athlete's movement direction.
     */
    direction dir;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Analyzed parameters.
     */
    std::vector<std::shared_ptr<parameter>> parameters;

    /**
     * @brief Number of frame in which attempt begins.
     */
    std::size_t start = 0;

    /**
     * @brief Number of frame in which athlete takes off.
     */
    std::size_t takeoff = 0;

    /**
     * @brief Number of frame in which athlete's hips are highest.
     */
    std::size_t culmination = 0;

    /**
     * @brief Update real life coordinates.
     * 
     * Shift coordinates so that (0,0,0) corresponds to feet touching ground in last step.
     * 
     * @param points Input points.
     * @returns shifted points.
     */
    model_point get_shift(const model_video_points &points) const {
        model_point left = points[takeoff][body_part::l_ankle];
        model_point right = points[takeoff][body_part::r_ankle];
        return get_part(left, right, std::less<double>()); // Lower foot.
    }

    /**
     * @brief Compute values of parameters to be analyzed.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * @param athlet Model representing athlete's movement.
     * @returns true if computation was successful (if beginning of attampt,
     * takeoff and culmination moments were found).
     */
    bool compute_parameters(const model_video_points &points, model &athlete) {
        if (!find_moments_of_interest(points))
            return false;

        athlete.update_coords(get_shift(points));

        parameters.push_back(std::make_shared<hips_height>());
        parameters.push_back(std::make_shared<body_part_height>(body_part::l_ankle));
        parameters.push_back(std::make_shared<body_part_height>(body_part::r_ankle));
        parameters.push_back(std::make_shared<vertical_tilt>(
            "Torso tilt", body_part::l_hip, body_part::r_hip, body_part::head, body_part::head));
        parameters.push_back(std::make_shared<steps_duration>(fps));
        parameters.push_back(std::make_shared<steps_angle>(fps));
        parameters.push_back(std::make_shared<hips_velocity_loss>(takeoff, fps));
        parameters.push_back(std::make_shared<shoulders_velocity_loss>(takeoff, fps));
        parameters.push_back(std::make_shared<takeoff_angle>(takeoff, fps));

        // Compute corresponding values.
        for (auto &param : parameters) {
            param->compute(athlete.get_real_points());
        }

        // Sort parameters alphabetically.
        std::sort(parameters.begin(), parameters.end());

        return true;
    }

    /**
     * @brief Find frame number in which attempt begins.
     * 
     * Attempt begins once at least one athlete's ankle moves.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * @returns frame number in which attempt begins, no value if such
     * frame could not be found.
     * 
     * @note Returns last frame in which ankles are static.
     */
    std::optional<std::size_t> find_start(const model_video_points &points) {
        std::size_t index = 0;
        // Values in previous frame.
        model_point left = std::nullopt;
        model_point right = std::nullopt;
        for (const auto &body : points) {
            std::optional<double> dist = distance(body[body_part::l_ankle], left);
            if (dist && *dist > 1) break;
            dist = distance(body[body_part::r_ankle], right);
            if (dist && *dist > 1) break;
            left = body[body_part::l_ankle];
            right = body[body_part::r_ankle];
            ++index;
        }
        if (index && index != points.size())
            return index - 1;
        return std::nullopt;
    }

    /**
     * @brief Find frame number in which athlete takes off.
     * 
     * Takeoff is the same as last step.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * @returns frame number in which athlete takes off, no value if no such
     * frame was found.
     */
    std::optional<std::size_t> find_takeoff(const model_video_points &points) {
        std::vector<std::size_t> steps = get_step_frames(points, fps);
        if (steps.size()) return steps.back();
        return std::nullopt;
    }

    /**
     * @brief Find frame number in which athlete's hips are highest.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * @returns frame number in which athlete's hips are highest, no value if no such
     * frame was found.
     */
    std::optional<std::size_t> find_culmination(const model_video_points &points) {
        std::optional<double> highest;
        std::size_t highest_idx;
        bool found = false;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (points[i][body_part::l_hip] && points[i][body_part::r_hip]) {
                found = true;
                double current = (points[i][body_part::l_hip]->z + points[i][body_part::r_hip]->z) / 2;
                highest = highest ? std::max(current, *highest) : current;
                highest_idx = (highest == current ? i : highest_idx);
            }
        }
        if (found) return highest_idx;
        return std::nullopt;
    }

    /**
     * @brief Decide which frames represent start of attempt, takeoff and culmination above bar.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * @returns true if all moments were found, false otherwise.
     */
    bool find_moments_of_interest(const model_video_points &points) {
        std::optional<std::size_t> val = find_start(points);
        if (val) start = *val;
        else return false;
        
        val = find_takeoff(points);
        if (val) takeoff = *val;
        else return false;
        
        val = find_culmination(points);
        if (val) culmination = *val;
        else return false;

        return true;
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
    void write_parameters() const {
        std::ofstream output;
        std::string o_filename = get_params_filename();
        output.open(o_filename);
        if (!output.is_open()) {
            std::cout << "Output file \"" << o_filename << "\" could not be opened" << std::endl;
        }
        output << std::fixed << std::setprecision(2);
        output << filename << std::endl;
        
        output << "Time";
        // Write names of parameters.
        for (const auto &param : parameters) {
            output << "," << param->name;
        }
        output << std::endl;

        // Write units.
        output << "s";
        for (const auto &param : parameters) {
            output << "," << ((param->unit != "" && param->unit[0] == ' ') ? param->unit.substr(1) : param->unit);
        }

        // Write values of parameters.
        for (std::size_t i = 0; i < points_frame.size(); ++i) {
            output << std::endl << (double)((int)i - (int)takeoff) / fps;
            for (const auto &param : parameters) {
                output << ",";
                if (i < param->size()) {
                    param->write_param(output, i);
                }
            }
        }
        output.close();
    }

};