#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <optional>

#include "person.hpp"

class vault_analyzer {

    /**
     * @brief Body parts of person in each frame transformed into correct coordinates.
     * 
     * @see forward.hpp.
     */
    video_body points;
    
    /// @brief Path to analyzed video.
    std::string filename; 

    /// @brief Returns centers of gravity of person in each frame.
    std::vector<cv::Point2d> get_centers_of_gravity() const {
        std::vector<cv::Point2d> res;
        for (const auto &p : points) {
            // Average of left and right hip (if possible).
            if (p[body_part::r_hip] && p[body_part::l_hip]) {
                res.push_back((*p[body_part::r_hip] + *p[body_part::l_hip]) / 2);
            } else if (p[body_part::r_hip]) {
                res.push_back(*p[body_part::r_hip]);
            } else if (p[body_part::l_hip]) {
                res.push_back(*p[body_part::l_hip]);
            } else {
                res.emplace_back();
            }
        }
        return res;
    }

    /**
     * @brief Write parameters in csv file.
     * 
     * Needs to be improved...
     */
    void write_params() const {
        std::vector<cv::Point2d> cogs = get_centers_of_gravity();
        std::ofstream file;
        file.open("cogs_analyzer_updated_coords.csv");
        file << "Athlete's center of gravity in video " << filename << std::endl;
        for (const auto &cog : cogs) {
            if (cog.y != 0)
                file << -cog.y << std::endl;
            else
                file << std::endl;
        }
        file.close();
    }

public:

    /**
     * @brief Analyze athlete's movements in analyzed video.
     * 
     * @param athlete Athlete to be analyzed.
     * @param filename Path to analyzed video.
     */
    void analyze(const person &athlete, const std::string &filename) {
        points = athlete.get_points();
        this->filename = filename;
        write_params();
    }

};