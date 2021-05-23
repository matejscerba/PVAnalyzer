#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "forward.hpp"
#include "person.hpp"

/**
 * @brief Struct representing detected athlete in during whole vault.
 */
struct model {
public:

    /**
     * @brief Constructor used after detection in video.
     * 
     * @param athlete Athlete whose model should be created.
     * @param video_filename Path to video from which this model is created.
     */
    model(const person &athlete, const std::string &video_filename) {
        this->model_filename = "";
        this->video_filename = video_filename;
        compute_frame_points(athlete.get_points());
        compute_frame_offsets(athlete.get_offsets());
        compute_real_points(athlete.get_direction());
    }

    /**
     * @brief Constructor used for loading model from file.
     * 
     * @param filename Path to file containing saved model.
     */
    model(const std::string &filename) {
        this->model_filename = filename;
        this->video_filename = "";
    }

    /**
     * @brief Save this model to file.
     * 
     * Create file and save this model to it.
     */
    void save() const {
        std::string output_dir = get_output_dir(video_filename);
        std::ofstream output;
        output.open(output_dir + MODEL_FILE);
        if (!output.is_open()) {
            std::cout << "Could not save detected model to file \"" << output_dir << "/model.txt" << "\"." << std::endl;
            return;
        }

        output << video_filename << std::endl;
        for (std::size_t frame_no = 0; frame_no < points_frame_c.size(); ++frame_no) {
            output << frame_no << std::endl << frame_offsets[frame_no] << std::endl;
            for (const auto &p : points_frame_c[frame_no]) {
                output << p << std::endl;
            }
        }
        output.close();
    }

    /**
     * @brief Load model from file.
     * 
     * @param[out] video_fn Path to video from which model was created.
     * @returns true if model was loaded correctly.
     */
    bool load(std::string &video_fn) {
        points_frame_c.clear();
        points_real_c.clear();
        frame_offsets.clear();

        std::ifstream input;
        input.open(model_filename);
        if (!input.is_open()) {
            std::cout << "Model file \"" << model_filename << "\" could not be opened." << std::endl;
            return false;
        }

        std::string line;
        std::getline(input, video_filename);
        video_fn = video_filename;
        for (;;) {
            if (!std::getline(input, line) || line == "") {
                break;
            }
            if (line.find(",") == std::string::npos) {
                // Frame number.
                continue;
            } else {
                frame_offsets.push_back(read_point(std::move(line)));
                points_frame_c.push_back(read_body(input));
            }
        }
        input.close();
        compute_real_points(get_direction());
        return true;
    }

    /**
     * @brief Get points in frame coordinates.
     */
    model_video_points get_frame_points() const {
        return points_frame_c;
    }

    /**
     * @brief Get points in real-life coordinates.
     */
    model_video_points get_real_points() const {
        return points_real_c;
    }

    /**
     * @brief Get direction of athlete's runup.
     * 
     * Compare x coordinate of first and last offset of frame.
     * 
     * @returns direction of camera's movement direction.
     * 
     * @note Camera's movement direction corresponds to athlete's movement direction.
     */
    direction get_direction() const {
        auto first = std::find_if(frame_offsets.begin(), frame_offsets.end(), [](const model_point &p) {
            return p;
        });
        auto last = std::find_if(frame_offsets.rbegin(), frame_offsets.rend(), [](const model_point &p) {
            return p;
        });
        if (first != frame_offsets.end()) {
            if ((*first)->x > (*last)->x) {
                return direction::left;
            } else if ((*first)->x < (*last)->x) {
                return direction::right;
            }
        }
        return direction::unknown;
    }

    /**
     * @brief Draw model's points to given frames.
     * 
     * @param frames Frames in which model should be drawn.
     * @returns frames with drawed model.
     */
    std::vector<cv::Mat> draw(const std::vector<cv::Mat> &frames) const {
        std::vector<cv::Mat> res;
        cv::Mat frame;
        for (std::size_t frame_no = 0; frame_no < frames.size(); ++frame_no) {
            frame = frames[frame_no].clone();
            draw_body(frame, model_to_frame(points_frame_c[frame_no]));
            res.push_back(frame);
        }
        return res;
    }

private:

    /**
     * @brief Path to file containing containing model.
     */
    std::string model_filename;

    /**
     * @brief Path to file containing video from which model was created.
     */
    std::string video_filename;

    /**
     * @brief Athlete's body parts in frame coordinates.
     */
    model_video_points points_frame_c;

    /**
     * @brief Athlete's body parts in real-life coordinates.
     */
    model_video_points points_real_c;

    /**
     * @brief Offsets of top-left corner of each frame in real-life coordinates.
     */
    model_points frame_offsets;

    /**
     * @brief Convert detected points to model's representation.
     * 
     * @param detected_pts Detected athlete's body parts.
     */
    void compute_frame_points(const frame_video_points &detected_pts) {
        points_frame_c.clear();
        for (const auto &pts : detected_pts) {
            model_points b;
            for (const auto &p : pts) {
                if (p) b.push_back(cv::Point3d(p->x, 0, p->y));
                else b.push_back(std::nullopt);
            }
            points_frame_c.push_back(std::move(b));
        }
    }

    /**
     * @brief Convert offsets of frames to model's representation.
     * 
     * @param offsets Offsets of each frame of video.
     */
    void compute_frame_offsets(const frame_points &offsets) {
        frame_offsets.clear();
        for (const auto &o : offsets) {
            if (o) {
                frame_offsets.push_back(cv::Point3d(
                    o->x, 0, o->y
                ));
            } else {
                frame_offsets.push_back(std::nullopt);
            }
        }
    }

    /**
     * @brief Compute athlete's body parts positions in real-life coordinates.
     * 
     * Use frame offsets and athlete's body parts positions in frame coordinates.
     * 
     * @param dir Direction of athlete's runup.
     */
    void compute_real_points(direction dir) {
        points_real_c.clear();
        double mult = 1.0;
        if (dir == direction::left) mult = -1.0;
        for (std::size_t i = 0; i < points_frame_c.size(); ++i) {
            model_points b;
            for (const auto &p : points_frame_c[i]) {
                if (p) {
                    b.push_back(cv::Point3d(
                        mult * (frame_offsets[i]->x + p->x),
                        frame_offsets[i]->y + p->y,
                        - (frame_offsets[i]->z + p->z)
                    ));
                } else {
                    b.push_back(std::nullopt);
                }
            }
            points_real_c.push_back(std::move(b));
        }
    }

    /**
     * @brief Read point from string.
     * 
     * @param s String containing point.
     * @returns point read from given string.
     * 
     * @note Point (x,y,z) is stored as "x,y,z".
     */
    model_point read_point(std::string &&s) const {
        if (s != "" && s[0] != ',') {
            std::replace(s.begin(), s.end(), ',', ' ');
            std::stringstream sstr(s);
            std::string value;
            sstr >> value;
            double x = std::stod(value);
            sstr >> value;
            double y = std::stod(value);
            sstr >> value;
            double z = std::stod(value);
            return cv::Point3d(x, y, z);
        } else {
            return std::nullopt;
        }
    }

    /**
     * @brief Read points representing athlete's body in one frame from input stream.
     * 
     * @param input Input stream holding athlete's body parts positions.
     * @returns athlete's body parts positions in one frame.
     */
    model_points read_body(std::ifstream &input) const {
        std::string line;
        model_points b;
        for (std::size_t i = 0; i < NPOINTS; ++i) {
            std::getline(input, line);
            b.push_back(read_point(std::move(line)));
        }
        return b;
    }

};