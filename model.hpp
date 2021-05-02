#pragma once

#include <opencv2/opencv.hpp>

#include "forward.hpp"
#include "person.hpp"

struct model {
public:

    model(const person &athlete, const std::string &video_filename) noexcept {
        this->model_filename = "";
        this->video_filename = video_filename;
        compute_frame_points(athlete.get_points());
        compute_frame_offsets(athlete.get_offsets());
        compute_real_points();
    }

    model(const std::string &filename) noexcept {
        this->model_filename = filename;
        this->video_filename = "";
    }

    void save() const noexcept {
        std::string output_filename = "models/" + create_output_filename();
        std::string ext = ".txt";
        std::ofstream output;
        output.open(output_filename + ext);
        if (!output.is_open()) {
            std::cout << "Could not save detected model to file \"" << output_filename << ext << "\"." << std::endl;
            return;
        }

        output << video_filename << std::endl;
        for (std::size_t frame_no = 0; frame_no < frame_points.size(); ++frame_no) {
            output << frame_no << std::endl << frame_offsets[frame_no] << std::endl;
            for (const auto &p : frame_points[frame_no]) {
                output << p << std::endl;
            }
        }
        output.close();
    }

    model_video_points get_frame_points() const noexcept {
        return frame_points;
    }

    model_video_points get_real_points() const noexcept {
        return real_points;
    }

    direction get_direction() const noexcept {
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

    bool load(std::string &video_filename) noexcept {
        frame_points.clear();
        real_points.clear();
        frame_offsets.clear();

        std::ifstream input;
        input.open(model_filename);
        if (!input.is_open()) {
            std::cout << "Model file \"" << model_filename << "\" could not be opened." << std::endl;
            return false;
        }

        std::string line;
        std::getline(input, video_filename);
        for (;;) {
            if (!std::getline(input, line) || line == "") {
                break;
            }
            if (line.find(",") == std::string::npos) {
                // Frame number is expected.
            } else {
                frame_offsets.push_back(parse_point(std::move(line)));
                read_body(input);
            }
        }
        input.close();
        compute_real_points();
        return true;
    }

    std::vector<cv::Mat> draw(const std::vector<cv::Mat> &frames) const noexcept {
        std::vector<cv::Mat> res;
        cv::Mat frame;
        for (std::size_t frame_no = 0; frame_no < frames.size(); ++frame_no) {
            frame = frames[frame_no].clone();
            draw_body(frame, model_to_frame(frame_points[frame_no]));
            res.push_back(frame);
        }
        return res;
    }

private:

    std::string model_filename;
    std::string video_filename;

    model_video_points frame_points;
    model_video_points real_points;

    model_offsets frame_offsets;

    void compute_frame_points(const frame_video_points &detected_pts) noexcept {
        frame_points.clear();
        for (const auto &pts : detected_pts) {
            model_points b;
            for (const auto &p : pts) {
                if (p) b.push_back(cv::Point3d(p->x, 0, p->y));
                else b.push_back(std::nullopt);
            }
            frame_points.push_back(std::move(b));
        }
    }

    void compute_frame_offsets(const std::vector<std::optional<cv::Point2d>> &offsets) noexcept {
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

    void compute_real_points() noexcept {
        real_points.clear();
        for (std::size_t i = 0; i < frame_points.size(); ++i) {
            model_points b;
            for (const auto &p : frame_points[i]) {
                if (p) {
                    b.push_back(cv::Point3d(
                        frame_offsets[i]->x + p->x,
                        frame_offsets[i]->y + p->y,
                        - (frame_offsets[i]->z + p->z)
                    ));
                } else {
                    b.push_back(std::nullopt);
                }
            }
            real_points.push_back(std::move(b));
        }
    }

    model_point parse_point(std::string &&s) const noexcept {
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

    void read_body(std::ifstream &input) noexcept {
        std::string line;
        model_points b;
        for (std::size_t i = 0; i < NPOINTS; ++i) {
            std::getline(input, line);
            b.push_back(parse_point(std::move(line)));
        }
        frame_points.push_back(b);
    }

};