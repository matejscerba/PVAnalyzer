#pragma once

#include <opencv2/opencv.hpp>

#include "forward.hpp"
#include "person.hpp"

struct model {
public:

    model(const person &athlete, const std::string &video_filename) noexcept {
        this->video_filename = video_filename;
        compute_frame_points(athlete.get_points());
        compute_frame_offsets(athlete.get_offsets());
        compute_real_points();
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
        for (std::size_t frame_no; frame_no < frame_points.size(); ++frame_no) {
            output << frame_no << std::endl << frame_offsets[frame_no] << std::endl;
            for (const auto &p : frame_points[frame_no]) {
                output << p << std::endl;
            }
        }
        output.close();
    }

    video_body get_frame_points() const noexcept {
        return video_body();
    }

    direction get_direction() const noexcept {
        return direction::left;
    }

private:

    std::string video_filename;

    model_video_body frame_points;
    model_video_body real_points;

    model_offsets frame_offsets;

    void compute_frame_points(const video_body &detected_pts) noexcept {
        frame_points.clear();
        for (const auto &pts : detected_pts) {
            model_body b;
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
            model_body b;
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

};