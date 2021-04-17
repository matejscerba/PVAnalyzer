#pragma once

#include <opencv2/opencv.hpp>

#include "movement_analyzer.hpp"

class person_checker {
public:

    person_checker(std::size_t frame_no, const cv::Mat &frame, double fps, const cv::Rect &box)
        : move_analyzer(frame_no, frame, box, fps) {
            this->fps = fps;
            first_frame = frame_no;
            tracker = cv::TrackerCSRT::create();
            tracker->init(frame, box);
            bboxes.push_back(box);
    }

    bool track(const cv::Mat &frame, std::size_t frame_no) {
        cv::Rect box;
        // Update tracker.
        if (tracker->update(frame, box)) {
            bboxes.push_back(box);
            return is_inside(get_corners(box), frame)
                && is_moving(frame_no)
                && move_analyzer.update(frame, box, frame_no);
        }

        return false;
    }

    std::size_t get_first_frame() const noexcept {
        return first_frame;
    }

    cv::Rect get_first_bbox() const noexcept {
        return bboxes.front();
    }

    bool vault_began(std::size_t frame_no) const noexcept {
        return move_analyzer.vault_frames(frame_no);
    }

    void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
        if (bboxes.size() > frame_no) {
            cv::Scalar color(0, 0, 255);
            if (move_analyzer.vault_frames(frame_no))
                color = cv::Scalar(0, 255, 0);
            
            cv::rectangle(frame, bboxes[frame_no].tl(), bboxes[frame_no].br(), cv::Scalar(255, 0, 0), 2);
            move_analyzer.draw(frame, frame_no);
        }
    }

private:

    double fps;

    std::size_t first_frame;

    cv::Ptr<cv::Tracker> tracker;

    std::vector<cv::Rect> bboxes;

    movement_analyzer move_analyzer;

    bool is_moving(std::size_t frame_no) const noexcept {
        if ((double)frame_no - (double)first_frame < fps / 3.0) {
            return true;
        } else {
            return move_analyzer.get_direction() != direction::unknown;
        }
    }

};