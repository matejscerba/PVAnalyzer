#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <vector>
#include <algorithm>

class movement_analyzer {

    cv::Point2d left_delta;
    cv::Point2d right_delta;
    std::vector<cv::Point2d> left_offsets;
    std::vector<cv::Point2d> right_offsets;
    cv::Rect2d left_background;
    cv::Rect2d right_background;
    cv::Ptr<cv::Tracker> left_tracker;
    cv::Ptr<cv::Tracker> right_tracker;

    bool _vault_began = false;
    const std::size_t vault_check_frames = 6;
    const double vault_threshold = -0.55 / 720;

    int dir = unknown;

    // Check if `background` is inside frame, create new one if neccessary.
    void update_position_rect(cv::Mat &frame, cv::Rect2d person) {
        if ((left_background.x <= 0) || (left_background.y <= 0) ||
            (left_background.x + left_background.width >= (double)frame.cols) ||
            (left_background.y + left_background.height >= (double)frame.rows)) {
                double left_x = person.x - person.width;
                // Make sure `background` fits inside frame.
                left_background = cv::Rect2d(
                    std::max(left_x, 0.0), person.y,
                    std::min(person.width, (double)frame.cols - left_x), person.height
                );
                
                left_tracker = cv::TrackerCSRT::create();
                left_tracker->init(frame, left_background);

                if (left_offsets.size())
                    left_delta = left_offsets.back();
        }
        if ((right_background.x <= 0) || (right_background.y <= 0) ||
            (right_background.x + right_background.width >= (double)frame.cols) ||
            (right_background.y + right_background.height >= (double)frame.rows)) {
                double right_x = person.x + person.width;
                // Make sure `right_background` fits inside frame.
                right_background = cv::Rect2d(
                    std::max(right_x, 0.0), person.y,
                    std::min(person.width, (double)frame.cols - right_x), person.height
                );
                
                right_tracker = cv::TrackerCSRT::create();
                right_tracker->init(frame, right_background);

                if (right_offsets.size())
                    right_delta = right_offsets.back();
        }
    }

    // Checks if athlete is running away from left or right part of background.
    void check_direction(cv::Rect2d person) {
        if ((dir == unknown) && left_offsets.size() && right_offsets.size()) {
            if (left_offsets.back().x > 2 * person.width)
                dir = right;
            else if (right_offsets.back().x < - 2 * person.width)
                dir = left;
        }
    }

    // Calculate offset of initial `background`.
    cv::Point2d get_offset(cv::Rect2d person, cv::Point2d delta, cv::Rect2d background) const {
        return delta + get_center(person) - get_center(background);
    }

    // Calculate center of given rectangle.
    cv::Point2d get_center(cv::Rect2d rect) const {
        return cv::Point2d(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    // Check if vault is beginning.
    void check_vault_beginning(double height, cv::Rect2d person) {
        std::vector<cv::Point2d> offsets;
        if (dir == right) offsets = left_offsets;
        else if (dir == left) offsets = right_offsets;

        if ((!_vault_began) && (offsets.size() > vault_check_frames)) {
            double size = (double)person.height / height;
            double runup_mean_delta = count_mean_delta(offsets.begin(), offsets.end() - vault_check_frames).y;
            double vault_mean_delta = count_mean_delta(offsets.end() - vault_check_frames, offsets.end()).y;

            if ((vault_mean_delta - runup_mean_delta) * size / height < vault_threshold)
                _vault_began = true;
        }
    }

    // Count mean of difference of consecutives values given by iterators.
    cv::Point2d count_mean_delta(std::vector<cv::Point2d>::const_iterator begin, std::vector<cv::Point2d>::const_iterator end) const {
        double n = end - begin - 1;
        if (n) {
            cv::Point2d sum = *(--end) - *begin;
            return sum / n;
        }
        return cv::Point2d();
    }

public:

    enum direction : int { right = -1, unknown = 0, left = 1 };

    bool update(cv::Mat &frame, cv::Rect2d person) {
        update_position_rect(frame, person);
        check_direction(person);
        bool res = true;
        if (left_tracker->update(frame, left_background)) {
            left_offsets.push_back(get_offset(person, left_delta, left_background));
        } else {
            // Tracker on left side failed - looks like right is valid.
            if (dir == unknown) dir = left;
            res = false;
        }
        if (right_tracker->update(frame, right_background)) {
            right_offsets.push_back(get_offset(person, right_delta, right_background));
        } else {
            if (dir == unknown) dir = right;
            if (!res) return false; // Both trackers failed
        }
        check_vault_beginning((double)frame.rows, person);
        return true;
    }

    int get_direction() const {
        return dir;
    }

    bool vault_began() const {
        return _vault_began;
    }

    void draw(cv::Mat &frame) const {
        cv::rectangle(frame, left_background.tl(), left_background.br(), cv::Scalar(255, 0, 0), 2);
        cv::rectangle(frame, right_background.tl(), right_background.br(), cv::Scalar(255, 0, 0), 2);
    }

};