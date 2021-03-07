#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <vector>
#include <algorithm>
#include <optional>

/**
 * @brief Handles movement of person's bounding box.
 * 
 * Tracks parts of background on both sides of person and determines which direction
 * is that person moving. Determines whether a vault has begun.
 */
class movement_analyzer {

    /// @brief Position of current `left_background` compared to initial `left_background`.
    cv::Point2d left_delta;

    /// @brief Position of current `right_background` compared to initial `right_background`.
    cv::Point2d right_delta;

    /// @brief Vector of positions of person's bounding rectangles compared to initial `left_background`.
    std::vector<cv::Point2d> left_offsets;

    /// @brief Vector of positions of person's bounding rectangles compared to initial `right_background`.
    std::vector<cv::Point2d> right_offsets;

    /// @brief Bounding box of current tracked background to the left of person.
    cv::Rect left_background = cv::Rect();

    /// @brief Bounding box of current tracked background to the right of person.
    cv::Rect right_background = cv::Rect();

    /// @brief Tracker used for tracking background to the left of person. 
    cv::Ptr<cv::Tracker> left_tracker;

    /// @brief Tracker used for tracking background to the right of person. 
    cv::Ptr<cv::Tracker> right_tracker;

    /// @brief In which frame the vault began (contains value if it was set).
    std::optional<std::size_t> _vault_began;

    /// @brief How many frames to check to determine if vault began.
    const std::size_t vault_check_frames = 6;

    /// @brief How much the person's coordinates must change in order to set `_vault_began` to true.
    const double vault_threshold = -0.55 / 720;

    /// @brief Horizontal direction of person's movement.
    int dir = unknown;

    /**
     * @brief Update parts of background to be tracked in order to fit inside frame.
     * 
     * @param frame Frame, inside which the background parts must fit.
     * @param person Person's bounding box, so that left background is initialized to the
     *     left of person, right to the right.
     * 
     * @note Background position is changed only if the old one is outside `frame`,
     *     updated `left_delta`/`right_delta` if new background part should be tracked.
    */
    void update_position_rect(const cv::Mat &frame, cv::Rect2d person) {
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

    /**
     * @brief Update person's movement direction.
     * 
     * If person is moving away from `left_background`, its movement's direction should be left.
     * If person is moving away from `right_background`, its movement's direction should be right.
     * Direction is set only once and if there is enough data to determine its value.
     * 
     * @param person Person, whose direction should be updated.
     * 
     * @note Person must move at least @f$2 * person.width@f$ pixels horizontally to determine its direction.
    */
    void update_direction(const cv::Rect2d &person) {
        if ((dir == unknown) && left_offsets.size() && right_offsets.size()) {
            if (left_offsets.back().x > 2 * person.width)
                dir = right;
            else if (right_offsets.back().x < - 2 * person.width)
                dir = left;
        }
    }

    /**
     * @brief Calculate person's offset of initial background.
     * 
     * Calculate offset of `person` and initial background, which is determined from `delta` and `background`.
     * `delta` determines offset of `background` and initial background.
     * 
     * @param person Bounding box, whose offset should be calculated.
     * @param delta Offset of `background` and initial background (typically `left_delta` or `right_delta`).
     * @param background Bounding box of part background (typically `left_background` or `right_background`).
     * 
     * @note This method returns coordinates of person's center so that initial background's center is at
     *     coordinates (0,0).
    */
    cv::Point2d get_offset(const cv::Rect2d &person, const cv::Point2d &delta, const cv::Rect2d &background) const {
        return delta + get_center(person) - get_center(background);
    }

    /// @brief Calculate center of given rectangle.
    cv::Point2d get_center(const cv::Rect2d &rect) const {
        return cv::Point2d(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    /**
     * @brief Check if vault is beginning.
     * 
     * Takes offsets of person and initial background and computes how much offsets changed
     * in the last `vault_check_frames`. If the change surpasses `vault_threshold`, it is
     * considered that the person took of and `_vault_began` is set to frame_no.
     * 
     * @param height Height of currently processed frame.
     * @param person Bounding box of person.
     * @param frame_no Number of processed frame.
     * 
     * @note `_vault_began` is set only once.
     * 
     * @see vault_check_frames
     * @see vault_threshold
     * @see _vault_began
    */
    void check_vault_beginning(double height, const cv::Rect2d &person, std::size_t frame_no) {
        std::vector<cv::Point2d> offsets;
        if (dir == right) offsets = left_offsets;
        else if (dir == left) offsets = right_offsets;

        if ((!_vault_began) && (offsets.size() > vault_check_frames)) {
            double size = (double)person.height / height;
            double runup_mean_delta = count_mean_delta(offsets.begin(), offsets.end() - vault_check_frames).y;
            double vault_mean_delta = count_mean_delta(offsets.end() - vault_check_frames, offsets.end()).y;

            if ((vault_mean_delta - runup_mean_delta) * size / height < vault_threshold)
                _vault_began = frame_no;
        }
    }

    /**
     * @brief Count mean of offset of given consecutive values in vector.
     * 
     * @param begin Iterator specifying beginning of values to be processed.
     * @param end Iterator specifying end of values to be processed.
     * @returns mean of offsets of given consecutive values.
    */
    cv::Point2d count_mean_delta(std::vector<cv::Point2d>::const_iterator begin, std::vector<cv::Point2d>::const_iterator end) const {
        double n = end - begin - 1;
        if (n) {
            cv::Point2d sum = *(--end) - *begin;
            return sum / n;
        }
        return cv::Point2d();
    }

public:

    /**
     * @brief Supported horizontal movement directions and their corresponding values.
    */
    enum direction : int {
        right = -1,
        unknown = 0,
        left = 1
    };

    /**
     * @brief Process given frame.
     * 
     * Update valid trackers, check direction and whether vault began.
     * 
     * @param frame Frame to be processed.
     * @param person Person's bounding box in given frame.
     * @param frame_no Number of processed frame.
     * @returns false if both trackers failed (unable to determine person's movement direction),
     *     true if tracking of at least one background part is OK.
    */
    bool update(const cv::Mat &frame, const cv::Rect2d &person, std::size_t frame_no) {
        update_position_rect(frame, person);
        bool res = true;
        if (left_tracker->update(frame, left_background)) {
            left_offsets.push_back(get_offset(person, left_delta, left_background));
        } else {
            // Tracker on left side failed - looks like right is valid, person is moving to the left.
            if (dir == unknown) dir = left;
            res = false;
        }
        if (right_tracker->update(frame, right_background)) {
            right_offsets.push_back(get_offset(person, right_delta, right_background));
        } else {
            if (dir == unknown) dir = right;
            if (!res) return false; // Both trackers failed
        }
        update_direction(person);
        check_vault_beginning((double)frame.rows, person, frame_no);
        return true;
    }

    /// @brief Get detected direction.
    int get_direction() const {
        return dir;
    }

    /// @brief Get information whether vault has begun before frame `frame_no`.
    bool vault_began(std::size_t frame_no) const {
        return _vault_began && (frame_no > _vault_began);
    }

    /**
     * @brief Draw bounding boxes of currently tracked background parts.
     * 
     * @param frame Frame in which bounding boxes should be drawn.
    */
    void draw(cv::Mat &frame) const {
        cv::rectangle(frame, left_background.tl(), left_background.br(), cv::Scalar(255, 0, 0), 2);
        cv::rectangle(frame, right_background.tl(), right_background.br(), cv::Scalar(255, 0, 0), 2);
    }

};