#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <vector>
#include <algorithm>
#include <optional>

#include "background_tracker.hpp"

/**
 * @brief Handles movement of person's bounding box.
 */
class movement_analyzer {
public:

    /**
     * @brief Default constructor.
     * 
     * @param frame First frame, where person was detected.
     * @param person Bounding box of person in `frame`.
     */
    movement_analyzer(std::size_t frame_no, const cv::Mat &frame, const cv::Rect2d &person, double fps) {
        left_direction_tracker = background_tracker(frame, frame_no, person, direction::left);
        right_direction_tracker = background_tracker(frame, frame_no, person, direction::right);
        this->fps = fps;
    }

    /**
     * @brief Process given frame.
     * 
     * Update valid trackers, update direction and check whether vault began in given frame.
     * 
     * @param frame Frame to be processed.
     * @param person Person's bounding box in given frame.
     * @param frame_no Number of processed frame.
     * @returns false if both trackers failed (unable to determine person's movement direction),
     *     true if tracking of at least one background tracker is OK.
    */
    bool update(const cv::Mat &frame, const cv::Rect2d &person, std::size_t frame_no) {
        bool res = false;
        if (left_direction_tracker) {
            if (left_direction_tracker->update(frame, person)) {
                if (!_vault_began && left_direction_tracker->is_vault_beginning(person.height, fps)) {
                    _vault_began = frame_no;
                }
                res = true;
            } else {
                if (right_direction_tracker)
                    left_direction_tracker.reset();
            }
        }
        if (right_direction_tracker) {
            if (right_direction_tracker->update(frame, person)) {
                if (!_vault_began && right_direction_tracker->is_vault_beginning(person.height, fps)) {
                    _vault_began = frame_no;
                }
                res = true;
            } else {
                if (left_direction_tracker)
                    right_direction_tracker.reset();
            }
        }
        update_direction(person);
        return res;
    }

    /// @brief Get detected direction.
    direction get_direction() const noexcept {
        return dir;
    }

    cv::Point2d last_movement() const noexcept {
        if (dir == direction::unknown) return cv::Point2d();
        if (left_direction_tracker)
            return left_direction_tracker->last_person_movement();
        if (right_direction_tracker)
            return right_direction_tracker->last_person_movement();
        return cv::Point2d();
    }

    /**
     * @brief Computes how many frames passed from beginning of vault.
     * 
     * @param frame_no Number of frame used for comparison to beginning of vault.
     * @returns number of frames after beginning of vault up to `frame_no`.
     * 
     * @note Returns 0 if vault has not begun yet.
     */
    std::size_t vault_frames(std::size_t frame_no) const {
        if (_vault_began && (frame_no > _vault_began.value())) {
            return frame_no - _vault_began.value();
        } else {
            return 0;
        }
    }

    /**
     * @brief Get offset of frame specified by its number.
     * 
     * @param frame_no Number of frame which offset should be computed.
     * @param real Returns valid offset if true, zero offset otherwise.
     * @returns offset of specified frame, invalid value if offset could not be computed.
     * 
     * @note Frame number 0 corresponds to first frame, in which person was detected.
     * @note If real is false, frame offset is irrelevant.
     * 
     * @see background_tracker::frame_offset.
     */
    std::optional<cv::Point2d> frame_offset(std::size_t frame_no, bool real = true) const {
        if (!real) return cv::Point2d();
        if (left_direction_tracker)
            return left_direction_tracker->frame_offset(frame_no);
        if (right_direction_tracker)
            return right_direction_tracker->frame_offset(frame_no);
        return std::nullopt;
    }

    /**
     * @brief Draw bounding boxes of currently tracked background parts.
     * 
     * @param frame Frame in which bounding boxes should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        if (left_direction_tracker)
            left_direction_tracker->draw(frame, frame_no);
        if (right_direction_tracker)
            right_direction_tracker->draw(frame, frame_no);
    }

private:

    /// @brief Background tracker used for tracking athlete assuming he is moving to the left.
    std::optional<background_tracker> left_direction_tracker;

    /// @brief Background tracker used for tracking athlete assuming he is moving to the right.
    std::optional<background_tracker> right_direction_tracker;

    /// @brief In which frame the vault began (contains value if it was set).
    std::optional<std::size_t> _vault_began;

    // TODO: rename to direction
    /// @brief Horizontal direction of person's movement.
    direction dir = direction::unknown;

    double fps;

    /**
     * @brief Update person's movement direction based on background trackers.
     * 
     * @param person Person, whose direction should be updated.
     * 
     * @note Invalidate the second tracker if person is moving according to the first one.
    */
    void update_direction(const cv::Rect2d &person) {
        if (dir == direction::unknown) {
            if (left_direction_tracker) {
                if (left_direction_tracker->is_valid_direction()) {
                    dir = direction::left;
                    right_direction_tracker.reset();
                }
            }
            if (right_direction_tracker) {
                if (right_direction_tracker->is_valid_direction()) {
                    dir = direction::right;
                    left_direction_tracker.reset();
                }
            }
        }
    }

};