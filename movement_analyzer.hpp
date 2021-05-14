#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <cstddef>
#include <optional>
#include <vector>

#include "forward.hpp"
#include "background_tracker.hpp"

/**
 * @brief Analyzes movement of person's bounding box.
 */
class movement_analyzer {
public:

    /**
     * @brief Default constructor.
     * 
     * @param frame First frame, where person was detected.
     * @param bbox Bounding box of person in `frame`.
     */
    movement_analyzer(std::size_t frame_no, const cv::Mat &frame, const std::vector<cv::Rect> &bboxes, double fps) noexcept {
        left_direction_tracker = background_tracker(frame, frame_no, bboxes, direction::left);
        right_direction_tracker = background_tracker(frame, frame_no, bboxes, direction::right);
        this->fps = fps;
        dir  = direction::unknown;
    }

    /**
     * @brief Process given frame.
     * 
     * Update valid trackers, update direction and check whether vault began in given frame.
     * 
     * @param frame Frame to be processed.
     * @param bbox Person's bounding box in given frame.
     * @param frame_no Number of processed frame.
     * @returns false if both trackers failed (unable to determine person's movement direction),
     * true if tracking of at least one background tracker is OK.
    */
    bool update(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes, std::size_t frame_no) noexcept {
        bool res = false;
        if (left_direction_tracker) {
            if (left_direction_tracker->update(frame, bboxes)) {
                // Tracker assuming runup to the left didn't fail.
                // if (!takeoff_frame && left_direction_tracker->is_vault_beginning(bbox.height, fps)) {
                //     takeoff_frame = frame_no;
                // }
                res = true;
            } else if (right_direction_tracker) {
                // Tracker failed, invalidate it.
                left_direction_tracker.reset();
            }
        }
        if (right_direction_tracker) {
            if (right_direction_tracker->update(frame, bboxes)) {
                // Tracker assuming runup to the right didn't fail.
                // if (!takeoff_frame && right_direction_tracker->is_vault_beginning(bbox.height, fps)) {
                //     takeoff_frame = frame_no;
                // }
                res = true;
            } else if (left_direction_tracker) {
                // Tracker failed, invalidate it.
                right_direction_tracker.reset();
            }
        }
        update_direction();
        return res;
    }

    /**
     * @brief Get detected direction.
     * 
     * @returns detected direction.
     */
    direction get_direction() const noexcept {
        return dir;
    }

    std::vector<frame_points> get_person_offsets() const noexcept {
        if (left_direction_tracker)
            return left_direction_tracker->get_person_offsets();
        else if (right_direction_tracker)
            return right_direction_tracker->get_person_offsets();
        return {};
    }

    /**
     * @brief Computes how many frames passed since beginning of vault.
     * 
     * @param frame_no Number of frame used for comparison to beginning of vault.
     * @returns number of frames after beginning of vault up to `frame_no`.
     * 
     * @note Returns 0 if vault has not begun yet.
     */
    std::size_t vault_frames(std::size_t frame_no) const noexcept {
        if (takeoff_frame && (frame_no > *takeoff_frame)) {
            return frame_no - *takeoff_frame;
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
     * @note If real is false, frame offset is irrelevant.
     * 
     * @see background_tracker::frame_offset.
     */
    std::optional<cv::Point2d> frame_offset(std::size_t frame_no, bool real = true) const noexcept {
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

    /**
     * @brief Background tracker used for tracking athlete assuming he is moving to the left.
     */
    std::optional<background_tracker> left_direction_tracker;

    /**
     * @brief Background tracker used for tracking athlete assuming he is moving to the right.
     */
    std::optional<background_tracker> right_direction_tracker;

    /**
     * @brief In which frame the vault began (contains value if it was set).
     */
    std::optional<std::size_t> takeoff_frame;

    /**
     * @brief Horizontal direction of person's movement.
     */
    direction dir;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Update person's movement direction based on background trackers.
     * 
     * @note Invalidate the second tracker if person is moving according to the first one.
    */
    void update_direction() noexcept {
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