#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

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
     * @param frame_no Number of first frame, where person was detected.
     * @param frame First frame, where person was detected.
     * @param bboxes Bounding boxes of person in `frame`.
     * @param fps Frame rate of processed video.
     */
    movement_analyzer(std::size_t frame_no, const cv::Mat &frame, const std::vector<cv::Rect> &bboxes, double fps) :
        bg_tracker(frame, frame_no, bboxes) {
            this->fps = fps;
            dir  = direction::unknown;
            direction_threshold = rect(bboxes).width;
    }

    /**
     * @brief Process given frame.
     * 
     * Update background tracker and update direction.
     * 
     * @param frame Frame to be processed.
     * @param bboxes Person's bounding boxes in given frame.
     * @param frame_no Number of processed frame.
     * @returns false if both trackers failed (unable to determine person's movement direction),
     * true if tracking of at least one background tracker is OK.
    */
    bool update(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes, std::size_t frame_no) {
        bool res = bg_tracker.update(frame, bboxes);
        update_direction();
        return res;
    }

    /**
     * @brief Get detected direction.
     * 
     * @returns detected direction.
     */
    direction get_direction() const {
        return dir;
    }

    /**
     * @brief Get person's offsets.
     * 
     * @returns person's offsets.
     */
    std::vector<frame_points> get_person_offsets() const {
        return bg_tracker.get_person_offsets();
    }

    /**
     * @brief Get offset of frame specified by its number.
     * 
     * @param frame_no Number of frame which offset should be computed.
     * @param real Returns valid offset if true, zero offset otherwise.
     * @returns offset of specified frame, invalid value if offset could not be computed.
     * 
     * @note If real is false, frame offset is irrelevant, only frame coordinates are considered.
     * 
     * @see background_tracker::frame_offset.
     */
    std::optional<cv::Point2d> get_frame_offset(std::size_t frame_no, bool real = true) const {
        if (!real) return cv::Point2d();
        return bg_tracker.get_frame_offset(frame_no);
    }

    /**
     * @brief Draw bounding boxes of currently tracked background part.
     * 
     * @param frame Frame in which bounding box should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        bg_tracker.draw(frame, frame_no);
    }

private:

    /**
     * @brief Tracker used for tracking background and handling frames' movements.
     */
    background_tracker bg_tracker;

    /**
     * @brief Horizontal direction of person's movement.
     */
    direction dir;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Horizontal distance, that person must move in order to get his runup direction.
     */
    double direction_threshold;

    /**
     * @brief Update person's movement direction based on background tracker.
     * 
     * @note Direction is updated only once.
    */
    void update_direction() {
        if (dir == direction::unknown) {
            frame_point offset = bg_tracker.get_person_offset();
            if (offset) {
                if (offset->x > direction_threshold) {
                    dir = direction::right;
                }
                if (offset->x < - direction_threshold) {
                    dir = direction::left;
                }
            }
        }
    }

};