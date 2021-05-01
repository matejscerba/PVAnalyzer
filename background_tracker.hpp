#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include "forward.hpp"

/**
 * @brief Handles background tracking used to determine person's movements.

 * Tracks part of background on one side of person and determines if person is
 * moving away from tracked part of background. Determines whether a vault has begun.
 */
class background_tracker {
public:

    /**
     * @brief Default contructor.
     * 
     * @param frame Frame in which person was detected.
     * @param frame_no Number of frame in which person was detected.
     * @param bbox Bounding box of person in `frame`.
     * @param dir Assumed direction of person's movement.
     */
    background_tracker(const cv::Mat &frame, std::size_t frame_no, const cv::Rect &bbox, direction dir) noexcept {
        backgrounds = std::vector<std::optional<cv::Rect>>(frame_no, std::nullopt);
        background_offsets = offsets(frame_no, std::nullopt);
        person_offsets = offsets(frame_no, std::nullopt);
        frame_offsets = offsets(frame_no, std::nullopt);
        this->dir = dir;
        valid_direction_threshold = bbox.width;
        tracker = cv::TrackerCSRT::create();
        update(frame, bbox);
    }

    /**
     * @brief Track background in given frame and update properties.
     * 
     * @param frame Frame in which to track background.
     * @param bbox Person's bounding box in `frame`.
     */
    bool update(const cv::Mat &frame, const cv::Rect &bbox) noexcept {
        cv::Rect background_rect;
        if (!backgrounds.size() || !backgrounds.back()) {
            // Tracking was not performed yet, tracker needs to be initialized.
            tracker->init(frame, update_rect(frame, background_rect, bbox));
        }
        if (tracker->update(frame, background_rect)) {
            update_properties(frame, background_rect, bbox);
            return true;
        }
        return false;
    }

    /**
     * @brief Check if assumed direction is correct.
     * 
     * @returns true if assumed direction is correct, false otherwise.
     */
    bool is_valid_direction() const noexcept {
        double d = 0;
        if (dir == direction::left)
            d = -1;
        else if (dir == direction::right)
            d = 1;
        return person_offsets.size() && person_offsets.back() && (d * person_offsets.back()->x > valid_direction_threshold);
    }

    /**
     * @brief Check if vault is beginning.
     * 
     * Compute difference of vertical movement changes of last `vault_check_frames`
     * and `vault_check_frames` before that and decide whether tracked person takes off.
     * 
     * @param person_height Height of person in frame.
     * @param fps Frame rate of processed video.
     * @returns true if vault is beginning, false otherwise.
     */
    bool is_vault_beginning(double person_height, double fps) const noexcept {
        if (!is_valid_direction())
            return false;
        int vault_check_frames = (int)(VAULT_CHECK_TIME * fps);
        if (person_offsets.size() > 2 * vault_check_frames) {
            std::optional<cv::Point2d> runup_mean_delta = count_mean_delta(person_offsets.end() - 2 * vault_check_frames, person_offsets.end() - vault_check_frames);
            std::optional<cv::Point2d> vault_mean_delta = count_mean_delta(person_offsets.end() - vault_check_frames, person_offsets.end());

            if (runup_mean_delta && vault_mean_delta && (vault_mean_delta->y - runup_mean_delta->y) * person_height < VAULT_THRESHOLD)
                return true;
        }
        return false;
    }

    /**
     * @brief Get offset of frame specified by its number.
     * 
     * @param frame_no Number of frame which offset should be computed.
     * @returns offset of specified frame, invalid value if offset could not be computed.
     * 
     * @see movement_analyzer::frame_offset.
     */
    std::optional<cv::Point2d> frame_offset(std::size_t frame_no) const noexcept {
        if (frame_no < frame_offsets.size())
            return frame_offsets[frame_no];
        return std::nullopt;
    }

    /**
     * @brief Draw bounding box of currently tracked background part.
     * 
     * @param frame Frame in which bounding box should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
        if (frame_no < backgrounds.size() && backgrounds[frame_no])
            cv::rectangle(frame, backgrounds[frame_no]->tl(), backgrounds[frame_no]->br(), cv::Scalar(255, 0, 0), 2);
    }

private:

    /**
     * @brief Tracker used to track background.
     */
    cv::Ptr<cv::Tracker> tracker;

    /**
     * @brief Vector of bounding boxes of tracked background.
     */
    std::vector<std::optional<cv::Rect>> backgrounds;

    /**
     * @brief Vector of offsets of background and initial person's position.
     * 
     * Offset of background's center and initial person's bounding box center.
     */
    offsets background_offsets;

    /**
     * @brief Vector of offsets of person and its initial position.
     * 
     * Offset of person's bounding box center and initial person's bounding box center.
     */
    offsets person_offsets;

    /**
     * @brief Vector of offsets of frames' origins and person's initial position.
     * 
     * Offset of top left corner of frame and initial person's bounding box center.
     */
    offsets frame_offsets;

    /**
     * @brief Assumed direction, which person is moving.
     * 
     * @see movement_analyzer::direction.
     */
    direction dir;

    /**
     * @brief Horizontal distance, that person must move in assumed direction in order to mark that direction as valid.
     */
    double valid_direction_threshold;

    /**
     * @brief Update background part to be tracked.
     * 
     * Change background part to be tracked if no part was tracked before or
     * of last part is out of bounds of frame.
     * 
     * @param frame Frame in which to update background part.
     * @param background Last background part that was tracked.
     * @param bbox Bounding box of person in `frame`.
     * @returns updated background bounding box.
     */
    cv::Rect update_rect(const cv::Mat &frame, const cv::Rect &background, const cv::Rect &bbox) noexcept {
        // TODO: Handle if bg rect is too small.
        cv::Rect bg = background;
        if (!backgrounds.size() || !backgrounds.back() || bg.x <= 0 || bg.y <= 0 ||
            bg.x + bg.width >= (double)frame.cols ||
            bg.y + bg.height >= (double)frame.rows) {
                // Compute horizontal shift for new background.
                int d = 0;
                if (dir == direction::left)
                    d = 1;
                else if (dir == direction::right)
                    d = -1;
                int shift = d * bbox.width;
                int x = std::max(0, bbox.x + shift);
                int width = std::min(bbox.x, bbox.width);
                width = std::min(width, frame.cols - (bbox.x + shift));
                width = std::max(10, width);
                // Create new background inside frame.
                bg = cv::Rect(
                    x, bbox.y,
                    width, bbox.height
                );
                // Initialize tracker with new background.
                tracker->init(frame, bg);
        }
        return bg;
    }

    /**
     * @brief Update all stored properties.
     * 
     * Compute all properties to be tracked for given frame.
     * 
     * @param frame Frame in which properties should be computed.
     * @param background Last tracked background's bounding box.
     * @param bbox Person's bounding box in `frame`.
     * 
     * @note All offsets are computed against person's initial position.
     */
    void update_properties(const cv::Mat &frame, const cv::Rect &background, const cv::Rect &bbox) noexcept {
        cv::Rect new_background = update_rect(frame, background, bbox);
        backgrounds.push_back(new_background);
        if (background_offsets.size() && background_offsets.back()) {
            // Compare last background's bounding box and current background's bounding box.
            // If background was not changed in `update_rect()`, background's offset remains the same.
            cv::Point2d last_offset = *background_offsets.back();
            cv::Point2d offset_change = get_center(new_background) - get_center(background);
            background_offsets.push_back(last_offset + offset_change);
        } else {
            background_offsets.push_back(get_center(new_background) - get_center(bbox));
        }
        person_offsets.push_back(get_center(bbox) - get_center(new_background) + background_offsets.back());
        frame_offsets.push_back(person_offsets.back() - get_center(bbox));
    }

};