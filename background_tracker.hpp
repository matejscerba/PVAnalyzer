#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <optional>

/**
 * @brief Handles background tracking used to determine person's movements.

 * Tracks part of background on one side of person and determines if person is
 * moving away from tracked part of background. Determines whether a vault has begun.
 */
class background_tracker {

    /// @brief Tracker used to track background.
    cv::Ptr<cv::Tracker> tracker;

    ///@brief Vector of bounding boxes of tracked background.
    std::vector<cv::Rect> backgrounds;

    /// @brief Vector of offsets of background and initial person's position.
    std::vector<cv::Point2d> background_offsets;

    /// @brief Vector of offsets of person and its initial position.
    std::vector<cv::Point2d> person_offsets;

    /// @brief Vector of offsets of frames' origins and person's initial position.
    std::vector<cv::Point2d> frame_offsets;

    /**
     * @brief Assumed direction, which person is moving.
     * 
     * @see movement_analyzer::direction.
     */
    int direction;

    /// @see movement_analyzer::vault_check_frames.
    std::size_t vault_check_frames;

    /// @see movement_analyzer::vault_threshold.
    double vault_threshold;

    /// @brief Horizontal distance, that person must move in assumed direction.
    double valid_direction_threshold;

    /**
     * @brief Compute center of given rectangle.
     * 
     * @param rect Rectangle whose center should be computed.
     * @returns center of `rect`.
     */
    cv::Point2d get_center(const cv::Rect &rect) const {
        return cv::Point2d((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
    }

    /**
     * @brief Update background part to be tracked.
     * 
     * Change background part to be tracked if no part was tracked before or
     * of last part is out of bounds of frame.
     * 
     * @param frame Frame in which to update background part.
     * @param background Last background part that was tracked.
     * @param person Bounding box of person in `frame`.
     * @returns updated background bounding box.
     */
    cv::Rect update_rect(const cv::Mat &frame, const cv::Rect &background, const cv::Rect &person) {
        cv::Rect bg = background;
        if (!backgrounds.size() || bg.x <= 0 || bg.y <= 0 ||
            bg.x + bg.width >= (double)frame.cols ||
            bg.y + bg.height >= (double)frame.rows) {
                // Compute horizontal shift for new background.
                int shift = direction * person.width;
                int x = std::max(0, person.x + shift);
                int width = std::min(person.x, person.width);
                width = std::min(width, frame.cols - (person.x + shift));
                // Create new background inside frame.
                bg = cv::Rect(
                    x, person.y,
                    width, person.height
                );
                // Initialize tracker with new background.
                tracker->init(frame, bg);
        }
        return bg;
    }

    /**
     * @brief Update all stored elements.
     * 
     * Compute all properties to be tracked for each frame for given frame.
     * 
     * @param frame Frame in which properties should be computed.
     * @param background Last tracked background's bounding box.
     * @param person Person's bounding box in `frame`.
     * 
     * @note All offsets are computed against person's initial position.
     */
    void update_elements(const cv::Mat &frame, const cv::Rect &background, const cv::Rect &person) {
        cv::Rect new_background = update_rect(frame, background, person);
        backgrounds.push_back(new_background);
        if (background_offsets.size()) {
            // Compare last background's bounding box and current background's bounding box.
            // If background was not changed in `update_rect()`, background's offset remains the same.
            cv::Point2d last_offset = background_offsets.back();
            cv::Point2d offset_change = get_center(new_background) - get_center(background);
            background_offsets.push_back(last_offset + offset_change);
        } else {
            background_offsets.push_back(get_center(new_background) - get_center(person));
        }
        person_offsets.push_back(get_center(person) - get_center(new_background) + background_offsets.back());
        frame_offsets.push_back(person_offsets.back() - get_center(person));
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
     * @brief Default contructor.
     * 
     * @param frame First frame in which person was detected.
     * @param person Bounding box of person in `frame`.
     * @param dir Assumed direction of person's movement.
     * @param check_frames Number of frames to check for vault's beginning.
     * @param threshold Threshold to determine beginning of vault.
     */
    background_tracker( const cv::Mat &frame,
                        const cv::Rect &person,
                        int dir,
                        std::size_t check_frames,
                        double threshold) {
        direction = dir;
        vault_check_frames = check_frames;
        vault_threshold = threshold;
        valid_direction_threshold = person.width;
        tracker = cv::TrackerCSRT::create();
        update(frame, person);
    }

    /**
     * @brief Track background in given frame and update properties.
     * 
     * @param frame Frame in which to track background.
     * @param person Person's bounding box in `frame`.
     */
    bool update(const cv::Mat &frame, const cv::Rect &person) {
        cv::Rect background_rect;
        if (!backgrounds.size()) {
            tracker->init(frame, update_rect(frame, background_rect, person));
        }
        if (tracker->update(frame, background_rect)) {
            update_elements(frame, background_rect, person);
            return true;
        }
        return false;
    }

    /**
     * @brief Check if assumed direction is correct.
     * 
     * @returns true if assumed direction is correct, false otherwise.
     */
    bool is_valid_direction() const {
        return person_offsets.size() && (- direction * person_offsets.back().x > valid_direction_threshold);
    }

    /**
     * @brief Check if vault is beginning.
     * 
     * @param frame_height Height of frame.
     * @param person_height Height of person in frame.
     * @returns true if vault is beginning, false otherwise.
     */
    bool is_vault_beginning(double frame_height, double person_height) const {
        if (person_offsets.size() > vault_check_frames) {
            double size = person_height / frame_height;
            double runup_mean_delta = count_mean_delta(person_offsets.begin(), person_offsets.end() - vault_check_frames).y;
            double vault_mean_delta = count_mean_delta(person_offsets.end() - vault_check_frames, person_offsets.end()).y;

            if ((vault_mean_delta - runup_mean_delta) * size / frame_height < vault_threshold)
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
     * @note Frame number 0 corresponds to first frame, in which person was detected.
     * 
     * @see movement_analyzer::frame_offset.
     */
    std::optional<cv::Point2d> frame_offset(std::size_t frame_no) const {
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
    void draw(cv::Mat &frame, std::size_t person_frame_no) const {
        cv::rectangle(frame, backgrounds[person_frame_no].tl(), backgrounds[person_frame_no].br(), cv::Scalar(255, 0, 0), 2);
    }

};