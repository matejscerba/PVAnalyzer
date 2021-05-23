#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include <iostream>
#include <fstream>
#include <ostream>

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
     */
    background_tracker(const cv::Mat &frame, std::size_t frame_no, const std::vector<cv::Rect> &bboxes) {
        backgrounds = std::vector<std::optional<cv::Rect>>(frame_no, std::nullopt);
        background_offsets = frame_points(frame_no, std::nullopt);
        person_offsets = std::vector<frame_points>(bboxes.size(), frame_points(frame_no, std::nullopt));
        frame_offsets = frame_points(frame_no, std::nullopt);
        tracker = cv::TrackerCSRT::create();
        update_rect(frame, cv::Rect(), bboxes);
    }

    /**
     * @brief Track background in given frame and update properties.
     * 
     * @param frame Frame in which to track background.
     * @param bbox Person's bounding box in `frame`.
     */
    bool update(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes) {
        cv::Rect background_rect;
        if (tracker->update(frame, background_rect)) {
            update_properties(frame, background_rect, bboxes);
            return true;
        }
        return false;
    }

    /**
     * @brief Get last offset of person.
     * 
     * @returns maximal offset of all bounding boxes.
     */
    frame_point get_person_offset() const {
        frame_point res = std::nullopt;
        double max = 0.0;
        for (std::size_t i = 0; i < person_offsets.size(); ++i)
            if (person_offsets[i].size() && person_offsets[i].back())
                if (cv::norm(*person_offsets[i].back()) > max) {
                    res = person_offsets[i].back();
                    max = cv::norm(*person_offsets[i].back());
                }
        return res;
    }


    /**
     * @brief Get offsets of all person's bounding boxes.
     * 
     * @returns offsets of all person's bounding boxes.
     */
    std::vector<frame_points> get_person_offsets() const {
        return person_offsets;
    }

    /**
     * @brief Get offset of frame specified by its number.
     * 
     * @param frame_no Number of frame which offset should be computed.
     * @returns offset of specified frame, invalid value if offset could not be computed.
     * 
     * @see movement_analyzer::frame_offset.
     */
    std::optional<cv::Point2d> get_frame_offset(std::size_t frame_no) const {
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
    void draw(cv::Mat &frame, std::size_t frame_no) const {
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
    frame_points background_offsets;

    /**
     * @brief Vector of vector of offsets of person and its initial position.
     * 
     * Offset of person's bounding box center and initial person's bounding box center.
     * 
     * @note First dimension specifies number of bounding box, second number of frame.
     */
    std::vector<frame_points> person_offsets;

    /**
     * @brief Vector of offsets of frames' origins and person's initial position.
     * 
     * Offset of top left corner of frame and initial person's bounding box center.
     */
    frame_points frame_offsets;

    /**
     * @brief Update background part to be tracked.
     * 
     * Change background part to be tracked if no part was tracked before or
     * of last part is out of bounds of frame.
     * 
     * @param frame Frame in which to update background part.
     * @param background Last background part that was tracked.
     * @param bboxes Bounding boxes of person in `frame`.
     * @returns updated background's bounding box.
     */
    cv::Rect update_rect(const cv::Mat &frame, const cv::Rect &background, const std::vector<cv::Rect> &bboxes) {
        cv::Rect bg = background;
        if (!backgrounds.size() || !backgrounds.back() || bg.x <= 0 || bg.y <= 0 ||
            bg.x + bg.width >= (double)frame.cols || area(bg & rect(bboxes)) > 0 ||
            bg.y + bg.height >= (double)frame.rows || !bg.width || !bg.height) {
                cv::Rect bbox = rect(bboxes);
                cv::Rect left(0, frame.rows / 4, frame.cols / 4, frame.rows / 2);
                cv::Rect right(3 * frame.cols / 4, frame.rows / 4, frame.cols / 4, frame.rows / 2);
                bg = area(left & bbox) <= area(right & bbox) ? left : right;
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
     * @param bboxes Person's bounding boxes in `frame`.
     * 
     * @note All offsets are computed against person's initial position.
     */
    void update_properties(const cv::Mat &frame, const cv::Rect &background, const std::vector<cv::Rect> &bboxes) {
        cv::Rect new_background = update_rect(frame, background, bboxes);
        backgrounds.push_back(new_background);
        if (background_offsets.size() && background_offsets.back()) {
            // Compare last background's bounding box and current background's bounding box.
            // If background was not changed in `update_rect()`, background's offset remains the same.
            cv::Point2d last_offset = *background_offsets.back();
            cv::Point2d offset_change = get_center(new_background) - get_center(background);
            background_offsets.push_back(last_offset + offset_change);
        } else {
            background_offsets.push_back(get_center(new_background) - get_center(rect(bboxes)));
        }
        for (std::size_t i = 0; i < bboxes.size(); ++i) {
            person_offsets[i].push_back(get_center(bboxes[i]) - get_center(new_background) + background_offsets.back());
        }
        frame_offsets.push_back(*background_offsets.back() - get_center(*backgrounds.back()));
    }

};