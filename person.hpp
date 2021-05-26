#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "forward.hpp"
#include "movement_analyzer.hpp"
#include "parts_detector.hpp"
#include "background_tracker.hpp"

/**
 * @brief Represents athlete through whole video.
 * 
 * Holds information about a athlete through whole video.
 * Tracks athlete through whole video.
 */
class person {
public:

    /**
     * @brief Default constructor.
     * 
     * @param frame_no Number of frame, in which athlete was detected for the first time.
     * @param box Bounding box of athlete in `frame`.
     * @param fps Frame rate of processed video.
     */
    person(std::size_t frame_no, const cv::Rect &box, double fps) {
        move_analyzer.reset();
        this->first_frame = frame_no;
        this->first_bbox = box;
        this->fps = fps;
        this->end = false;
        parts_dtor = parts_detector();
    }

    /**
     * @brief Detect athlete and his body parts in given video.
     * 
     * @param raw_frames Frames of video to be processed.
     * @returns Frames with drawings of detections.
     */
    std::vector<cv::Mat> detect(const std::vector<cv::Mat> &raw_frames) {
        std::vector<cv::Mat> frames;

        cv::Mat frame;
        bboxes = std::vector<std::vector<cv::Rect>>(first_frame, std::vector<cv::Rect>(GRID_SIZE * GRID_SIZE, cv::Rect()));
        points = frame_video_points(raw_frames.size(), frame_points(NPOINTS, std::nullopt));
        cv::Ptr<cv::Tracker> my_tracker = cv::TrackerCSRT::create();
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Processing frame " << frame_no + 1 << "/" << raw_frames.size() << std::endl;

            frame = raw_frames[frame_no].clone();

            // Detect athlete's body in current frame.
            if (!end && track(frame, frame_no)) {
                draw(frame, frame_no);
            }
            // if (frame_no >= first_frame) {
            //     if (frame_no == first_frame)
            //         my_tracker->init(frame, first_bbox);
            //         cv::Rect bbox;
            //     if (my_tracker->update(frame, bbox)) {
            //         cv::rectangle(frame, bbox.tl(), bbox.br(), cv::Scalar(255, 0, 0), 2);
            //     }
            // }
            // cv::imshow("frame", frame);
            // cv::waitKey(1);
            // if (cv::waitKey() == 27)
            //     cv::imwrite("img.png", frame);

            // Save current modified frame.
            frames.push_back(frame);
        }
        return frames;
    }

    /**
     * @brief Get all body parts of athlete throughout whole video.
     * 
     * @param real Transforms detected points into real life coordinates if true.
     * @returns detected body parts in whole video in specified coordinates.
     * 
     * @note If `real` is false, return points in frame coordinates.
     */
    frame_video_points get_points(bool real = false) const {
        frame_video_points res;
        for (std::size_t i = 0; i < points.size(); ++i) {
            frame_points transformed;
            for (const auto &p : points[i]) {
                transformed.push_back(p + move_analyzer->get_frame_offset(i, real));
            }
            res.push_back(std::move(transformed));
        }
        return res;
    }

    /**
     * @brief Get offsets of all frames.
     * 
     * @returns offsets of all frames.
     */
    frame_points get_offsets() const {
        frame_points res;
        for (std::size_t i = 0; i < points.size(); ++i)
            res.push_back(move_analyzer->get_frame_offset(i));
        return res;
    }

    /**
     * @brief Get athlete's direction.
     * 
     * @returns athlete's direction.
     */
    direction get_direction() const {
        return move_analyzer->get_direction();
    }

private:

    /**
     * @brief Number of first frame in which athlete was detected.
     */
    std::size_t first_frame;

    /**
     * @brief Bounding box of athlete in first frame in which he was detected.
     */
    cv::Rect first_bbox;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Trackers used to track athlete.
     */
    std::vector<cv::Ptr<cv::Tracker>> trackers;

    /**
     * @brief Athlete's bounding boxes.
     * 
     * @note First dimension specifies frame number, second tracker.
     */
    std::vector<std::vector<cv::Rect>> bboxes;

    /**
     * @brief Mask of valid trackers.
     */
    std::vector<bool> valid;

    /**
     * @brief Specifies for how many frames trackers should not be updated.
     */
    std::vector<int> dont_update;

    /**
     * @brief Specifies whether tracking failed and processing is over.
     */
    bool end;

    /**
     * @brief Detected body parts of athlete in all frames in frame coordinates.
     */
    frame_video_points points;

    /**
     * @brief Athlete's body parts detector.
     */
    parts_detector parts_dtor;

    /**
     * @brief Analyzes athlete's movement.
     */
    std::optional<movement_analyzer> move_analyzer;

    /**
     * @brief Track athlete in current frame and detect his body parts.
     * 
     * @param frame Frame to be processed.
     * @param frame_no Number of given frame.
     * @returns true if detection was OK, otherwise returns false.
     * 
     * @note Returns false if athlete was not yet detected in video, although
     * it is not an error.
    */
    bool track(const cv::Mat &frame, std::size_t frame_no) {
        cv::Rect bbox = first_bbox;
        if (frame_no == first_frame) {
            move_analyzer = movement_analyzer(frame_no, frame, split(bbox), fps);
            init_trackers(frame, split(bbox));
        } else if (frame_no < first_frame) {
            return false;
        }
        if (!update_trackers(frame, frame_no)) {
            end = true;
            return false;
        }
        bbox = rect(frame) & rect(bboxes.back());
        frame_points body = parts_dtor.detect(frame, bbox, average_dist(bboxes.back()));
        points[frame_no] = std::move(body);
        return true;
    }

    /**
     * @brief Initialize trackers in given bounding boxes.
     * 
     * @param frame Frame in which to initialize trackers.
     * @param current_bboxes Bounding boxes to whose trackers should be initialize.
     */
    void init_trackers(const cv::Mat &frame, const std::vector<cv::Rect> &current_bboxes) {
        trackers.clear();
        for (const auto &bbox : current_bboxes) {
            cv::Ptr<cv::Tracker> t = cv::TrackerCSRT::create();
            t->init(frame, bbox);
            trackers.push_back(t);
        }
        valid = std::vector<bool>(current_bboxes.size(), true);
        dont_update = std::vector<int>(current_bboxes.size(), 0);
    }

    /**
     * @brief Update trackers in given frame.
     * 
     * @param frame Frame in which trackers should be updated.
     * @param frame_no Number of currently processed frame.
     */
    bool update_trackers(const cv::Mat &frame, std::size_t frame_no) {
        bool res = false;
        bboxes.emplace_back();
        for (std::size_t i = 0; i < trackers.size(); ++i) {
            cv::Rect bbox;
            if (valid[i] && trackers[i]->update(frame, bbox)) {
                bboxes.back().push_back(bbox);
                res = true;
            } else {
                bboxes.back().emplace_back();
                valid[i] = false;
            }
        }
        if (res && move_analyzer->update(frame, bboxes.back(), frame_no)) {
            update_bboxes(frame);
        }
        return res;
    }

    /**
     * @brief Update trackers' bounding boxes if they lost athlete.
     * 
     * @param frame Frame in which bounding boxes should be updated.
     */
    void update_bboxes(const cv::Mat &frame) {
        std::vector<frame_points> person_offsets = move_analyzer->get_person_offsets();
        if (move_analyzer->get_direction() != direction::unknown && person_offsets.size() && person_offsets.front().size() > GRID_UPDATE_FRAMES) {
            std::vector<cv::Point2d> offsets;
            cv::Point2d max = cv::Point2d();
            cv::Rect max_rect;
            std::size_t max_idx;
            for (std::size_t i = 0; i < person_offsets.size(); ++i) {
                frame_point m = count_offset(person_offsets[i].end() - GRID_UPDATE_FRAMES, person_offsets[i].end());
                if (valid[i] && !dont_update[i] && m && cv::norm(*m) > cv::norm(max) && check_direction(m)) {
                    max = *m;
                    offsets.push_back(*m);
                    max_rect = bboxes.back()[i];
                    max_idx = i;
                } else {
                    offsets.emplace_back();
                }
                if (!valid[i]) dont_update[i] = 0; // Invalid tracker, update ASAP.
            }

            if (max == cv::Point2d()) {
                for (auto &val : dont_update) {
                    val = std::max(--val, 0);
                }
                return;
            }

            cv::Size grid_size = first_bbox.size();
            cv::Size win_size = grid_size / GRID_SIZE;
            std::size_t row = max_idx / GRID_SIZE;
            std::size_t col = max_idx % GRID_SIZE;
            cv::Point offset(col * win_size.width, row * win_size.height);
            cv::Point grid_tl = get_center(max_rect);
            grid_tl -= cv::Point(win_size.width / 2, win_size.height / 2);
            grid_tl -= offset;

            // Fit grid inside frame
            cv::Rect grid = cv::Rect(grid_tl, grid_size) & rect(frame);
            win_size = grid.size() / GRID_SIZE;

            for (row = 0; row < GRID_SIZE; ++row) {
                offset = cv::Point(0, row * win_size.height);
                for (col = 0; col < GRID_SIZE; ++col) {
                    // Lost athlete.
                    if (cv::norm(offsets[row * GRID_SIZE + col]) < cv::norm(max) / 4) {
                        offset = cv::Point(col * win_size.width, offset.y);
                        trackers[row * GRID_SIZE + col]->init(frame, cv::Rect(grid.tl() + offset, win_size));
                        valid[row * GRID_SIZE + col] = true;
                        dont_update[row * GRID_SIZE + col] = GRID_UPDATE_FRAMES;
                    }
                }
            }
        }
    }

    /**
     * @brief Check if given point is in same direction as athlete's direction.
     * 
     * @param p Point to be checked.
     * @returns true if p is horizontally in the same direction as athlete's runup.
     */
    bool check_direction(const frame_point &p) const {
        direction dir = move_analyzer->get_direction();
        double mul = 0;
        if (dir == direction::left) {
            mul = -1;
        } else if (dir == direction::right) {
            mul = 1;
        }
        return (p) ? p->x * mul > 0 : false;
    }

    /**
     * @brief Draw athlete in `frame`.
     * 
     * Draw athlete's tracker bounding box, connected body
     * parts forming a "stickman" in `frame` and movement
     * analyzer's detections.
     * 
     * @param frame Frame in which athlete should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        if (bboxes.size() > frame_no) {
            std::size_t i = 0;
            for (const auto &bbox : bboxes[frame_no]) {
                cv::rectangle(frame, bbox.tl(), bbox.br(), cv::Scalar(255, 0, 0), 2);
                // cv::rectangle(frame, bbox.tl(), bbox.br(), cv::Scalar(i * 30, i * 30, i * 30), 1);
                ++i;
            }
        }

        if (points.size() > frame_no) {
            draw_body(frame, points[frame_no]);
        }

        if (move_analyzer) {
            move_analyzer->draw(frame, frame_no);
        }
    }

};