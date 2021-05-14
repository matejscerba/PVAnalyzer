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
    person(std::size_t frame_no, const cv::Rect &box, double fps) noexcept {
        valid_tracker = false;
        move_analyzer.reset();
        this->first_frame = frame_no;
        this->first_bbox = box;
        this->fps = fps;
        tracker = cv::TrackerCSRT::create();
        parts_dtor = parts_detector();
    }

    /**
     * @brief Detect athlete and his body parts in given video.
     * 
     * @param raw_frames Frames of video to be processed.
     * @returns Frames with drawings of detections.
     */
    std::vector<cv::Mat> detect(const std::vector<cv::Mat> &raw_frames) noexcept {
        std::vector<cv::Mat> frames;

        cv::Mat frame;
        bboxes = std::vector<std::optional<cv::Rect>>(raw_frames.size(), std::nullopt);
        points = frame_video_points(raw_frames.size(), frame_points(NPOINTS, std::nullopt));
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Processing frame " << frame_no << "/" << raw_frames.size() - 1 << std::endl;

            frame = raw_frames[frame_no].clone();

            // Detect athlete's body in current frame.
            if (track(frame, frame_no)) {
                draw(frame, frame_no);

                cv::imshow("frame", frame);
                cv::waitKey(1);
            }

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
    frame_video_points get_points(bool real = false) const noexcept {
        frame_video_points res;
        for (std::size_t i = 0; i < points.size(); ++i) {
            frame_points transformed;
            for (const auto &p : points[i]) {
                transformed.push_back(p + move_analyzer->frame_offset(i, real));
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
    frame_points get_offsets() const noexcept {
        frame_points res;
        for (std::size_t i = 0; i < points.size(); ++i)
            res.push_back(move_analyzer->frame_offset(i));
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
     * @brief Specifies if tracker is valid.
     */
    bool valid_tracker;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Tracker used to track athlete in frames.
     */
    cv::Ptr<cv::Tracker> tracker;

    /**
     * @brief Tracker's bounding boxes in all frames.
     */
    std::vector<std::optional<cv::Rect>> bboxes;

    /**
     * @brief Detected body parts of athlete in all frames in frame coordinates 
     */
    frame_video_points points;

    /**
     * @brief Athlete's body parts detector.
     */
    parts_detector parts_dtor;

    /**
     * @brief Analyzes this person's movement.
     */
    std::optional<movement_analyzer> move_analyzer;

    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<std::vector<cv::Rect>> trackers_bboxes;
    std::vector<bool> valid;
    std::vector<int> dont_update;

    /**
     * @brief Track person in next frame.
     * 
     * Track athlete in given frame and detect athlete's body parts.
     * 
     * @param frame Frame in which person should be detected.
     * @param frame_no Number of given frame.
     * @returns true if detection was OK, false if athlete was not detected.
     * 
     * @note Returns false if athlete was not yet detected in video, although
     * it is not an error.
    */
    bool track(const cv::Mat &frame, std::size_t frame_no) noexcept {
        // cv::Rect bbox = first_bbox;
        // valid_tracker = (frame_no == first_frame) ||
        //                 (valid_tracker && tracker->update(frame, bbox) && is_inside(bbox, frame));

        // if (!valid_tracker) return false;

        // bbox = get_tracker_bbox(frame, body, bbox);
        // tracker->init(frame, bbox);
        // bboxes[frame_no] = bbox;
        // if (frame_no == first_frame) {
        //     move_analyzer = movement_analyzer(frame_no, frame, bbox, fps);
        // } else {
        //     valid_tracker = move_analyzer->update(frame, bbox, frame_no);
        // }
        // return valid_tracker;
        cv::Rect bbox = first_bbox;
        if (frame_no == first_frame) {
            std::vector<cv::Rect> bboxes = split(bbox);
            move_analyzer = movement_analyzer(frame_no, frame, bboxes, fps);
            init_trackers(frame, bboxes);
        } else if (frame_no < first_frame) {
            return false;
        }
        if (!update_trackers(frame, frame_no)) return false;
        bbox = filter(frame, trackers_bboxes.back());
        frame_points body = parts_dtor.detect(frame, bbox, average_dist(trackers_bboxes.back()));
        points[frame_no] = std::move(body);
        return true;
    }

    cv::Rect filter(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes) const noexcept {
        cv::Rect fr(0, 0, frame.cols, frame.rows);
        cv::Rect res = rect(bboxes);
        // bool found = false;
        // for (std::size_t i = 0; i < bboxes.size(); ++i) {
        //     if ()
        // }
        //
        return res & fr;
    }

    void init_trackers(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes) noexcept {
        trackers.clear();
        for (const auto &bbox : bboxes) {
            cv::Ptr<cv::Tracker> t = cv::TrackerCSRT::create();
            t->init(frame, bbox);
            trackers.push_back(t);
        }
        valid = std::vector<bool>(bboxes.size(), true);
        dont_update = std::vector<int>(bboxes.size(), 0);
    }

    bool update_trackers(const cv::Mat &frame, std::size_t frame_no) noexcept {
        // std::cout << "b" << std::endl;
        trackers_bboxes.push_back(std::vector<cv::Rect>(trackers.size(), cv::Rect()));
        bool res = false;
        // std::cout << trackers.size() << std::endl;
        for (std::size_t i = 0; i < trackers.size(); ++i) {
            // std::cout << trackers_bboxes.back()[i] << std::endl;
            if (valid[i] && trackers[i]->update(frame, trackers_bboxes.back()[i])) {
                res = true;
                // std::cout << trackers_bboxes.back()[i] << std::endl << std::endl;
            } else {
                valid[i] = false;
            }
        }
        if (res && move_analyzer->update(frame, trackers_bboxes.back(), frame_no)) {
            update_bboxes(frame);
        }
        return res;
    }

    void update_bboxes(const cv::Mat &frame) noexcept {
        std::vector<frame_points> person_offsets = move_analyzer->get_person_offsets();
        int frames_to_check = 3;
        if (move_analyzer->get_direction() != direction::unknown && person_offsets.size() && person_offsets.front().size() > frames_to_check) {
            std::vector<cv::Point2d> mean_deltas;
            cv::Point2d max = cv::Point2d();
            cv::Rect max_rect;
            std::size_t max_idx;
            for (std::size_t i = 0; i < person_offsets.size(); ++i) {
                frame_point m = count_mean_delta(person_offsets[i].end() - frames_to_check, person_offsets[i].end());
                if (valid[i] && !dont_update[i] && m && cv::norm(*m) > cv::norm(max)) {
                    max = *m;
                    mean_deltas.push_back(*m);
                    max_rect = trackers_bboxes.back()[i];
                    max_idx = i;
                } else {
                    mean_deltas.emplace_back();
                }
                if (!valid[i]) dont_update[i] = 0; // Invalid tracker, update ASAP.
            }
            cv::Size size(max_rect.br() - max_rect.tl());
            cv::Point offset_x = parts_dtor.rotate(cv::Size(size.width, 0));
            cv::Point offset_y = parts_dtor.rotate(cv::Size(0, size.height));
            int valid_row = max_idx / 3;
            int valid_col = max_idx % 3;
            for (std::size_t i = 0; i < mean_deltas.size(); ++i) {
                if (!dont_update[i] && cv::norm(2 * mean_deltas[i]) < cv::norm(max)) {
                    int row = i / 3;
                    int col = i % 3;
                    cv::Point origin = max_rect.tl() + cv::Point((col - valid_col) * offset_x) + cv::Point((row - valid_row) * offset_y);
                    trackers[i]->init(frame, fit_inside(frame, cv::Rect(origin, size)));
                    valid[i] = true;
                    // std::cout << "updating " << row << ":" << col << " from " << trackers_bboxes.back()[i] << std::endl;
                    dont_update[i] = frames_to_check;
                } else {
                    dont_update[i] = std::max(0, --dont_update[i]);
                }
            }
        }
    }

    bool check(const cv::Rect &r, const cv::Rect &s) const noexcept {
        int top = std::max(r.y, s.y);
        int right = std::min(r.x + r.width, s.x + s.width);
        int bottom = std::min(r.y + r.height, s.x + s.height);
        int left = std::max(r.x, s.x);
        cv::Rect overlap(left, top, right - left, bottom - top);
        return area(overlap) / area(r) >= 0.8 && area(overlap) / area(s) >= 0.8;
    }

    /**
     * @brief Update tracker's bounding box so that it encloses athlete's torso.
     * 
     * Make bounding box as small as possible so that it holds whole torso.
     * 
     * @param frame Frame in which body parts were detected.
     * @param body Athlete's body parts in frame.
     * @param bbox Current tracker's bouning box.
     * @returns Updated tracker's bounding box.
     */
    cv::Rect get_tracker_bbox(const cv::Mat &frame, const frame_points &body, const cv::Rect &bbox) noexcept {
        if ((body[body_part::l_hip] || body[body_part::r_hip]) && body[body_part::head]) {
            std::vector<cv::Point2d> torso_corners;
            cv::Point2d head = *body[body_part::head];
            cv::Point2d hip;
            if (body[body_part::l_hip] && body[body_part::r_hip])
                hip = (*body[body_part::l_hip] + *body[body_part::r_hip]) / 2;
            else if (body[body_part::l_hip])
                hip = *body[body_part::l_hip];
            else
                hip = *body[body_part::r_hip];
            cv::Point2d torso = hip - head;
            cv::Point2d perp(torso.y, - torso.x);
            perp *= 1.0 / 4.0;
            torso_corners.push_back(head + perp);
            torso_corners.push_back(head - perp);
            torso_corners.push_back(hip + perp);
            torso_corners.push_back(hip - perp);
            double top = torso_corners.front().y;
            double right = torso_corners.front().x;
            double bottom = torso_corners.front().y;
            double left = torso_corners.front().x;
            for (const auto &c : torso_corners) {
                top = std::min(top, c.y);
                right = std::max(right, c.x);
                bottom = std::max(bottom, c.y);
                left = std::min(left, c.x);
            }
            cv::Rect new_bbox(left, top, right - left, bottom - top);
            return check(new_bbox, bbox) ? new_bbox : bbox;
        }
        return bbox;
    }

    /**
     * @brief Draw athlete in `frame`.
     * 
     * Draw athlete's tracker bounding box, connected body
     * parts forming a "stickman" in `frame` and movement
     * analyzer's detections.
     * 
     * @param frame Frame in which this person should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
        // if (bboxes.size() > frame_no && bboxes[frame_no]) {
        //     cv::Scalar color(0, 0, 255);
        //     if (move_analyzer->vault_frames(frame_no))
        //         color = cv::Scalar(0, 255, 0);

        //     cv::rectangle(frame, bboxes[frame_no]->tl(), bboxes[frame_no]->br(), color, 1);
        // }

        if (trackers_bboxes.size() > frame_no) {
            for (const auto &bbox : trackers_bboxes[frame_no]) {
                cv::rectangle(frame, bbox.tl(), bbox.br(), cv::Scalar(0, 0, 255), 1);
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