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
        this->end = false;
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
            if (!end && track(frame, frame_no)) {
                draw(frame, frame_no);

                // cv::imshow("frame", frame);
                // cv::waitKey(1);
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

    bool end;

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
        if (!update_trackers(frame, frame_no)) {
            end = true;
            return false;
        }
        bbox = rect(frame) & rect(trackers_bboxes.back());
        frame_points body = parts_dtor.detect(frame, bbox, average_dist(trackers_bboxes.back()));
        points[frame_no] = std::move(body);
        return true;
    }

    std::vector<cv::Rect> get_valid(const std::vector<cv::Rect> &bboxes) const noexcept {
        std::vector<cv::Rect> res;
        for (std::size_t i = 0; i < valid.size(); ++i) {
            if (valid[i]) res.push_back(bboxes[i]);
        }
        return res;
    }

    // cv::Rect filter(const cv::Mat &frame, const std::vector<cv::Rect> &bboxes) const noexcept {
    //     cv::Rect fr(0, 0, frame.cols, frame.rows);
    //     cv::Rect res = rect(bboxes);
    //     // bool found = false;
    //     // for (std::size_t i = 0; i < bboxes.size(); ++i) {
    //     //     if ()
    //     // }
    //     //
    //     return res & fr;
    // }

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
        bool res = false;
        // std::cout << trackers.size() << std::endl;
        trackers_bboxes.emplace_back();
        for (std::size_t i = 0; i < trackers.size(); ++i) {
            // std::cout << trackers_bboxes.back()[i] << std::endl;
            cv::Rect bbox;
            if (valid[i] && trackers[i]->update(frame, bbox)) {
                trackers_bboxes.back().push_back(bbox);
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
        int frames_to_check = 2;
        if (move_analyzer->get_direction() != direction::unknown && person_offsets.size() && person_offsets.front().size() > frames_to_check) {
            std::vector<cv::Point2d> mean_deltas;
            cv::Point2d max = cv::Point2d();
            cv::Rect max_rect;
            std::size_t max_idx;
            for (std::size_t i = 0; i < person_offsets.size(); ++i) {
                frame_point m = count_mean_delta(person_offsets[i].end() - frames_to_check, person_offsets[i].end());
                if (valid[i] && !dont_update[i] && m && cv::norm(*m) > cv::norm(max) && check_direction(m)) {
                    max = *m;
                    mean_deltas.push_back(*m);
                    max_rect = trackers_bboxes.back()[i];
                    // std::cout << max_rect << std::endl;
                    max_idx = i;
                    // std::cout << max_idx << std::endl;
                } else {
                    mean_deltas.emplace_back();
                }
                if (!valid[i]) dont_update[i] = 0; // Invalid tracker, update ASAP.
            }

            if (max == cv::Point2d()) {
                for (auto &val : dont_update) {
                    val = std::max(--val, 0);
                }
                return;
            }

            cv::Size grid_size = parts_dtor.get_scale_factor() * first_bbox.size();
            cv::Size win_size = grid_size / 3;
            std::size_t row = max_idx / 3;
            std::size_t col = max_idx % 3;
            cv::Point offset(col * win_size.width, row * win_size.height);
            cv::Point grid_tl = get_center(max_rect);
            grid_tl -= cv::Point(win_size.width / 2, win_size.height / 2);
            grid_tl -= offset;
            // std::cout << grid_tl << grid_size << rect(frame) << std::endl;
            cv::Rect grid = cv::Rect(grid_tl, grid_size) & rect(frame);
            // std::cout << grid << std::endl;
            

            for (row = 0; row < 3; ++row) {
                offset = cv::Point(0, row * win_size.height);
                for (col = 0; col < 3; ++col) {
                    if (cv::norm(mean_deltas[row * 3 + col]) < cv::norm(max) / 2) {
                        offset = cv::Point(col * win_size.width, offset.y);
                        trackers[row * 3 + col]->init(frame, cv::Rect(grid.tl() + offset, grid.size() / 3));
                        valid[row * 3 + col] = true;
                        dont_update[row * 3 + col] = frames_to_check;
                    }
                }
            }

            // std::cout << max_rect.tl() << " " << max_rect.br() << std::endl;
            // cv::Size size(max_rect.br() - max_rect.tl());
            // cv::Point offset_x = parts_dtor.rotate(cv::Size(size.width, 0));
            // cv::Point offset_y = parts_dtor.rotate(cv::Size(0, size.height));
            // size = cv::Size(parts_dtor.get_scale_factor() * size.width, parts_dtor.get_scale_factor() * size.height);
            // int valid_row = max_idx / 3;
            // int valid_col = max_idx % 3;
            // for (std::size_t i = 0; i < mean_deltas.size(); ++i) {
            //     if (valid[i] && !dont_update[i] && cv::norm(2 * mean_deltas[i]) < cv::norm(max)) {
            //         if (max.x || max.y) {
            //             int row = i / 3;
            //             int col = i % 3;
            //             cv::Point origin = max_rect.tl() + cv::Point((col - valid_col) * offset_x) + cv::Point((row - valid_row) * offset_y);
            //             // std::cout << parts_dtor.get_scale_factor() << cv::Rect(origin, size) << fit_inside(frame, cv::Rect(origin, size));
            //             // std::cout << "updating" << i << std::endl;
            //             if (area(cv::Rect(origin, size) & rect(frame)) == 0 ||
            //                 (cv::Rect(origin, size) & rect(frame)).width < 3 ||
            //                 (cv::Rect(origin, size) & rect(frame)).height < 3) {
            //                     valid[i] = false;
            //                     continue;
            //             }
            //             // std::cout << "init" << i << cv::Rect(origin, size) << fit_inside(frame, cv::Rect(origin, size)) << std::endl;
            //             trackers[i]->init(frame, rect(frame) & cv::Rect(origin, size));
            //             // std::cout << "done" << std::endl;
            //             valid[i] = true;
            //             // std::cout << "updating " << row << ":" << col << " from " << trackers_bboxes.back()[i] << std::endl;
            //             dont_update[i] = frames_to_check;
            //         }
            //     } else {
            //         dont_update[i] = std::max(0, --dont_update[i]);
            //     }
            // }
        }
    }

    bool check_direction(const frame_point &p) const noexcept {
        direction dir = move_analyzer->get_direction();
        double mul = 0;
        if (dir == direction::left) {
            mul = -1;
        } else if (dir == direction::right) {
            mul = 1;
        }
        return (p) ? p->x * mul > 0 : false;
    }

    bool check(const cv::Rect &r, const cv::Rect &s) const noexcept {
        cv::Rect overlap = r & s;
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
            std::size_t i = 0;
            for (const auto &bbox : trackers_bboxes[frame_no]) {
                cv::rectangle(frame, bbox.tl(), bbox.br(), cv::Scalar(i * 30, i * 30, i * 30), 1);
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