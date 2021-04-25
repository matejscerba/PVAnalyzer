#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <optional>

#include "movement_analyzer.hpp"
#include "parts_detector.hpp"

/**
 * @brief Represents person through whole video.
 * 
 * Holds information about a person through whole video (or its part
 * when certain person is in frame). Handles transitions between frames.
 * Handles rotations during vault as well.
*/
class person {
public:

    /**
     * @brief Default constructor.
     * 
     * @param frame_no Number of frame, in which this person was detected for the first time.
     * @param frame Current frame, this person was already detected here.
     * @param fps Frame rate of processed video.
     * @param box Bounding box of this person in `frame`.
     * @param net Deep neural network used for detecting person's body parts.
    */
    person(std::size_t frame_no, const cv::Rect &box, double fps) noexcept {
        valid_tracker = false;
        move_analyzer = std::nullopt;
        last_angle = 0.0;
        this->first_frame = frame_no;
        this->first_bbox = box;
        this->fps = fps;
        tracker = cv::TrackerCSRT::create();
        parts_dtor = parts_detector();
    }

    std::vector<cv::Mat> detect(const std::vector<cv::Mat> &raw_frames) {
        std::vector<cv::Mat> frames;

        cv::Mat frame;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Processing frame " << frame_no << std::endl;

            frame = raw_frames[frame_no].clone();

            // Detect athlete's body in current frame.
            if (track(frame, frame_no)) {
                draw(frame, frame_no);
            }

            // cv::imshow("frame", frame);
            // cv::waitKey();

            // Save current modified frame.
            frames.push_back(frame);
        }
        return frames;
    }

    /**
     * @brief Get point of all body parts of person.
     * 
     * @param real Transforms detected points into real life coordinates if true.
     * @returns detected body parts in part of video where person was detected.
     */
    video_body get_points(bool real = false) const {
        video_body res;
        for (std::size_t i = 0; i < points.size(); ++i) {
            frame_body transformed;
            std::transform(points[i].begin(), points[i].end(), std::back_inserter(transformed),
                           [i, this, real](const frame_part &p) { return p + this->move_analyzer->frame_offset(i, real); }
            );
            res.push_back(std::move(transformed));
        }
        return res;
    }

    std::vector<std::optional<cv::Point2d>> get_offsets() const noexcept {
        std::vector<std::optional<cv::Point2d>> res;
        for (std::size_t i = 0; i < points.size(); ++i)
            res.push_back(move_analyzer->frame_offset(i));
        return res;
    }

    direction get_direction() const {
        return move_analyzer->get_direction();
    }

private:

    std::size_t first_frame;

    cv::Rect first_bbox;

    bool valid_tracker;

    double fps;

    double last_angle;

    /// @brief Tracker used to track person in frames.
    cv::Ptr<cv::Tracker> tracker;

    person_corners scaled_corners;

    std::vector<std::optional<cv::Mat>> cropped_frames;

    /**
     * @brief Corners of bounding boxes of person in each frame.
     * 
     * Corners of bounding box in frame number `first_frame_no` are stored at index 0.
     * Certain corners are supposed to be accessed using `enum corner`. For example:
     * bottom left corner of bounding box in current frame should be accessed as
     * `corners[current_frame_no - first_frame_no][corner::bl]`.
     * 
     * @note Bounding box (which corners represent) can be rotated, so it
     *     is saved as individual corners.
    */
    person_corners corners;

    video_body points;

    parts_detector parts_dtor;

    /// @brief Analyzes this person's movement.
    std::optional<movement_analyzer> move_analyzer;

    /**
     * @brief Compute position of given points after transformation.
     * 
     * @param src Vector of given points.
     * @param back Indicator whether to transform points in the opposite rotation.
     * @returns vector of points after rotation specified `frame_no` (used to compute
     *     rotation angle), in the opposite direction if `back` is true.
    */
    std::vector<cv::Point2d> transform(const std::vector<cv::Point2d> &src, bool back = false) {
        std::vector<cv::Point2d> res;
        if (corners.size() && corners.back()) {
            double angle = get_angle();
            if (back) angle *= -1;
            cv::Point center = get_center(*corners.back());
            cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::transform(src, res, rotation);
        } else {
            res = src;
        }
        return res;
    }

    /**
     * @brief Compute angle from given number of frame.
     * 
     * @param frame_no Number of frame used to determine angle of rotation.
     */
    double get_angle() {
        double res = 0;
        if (points.size()) {
            frame_body body = points.back();
            std::optional<double> angle = get_vertical_tilt_angle(
                (body[body_part::l_hip] + body[body_part::r_hip]) / 2.0,
                body[body_part::head]
            );
            res = angle ? - *angle : last_angle;
        }
        return res;
    }

    /**
     * @brief Rotate frame so that person's angle of rotation is as low as possible.
     * 
     * @param frame Frame to be rotated.
     * @param frame_no Number of frame to be rotated.
     * @returns rotated frame.
     */
    cv::Mat rotate(const cv::Mat &frame, double shift = 0.0) {
        cv::Mat rotated;
        if (corners.size() && corners.back()) {
            double angle = get_angle() + shift;
            if (shift == 0.0) last_angle = angle;
            cv::Point center = get_center(*corners.back());
            cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(frame, rotated, rotation, frame.size());
        } else {
            rotated = frame;
        }
        return rotated;
    }

    /**
     * @brief Scale rectangle by given factor if it fits inside frame.
     * 
     * @param rect Given rectangle to be scaled.
     * @param frame Frame in which scaled rectangle must fit.
     * @returns scaled rectangle if it fits inside frame, `rect` otherwise.
     * 
     * @note If scaled rectangle does not fit inside frame, scaling is not performed.
    */
    cv::Rect scale(const cv::Rect &rect, const cv::Mat &frame) {
        cv::Point center = get_center(rect);
        cv::Point diag = center - rect.tl();
        cv::Point tl = center - scale_factor * diag;
        cv::Point br = center + scale_factor * diag;
        int top = std::max(0, tl.y);
        int right = std::min(frame.cols, br.x);
        int bottom = std::min(frame.rows, br.y);
        int left = std::max(0, tl.x);
        return cv::Rect(left, top, right - left, bottom - top);
    }

    cv::Rect scale_square(const cv::Rect &rect, const cv::Mat &frame) {
        cv::Point center = get_center(rect);
        cv::Point diag = center - rect.tl();
        int diff = std::max(diag.x, diag.y);
        int top = std::max(0, center.y - diff);
        int right = std::min(frame.cols, center.x + diff);
        int bottom = std::min(frame.rows, center.y + diff);
        int left = std::max(0, center.x - diff);
        return cv::Rect(left, top, right - left, bottom - top);
    }

    /**
     * @param frame_no Number of frame where to get bounding box.
     * @returns bounding box of this person in current frame.
     */
    std::optional<cv::Rect> bbox(std::size_t frame_no) const {
        if (corners[frame_no]) {
            return cv::Rect(
                cv::Point((*corners[frame_no])[corner::tl]),
                cv::Point((*corners[frame_no])[corner::br])
            );
        }
        return std::nullopt;
    }

    /// @returns width of this person's bounding box in given frame.
    double last_width() const {
        return cv::norm((*corners.back())[corner::tr] - (*corners.back())[corner::tl]);
    }

    /// @returns height of this person's bounding box in given frame.
    double last_height() const {
        return cv::norm((*corners.back())[corner::bl] - (*corners.back())[corner::tl]);
    }

    void update_properties(const cv::Mat &frame, const cv::Rect &bbox) noexcept {
        if (valid_tracker) {
            cv::Rect scaled = scale(bbox, frame);
            scaled_corners.push_back(transform(get_corners(scaled), true));
            // cropped_frames.push_back(frame(scaled).clone());
            // cv::imshow("in_update", *cropped_frames.back());
            corners.push_back(transform(get_corners(bbox), true));
        } else {
            scaled_corners.push_back(std::nullopt);
            // cropped_frames.push_back(std::nullopt);
            corners.push_back(std::nullopt);
        }
        points.push_back(frame_body(npoints, std::nullopt));
    }

    void update_points(frame_body &body, double shift) noexcept {
        cv::Mat rotation = cv::getRotationMatrix2D(cv::Point(), -last_angle - shift, 1.0);
        std::for_each(body.begin(), body.end(), [this, rotation](frame_part &p){
            if (p) {
                std::vector<cv::Point2d> res;
                cv::transform(std::vector<cv::Point2d>{*p}, res, rotation);
                p = (*(this->scaled_corners.back()))[corner::tl] + res.front();
            }
        });
        points.back() = body;
    }

    /**
     * @brief Track person in next frame.
     * 
     * Rotates frame so that person's angle of rotation is as low as possible.
     * 
     * @param frame Next frame in which person should be tracked.
     * @param frame_no Number of given frame.
     * @returns true if detection was OK, false if an error occured.
    */
    bool track(const cv::Mat &frame, std::size_t frame_no) {
        cv::Rect bbox = first_bbox;
        cv::Mat rotated = rotate(frame);
        if (frame_no == first_frame) {
            tracker->init(rotated, bbox);
            move_analyzer = movement_analyzer(frame_no, rotated, bbox, fps);
            valid_tracker = true;
        } else if (valid_tracker && tracker->update(rotated, bbox) && is_inside(get_corners(bbox), rotated)) {
            valid_tracker = move_analyzer->update(frame, bbox, frame_no);
        } else {
            valid_tracker = false;
        }
        update_properties(rotated, bbox);
        if (valid_tracker) {
            cv::Point2d size(last_width(), last_height());
            cv::Point2d center = get_center(*corners.back());
            cv::Point2d tl = center - size / 2;
            cv::Point2d br = center + size / 2;
            cv::Rect rect(tl, br);
            int max_parts = -1;
            for (std::size_t i = 0; i < SHIFTS.size(); ++i) {
                // Rect
                cv::Mat r;
                double angle = get_angle() + SHIFTS[i];
                cv::Point center = get_center(*corners.back());
                cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
                cv::Mat back_rot = cv::getRotationMatrix2D(center, -angle, 1.0);
                cv::warpAffine(frame, r, rotation, frame.size());
                cv::Rect2d scaled_rect = scale(rect, r);
                frame_body detected = parts_dtor.deep_detect(r(scaled_rect));
                if (check(detected) > max_parts) {
                    max_parts = check(detected);
                    std::for_each(detected.begin(), detected.end(), [scaled_rect, back_rot](frame_part &part) {
                        if (part) {
                            *part += scaled_rect.tl();
                            std::vector<cv::Point2d> res;
                            cv::transform(std::vector<cv::Point2d>{*part}, res, back_rot);
                            part = res.front();
                        }
                    });
                    points.back() = detected;
                }
                if (max_parts == npoints) break;
                if (move_analyzer->vault_frames(frame_no)) continue;
                // Square
                scaled_rect = scale_square(scaled_rect, r);
                detected = parts_dtor.deep_detect(r(scaled_rect));
                if (check(detected) > max_parts) {
                    max_parts = check(detected);
                    std::for_each(detected.begin(), detected.end(), [scaled_rect, back_rot](frame_part &part){
                        if (part) {
                            *part += scaled_rect.tl();
                            std::vector<cv::Point2d> res;
                            cv::transform(std::vector<cv::Point2d>{*part}, res, back_rot);
                            part = res.front();
                        }
                    });
                    points.back() = detected;
                }
                if (max_parts == npoints) break;
                // TODO: Update properties.
            }
        }
        return valid_tracker;
    }

    int check(const frame_body &body) const noexcept {
        int res = 0;
        for (const auto &part : body) {
            if (part) ++res;
        }
        return res;
    }

    // cv::Rect rotate_last_bbox() noexcept {
    //     cv::Point2d last_center = get_center(*corners.back());
    //     cv::Point2d rotated = transform(std::vector<cv::Point2d>{last_center}).front();
    //     cv::Point2d size(last_width(), last_height());
    //     return cv::Rect(
    //         rotated - size / 2,
    //         rotated + size / 2
    //     );
    // }

    /**
     * @brief Detect body parts of this person inside given frame.
     * 
     * Use deep neural network to detect body parts of this person inside frame,
     * crop frame to smaller input for speed optimization.
     * 
     * @param frame Frame which to crop and detect body parts on.
     * @param frame_no Number of given frame.
    */

    /**
     * @brief Draw this person in `frame`.
     * 
     * Draw this person's unscaled bounding box, scaled bounidng box
     * and connected body parts forming a "stickman" in `frame`.
     * 
     * @param frame Frame in which this person should be drawn.
     * @param frame_no Number of given frame.
    */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        
        if (corners.size() > frame_no && corners[frame_no]) {
        // Rectangle which is tracked.
            cv::Point2d tl = (*corners[frame_no])[corner::tl];
            cv::Point2d tr = (*corners[frame_no])[corner::tr];
            cv::Point2d bl = (*corners[frame_no])[corner::bl];
            cv::Point2d br = (*corners[frame_no])[corner::br];

            cv::Scalar color(0, 0, 255);
            if (move_analyzer->vault_frames(frame_no))
                color = cv::Scalar(0, 255, 0);

            cv::line(frame, tl, tr, color, 1);
            cv::line(frame, tr, br, color, 1);
            cv::line(frame, br, bl, color, 1);
            cv::line(frame, bl, tl, color, 1);

            // Scaled rectangle.
            tl = (*scaled_corners[frame_no])[corner::tl];
            tr = (*scaled_corners[frame_no])[corner::tr];
            bl = (*scaled_corners[frame_no])[corner::bl];
            br = (*scaled_corners[frame_no])[corner::br];

            color = cv::Scalar(0, 0, 127);
            if (move_analyzer->vault_frames(frame_no))
                color = cv::Scalar(0, 127, 0);

            cv::line(frame, tl, tr, color, 1);
            cv::line(frame, tr, br, color, 1);
            cv::line(frame, br, bl, color, 1);
            cv::line(frame, bl, tl, color, 1);
        }

        // std::cout << "." << points.size() << " " << idx << " " << frame_no << std::endl;
        // Body parts if they were detected.
        if (points.size() > frame_no) {
            for (int n = 0; n < npairs; n++) {
                std::size_t a_idx = pairs[n][0];
                std::size_t b_idx = pairs[n][1];
                std::optional<cv::Point2d> a = points[frame_no][a_idx];
                std::optional<cv::Point2d> b = points[frame_no][b_idx];

                // Check if points `a` and `b` are valid.
                if (a && b) {
                    cv::Scalar c(0, 255, 255);
                    if ((a_idx == body_part::l_ankle) || (a_idx == body_part::l_knee) || (a_idx == body_part::l_hip) ||
                        (a_idx == body_part::l_wrist) || (a_idx == body_part::l_elbow) || (a_idx == body_part::l_shoulder) ||
                        (b_idx == body_part::l_ankle) || (b_idx == body_part::l_knee) || (b_idx == body_part::l_hip) ||
                        (b_idx == body_part::l_wrist) || (b_idx == body_part::l_elbow) || (b_idx == body_part::l_shoulder)) {
                            c = cv::Scalar(255, 0, 255);
                    } else if ((a_idx == body_part::r_ankle) || (a_idx == body_part::r_knee) || (a_idx == body_part::r_hip) ||
                        (a_idx == body_part::r_wrist) || (a_idx == body_part::r_elbow) || (a_idx == body_part::r_shoulder) ||
                        (b_idx == body_part::r_ankle) || (b_idx == body_part::r_knee) || (b_idx == body_part::r_hip) ||
                        (b_idx == body_part::r_wrist) || (b_idx == body_part::r_elbow) || (b_idx == body_part::r_shoulder)) {
                            c = cv::Scalar(255, 255, 0);
                    }
                    // Draw points representing joints and connect them with lines.
                    cv::line(frame, *a, *b, c, 2);
                    cv::circle(frame, *a, 2, cv::Scalar(0, 0, 255), -1);
                    cv::circle(frame, *b, 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        // Movement analyzer.
        if (move_analyzer)
            move_analyzer->draw(frame, frame_no);
    }

};