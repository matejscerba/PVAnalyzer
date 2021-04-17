#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <optional>

#include "movement_analyzer.hpp"

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
    person(std::size_t frame_no, const cv::Rect &box, double fps, cv::dnn::Net &net) noexcept {
        valid_tracker = false;
        move_analyzer = std::nullopt;
        this->first_frame = frame_no;
        this->first_bbox = box;
        this->fps = fps;
        this->net = net;
        tracker = cv::TrackerCSRT::create();
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
    double width(std::size_t frame_no) const {
        return cv::norm((*corners[frame_no])[corner::tr] - (*corners[frame_no])[corner::tl]);
    }

    /// @returns height of this person's bounding box in given frame.
    double height(std::size_t frame_no) const {
        return cv::norm((*corners[frame_no])[corner::bl] - (*corners[frame_no])[corner::tl]);
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
        if (frame_no < first_frame) {
            angles.push_back(0.0);
            corners.push_back(std::nullopt);
            scaled_corners.push_back(std::nullopt);
            cropped_frames.push_back(std::nullopt);
            points.push_back(frame_body(npoints, std::nullopt));
        } else if (frame_no == first_frame) {
            tracker->init(frame, first_bbox);
            move_analyzer = movement_analyzer(frame_no, frame, first_bbox, fps);
            cv::Rect scaled = scale(first_bbox, frame);
            angles.push_back(0.0);
            corners.push_back(get_corners(first_bbox));
            scaled_corners.push_back(get_corners(scaled));
            cropped_frames.push_back(frame(scaled).clone());
            points.push_back(frame_body(npoints, std::nullopt));
        } else {
            cv::Rect bbox;
            cv::Mat rotated = rotate(frame);
            if (tracker->update(frame, bbox) && is_inside(get_corners(bbox), frame)) {
                cv::Rect scaled = scale(bbox, frame);
                angles.push_back(0.0);
                corners.push_back(get_corners(bbox));
                scaled_corners.push_back(get_corners(scaled));
                cropped_frames.push_back(frame(scaled).clone());
                points.push_back(frame_body(npoints, std::nullopt));
                return move_analyzer->update(frame, bbox, frame_no);
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Detect body parts of this person inside given frame.
     * 
     * Use deep neural network to detect body parts of this person inside frame,
     * crop frame to smaller input for speed optimization.
     * 
     * @param frame Frame which to crop and detect body parts on.
     * @param frame_no Number of given frame.
    */
    void detect(const cv::Mat &frame, std::size_t frame_no) {
        std::optional<cv::Mat> output = std::nullopt;
        // Crop person from frame.
        if (cropped_frames[frame_no]) {
            cv::Mat input = *cropped_frames[frame_no];
            // Process cropped frame.
            cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
            net.setInput(blob);
            output = net.forward();
        }
        extract_points(output, frame_no);
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
            tl = (*corners[frame_no])[corner::tl]
                + (1.0 - scale_factor) * 0.5 * ((*corners[frame_no])[corner::br] - (*corners[frame_no])[corner::tl]);
            tr = (*corners[frame_no])[corner::tr]
                + (1.0 - scale_factor) * 0.5 * ((*corners[frame_no])[corner::bl] - (*corners[frame_no])[corner::tr]);
            bl = (*corners[frame_no])[corner::bl]
                - (1.0 - scale_factor) * 0.5 * ((*corners[frame_no])[corner::bl] - (*corners[frame_no])[corner::tr]);
            br = (*corners[frame_no])[corner::br]
                - (1.0 - scale_factor) * 0.5 * ((*corners[frame_no])[corner::br] - (*corners[frame_no])[corner::tl]);

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

private:

    friend class vault_analyzer;

    std::size_t first_frame;

    cv::Rect first_bbox;

    bool valid_tracker;

    double fps;

    /// @brief Deep neural network used for detecting person's body parts.
    cv::dnn::Net net;

    /// @brief Tracker used to track person in frames.
    cv::Ptr<cv::Tracker> tracker;

    std::vector<double> angles;

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

    /**
     * @brief Body parts of person in each frame.
     * 
     * Body parts are saved in similar way as corners, after accessing body parts
     * for certain frame, indices correspond to `pairs`.
     */
    video_body points;

    /// @brief Analyzes this person's movement.
    std::optional<movement_analyzer> move_analyzer;

    void update_properties(const cv::Mat &frame, const cv::Rect &bbox) noexcept {
        cv::Rect scaled = scale(bbox, frame);
        angles.push_back(get_angle());
        corners.push_back(transform(get_corners(bbox), frame, true));
        scaled_corners.push_back(transform(get_corners(scaled), frame, true));
        cropped_frames.push_back(frame(scaled).clone());
    }

    /**
     * @brief Save detected body parts to `points`.
     * 
     * Process output of deep neural network used to detect body parts
     * and save correct values to points.
     * 
     * @param output Result given by `net.forward()` applied on scaled bounding
     *     box of this person in current frame.
     * @param frame_no Number of frame in which to process `output`.
    */
    void extract_points(std::optional<cv::Mat> &output, std::size_t frame_no) {
        if (output) {
            int h = output->size[2];
            int w = output->size[3];

            // Scale by `scaling_factor` if last rectangle was scaled.
            double factor = 1;
            double sx = factor * width(frame_no) / (double)w;
            double sy = factor * height(frame_no) / (double)h;

            // Get points from output.
            for (int n = 0; n < npoints; n++) {
                cv::Mat probMat(h, w, CV_32F, output->ptr(0, n));

                // Get point in output with maximum probability of "being point `n`".
                frame_part p = std::nullopt;
                cv::Point max;
                double prob;
                cv::minMaxLoc(probMat, 0, &prob, 0, &max);

                // Check point probability against a threshold
                if (prob > detection_threshold) {
                    p = max;
                    p->x *= sx; p->y *= sy; // Scale point so it fits original frame.

                    // Move point `p` so it is in correct position in frame.
                    p = (*corners[frame_no])[corner::tl]
                        + (1.0 - scale_factor) * 0.5 * ((*corners[frame_no])[corner::br] - (*corners[frame_no])[corner::tl])
                        + p->x * ((*corners[frame_no])[corner::tr] - (*corners[frame_no])[corner::tl]) / width(frame_no)
                        + p->y * ((*corners[frame_no])[corner::bl] - (*corners[frame_no])[corner::tl]) / height(frame_no);
                }

                points[frame_no][n] = p;
            }
        }
    }

    /**
     * @brief Compute position of given points after transformation.
     * 
     * @param src Vector of given points.
     * @param frame Frame in which points are detected.
     * @param frame_no Number of given frame.
     * @param back Indicator whether to transform points in the opposite rotation.
     * @returns vector of points after rotation specified `frame_no` (used to compute
     *     rotation angle), in the opposite direction if `back` is true.
    */
    std::vector<cv::Point2d> transform(const std::vector<cv::Point2d> &src,
                                       const cv::Mat &frame,
                                       bool back = false) const {
        double angle = get_angle();
        if (back) angle *= -1;
        cv::Mat rotation = cv::getRotationMatrix2D(get_center(frame), angle, 1.0);
        std::vector<cv::Point2d> res;
        cv::transform(src, res, rotation);
        return res;
    }

    /**
     * @brief Compute angle from given number of frame.
     * 
     * @param frame_no Number of frame used to determine angle of rotation.
     */
    double get_angle() const {
        return 0;
        // std::size_t frames = move_analyzer.vault_frames(frame_no);
        // direction dir = move_analyzer.get_direction();
        // double d = 0;
        // if (dir == direction::left)
        //     d = 1;
        // else if (dir == direction::right)
        //     d = -1;
        // return d * 180.0 * std::min(1.0, (double)frames / (double)vault_frames);
    }

    /**
     * @brief Rotate frame so that person's angle of rotation is as low as possible.
     * 
     * @param frame Frame to be rotated.
     * @param frame_no Number of frame to be rotated.
     * @returns rotated frame.
     */
    cv::Mat rotate(const cv::Mat &frame) const {
        double angle = get_angle();
        cv::Mat rotation = cv::getRotationMatrix2D(get_center(frame), angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(frame, rotated, rotation, frame.size());
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
        // cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
        // cv::Point diag = center - rect.tl();
        // cv::Point tl = center - scale_factor * diag;
        // cv::Point br = center + scale_factor * diag;

        // if (tl.x >= 0 && tl.y >= 0 && br.x <= frame.cols && br.y <= frame.rows) {
        //     // Make sure it fits inside frame.
        //     return cv::Rect(tl, br);
        // } else {
        //     scaling_performed = false;
        //     return rect;
        // }
        return rect;
    }
};