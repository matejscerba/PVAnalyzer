#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "forward.hpp"

/**
 * @brief Handles body parts detections.
 */
class parts_detector {
public:

    /**
     * @brief Default constructor.
     * 
     * Load net used for detections.
     */
    parts_detector() noexcept {
        net = cv::dnn::readNet(PROTOFILE, CAFFEMODEL);
        last_size = cv::Point2d();
        center_shift = cv::Point2d();
        angle = 0.0;
        _update_angle = true;
    }

    /**
     * @brief Detect body parts in given image.
     * 
     * Rotate frame based on torso tilt in last frame, crop window used
     * for detection and detect body parts in cropped window.
     * 
     * @param frame Image in which body parts detections should be made.
     * @param bbox Athlete's bounding box.
     * @returns detected body parts in `frame`.
     */
    frame_points detect(const cv::Mat &frame, const cv::Rect &bbox, double distance) noexcept {
        cv::Rect2d window_rect = get_window_rect(frame, bbox);
        int best = -1;
        frame_points body;
        for (auto shift : SHIFTS) {
            cv::Mat rotation;
            cv::Mat back_rot;
            cv::Point2d center = get_center(bbox) + center_shift;
            cv::Mat window = crop_window(frame, window_rect, center, shift, rotation, back_rot);

            cv::Mat blob = cv::dnn::blobFromImage(window, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
            net.setInput(blob);
            frame_points detected = extract_points(window, net.forward());
            fit_in_frame(detected, window_rect, back_rot);

            if (count(detected) > best) {
                best = count(detected);
                body = detected;
            }
            if (best == NPOINTS) break;
        }
        update_size(body);
        update_center_shift(bbox, body, distance);
        update_angle(body);
        return body;
    }

    cv::Point rotate(const cv::Size &s) const noexcept {
        std::vector<cv::Point2d> src;
        src.emplace_back(s.width, s.height);
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(), angle, 1.0);
        std::vector<cv::Point2d> res;
        cv::transform(src, res, rot);
        return res.front();
    }

private:

    /**
     * @brief Net used for body parts detections.
     */
    cv::dnn::Net net;

    /**
     * @brief Last valid size of detected body.
     * 
     * Distance between two furthest points of last valid body detection represents
     * both components of this point.
     * 
     * @note Valid detection holds all body parts.
     */
    cv::Point2d last_size;

    /**
     * @brief Position of center of athlete's body parts detection window against athlete's bounding box.
     */
    cv::Point2d center_shift;

    /**
     * @brief Angle of last torso tilt.
     */
    double angle;

    bool _update_angle;

    /**
     * @brief Scale rectangle by given factor and fit it inside frame.
     * 
     * @param rect Given rectangle to be scaled.
     * @param frame Frame in which scaled rectangle must fit.
     * @param factor Scale factor specifying ratio of new vs old rectangle.
     * @returns scaled rectangle, which fits inside frame.
     * 
     * @note If scaled rectangle does not fit inside frame, smaller rectangle
     * which fits inside frame is returned.
    */
    cv::Rect scale(const cv::Rect &rect, const cv::Mat &frame, double factor = BASE_SCALE_FACTOR) const noexcept {
        cv::Point center = get_center(rect);
        cv::Point diag = center - rect.tl();
        cv::Point tl = center - factor * diag;
        cv::Point br = center + factor * diag;
        int top = std::max(0, tl.y);
        int right = std::min(frame.cols, br.x);
        int bottom = std::min(frame.rows, br.y);
        int left = std::max(0, tl.x);
        return cv::Rect(left, top, right - left, bottom - top);
    }

    /**
     * @brief Get position and size of cropped part of frame which will be used for body parts detection.
     * 
     * @param frame Frame in which detections will be made.
     * @param bbox Athlete's bounding box.
     * @returns rectangle specifying window used for body parts detection.
     */
    cv::Rect2d get_window_rect(const cv::Mat &frame, const cv::Rect &bbox) const noexcept {
        cv::Point2d center = get_center(bbox);
        center += center_shift;
        cv::Point2d tl = center - last_size / 2;
        cv::Point2d br = center + last_size / 2;
        cv::Rect rect(tl, br);
        cv::Rect2d window_rect = scale(rect, frame, 1.3);
        if (last_size == cv::Point2d()) {
            window_rect = scale(bbox, frame);
        }
        return window_rect;
    }

    /**
     * @brief Crop part of given frame that will be used for body parts detection.
     * 
     * Rotate frame by correct angle first.
     * 
     * @param frame Frame in which detections will be made.
     * @param window_rect Rectangle specifying window used for body parts detection.
     * @param center Center of rotation.
     * @param shift Shift added to last angle during rotation.
     * @param[out] rot Rotation matrix.
     * @param[out] back_rot Inverse rotation matrix.
     * @returns part of frame in which body parts detections will be made.
     */
    cv::Mat crop_window(const cv::Mat &frame,
                        const cv::Rect2d &window_rect,
                        const cv::Point2d &center,
                        double shift,
                        cv::Mat &rot,
                        cv::Mat &back_rot) const noexcept {

        cv::Mat res;
        double current_angle = angle + shift;
        rot = cv::getRotationMatrix2D(center, current_angle, 1.0);
        back_rot = cv::getRotationMatrix2D(center, - current_angle, 1.0);
        cv::warpAffine(frame, res, rot, frame.size());
        return res(window_rect);
    }

    cv::Mat make_smaller(const cv::Mat &frame) const noexcept {
        double x = std::min(1.0, 150.0 / (double)frame.rows);
        cv::Mat res;
        cv::resize(frame, res, cv::Size(x * frame.cols, x * frame.rows));
        return res;
    }

    /**
     * @brief Extract detected points from output of net.
     * 
     * Extract detections from output of net and scale them so they fit inside
     * image which net got as input.
     * 
     * @param input Image given as input to net.
     * @param output Output of net.
     * @returns detections fitted inside `input`.
     */
    frame_points extract_points(const cv::Mat &input, cv::Mat &&output) noexcept {
        frame_points body(NPOINTS, std::nullopt);
        int h = output.size[2];
        int w = output.size[3];

        double sx = (double)input.cols / (double)w;
        double sy = (double)input.rows / (double)h;

        // Get points from output.
        for (std::size_t i = 0; i < NPOINTS; ++i) {
            cv::Mat probMat(h, w, CV_32F, output.ptr(0, i));

            // Get point in output with maximum probability of "being point `i`".
            frame_point p = std::nullopt;
            cv::Point max;
            double prob;
            cv::minMaxLoc(probMat, 0, &prob, 0, &max);

            // Check point probability against a threshold.
            if (prob > DET_THRESHOLD) {
                p = max;
                p->x *= sx; p->y *= sy; // Scale point so it fits input.
            }

            body[i] = p;
        }
        return body;
    }

    /**
     * @brief Count valid body parts.
     * 
     * @returns number of detected body parts.
     */
    int count(const frame_points &body) const noexcept {
        int res = 0;
        for (const auto &part : body) {
            if (part) ++res;
        }
        return res;
    }

    /**
     * @brief Move body parts so they fit inside frame properly.
     * 
     * @param body Detected body parts inside window used for detection.
     * @param window_rect Rectangle specifying window used for body parts detection.
     * @param back_rot Inverse rotation matrix.
     */
    void fit_in_frame(frame_points &body, const cv::Rect2d &window_rect, const cv::Mat &back_rot) noexcept {
        for (auto &p : body) {
            if (p) {
                *p += window_rect.tl();
                std::vector<cv::Point2d> res;
                cv::transform(std::vector<cv::Point2d>{*p}, res, back_rot);
                p = res.front();
            }
        }
    }

    /**
     * @brief Update size of last detected body.
     * 
     * @param body Last detected body.
     */
    void update_size(const frame_points &body) noexcept {
        double max = 0.0;
        for (const auto &p : body) {
            for (const auto &q : body) {
                std::optional<double> dist = distance(p, q);
                if (dist && *dist > max) max = *dist;
            }
        }
        if (count(body) == NPOINTS && max > last_size.x) {
            last_size = cv::Point2d(max, max);
        }
        // if (count(body) == NPOINTS) {
        //     if (last_size.x != 0) {
        //         last_size *= 2;
        //         last_size += cv::Point2d(max, max);
        //         last_size /= 3;
        //     } else {
        //         last_size = cv::Point2d(max, max);
        //     }
        // }
    }

    /**
     * @brief Update position of window used for body parts detection against athlete's bounding box center.
     * 
     * Window used for body parts detection should have its center between athlete's hips.
     * 
     * @param bbox Athlete's bounding box.
     * @param body Detected body parts.
     */
    void update_center_shift(const cv::Rect &bbox, const frame_points &body, double distance) noexcept {
        if (bbox != cv::Rect()) {
            cv::Point2d center = get_center(bbox);
            if ((body[body_part::l_hip] || body[body_part::r_hip])) {
                center_shift = (*body[body_part::l_hip] + *body[body_part::r_hip]) / 2 - center;
                _update_angle = true;
                if (std::abs(center_shift.x) > distance) {
                    center_shift *= distance / std::abs(center_shift.x);
                    center_shift = cv::Point2d(- center_shift.x, center_shift.y);
                    _update_angle = false;
                }
                if (std::abs(center_shift.y) > distance) {
                    center_shift *= distance / std::abs(center_shift.y);
                    center_shift = cv::Point2d(center_shift.x, - center_shift.y);
                    _update_angle = false;
                }
            }
        }
    }

    /**
     * @brief Compute angle based on athlete's last torso tilt.
     * 
     * @returns vertical angle of last torso tilt, `last_angle` otherwise.
     */
    void update_angle(const frame_points &body) noexcept {
        std::optional<double> current_angle = get_vertical_tilt_angle(
            (body[body_part::l_hip] + body[body_part::r_hip]) / 2.0,
            body[body_part::head]
        );
        if (_update_angle && current_angle) angle = *current_angle;
    }

};