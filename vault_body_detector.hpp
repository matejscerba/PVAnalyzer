#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

/**
 * @brief Detects athlete in each frame of vault.
 * 
 * Rotates frames so that athlete's angle of rotation is as low as possible.
 * Handles coordinates' transformations.
*/
class vault_body_detector {

    /// @brief Supposed duration of vault in seconds.
    const double vault_duration = 0.8;

    /// @brief Frame rate of processed video.
    std::size_t fps;

    /// @brief Number of currently processed frame from vault beginning.
    std::size_t current_frame = 0;

    /**
     * @brief Direction multiplier.
     * 
     * Direction determines which way the frame should be rotated
     * (-1 = clockwise, +1 = counterclockwise).
     * 
     * @see movement_analyzer::direction
     */
    int dir;

    /**
     * @brief Matrix representing rotation.
     * 
     * Corresponds with `dir` - rotates frame so that athlete's head is above his feet.
     * 
     * @note Holds result of function `cv::getRotationMatrix2D` method.
     */
    cv::Mat rotation;

    /**
     * @brief Matrix representing rotation back.
     * 
     * Rotates frame in opposite way, than `dir` determines, in composition with
     * `rotation` doesn't rotate frame.
     * 
     * @note Holds result of function `cv::getRotationMatrix2D` method.
     */
    cv::Mat rotation_back;

    /// @brief Holds angle of rotation of current frame.
    double alpha;

    /// @brief Holds information, whether scaling was performed.
    bool scaling_performed = true;

    /**
     * @brief Extract corners from rectangle.
     * 
     * @param rect Rectangle to extract corners from.
     * @returns vector of points representing corners so that indices
     *     correspond to `enum corner`.
    */
    std::vector<cv::Point2d> get_corners(const cv::Rect2d &rect) const {
        return {
            rect.tl(), cv::Point2d(rect.br().x, rect.tl().y),
            cv::Point2d(rect.tl().x, rect.br().y), rect.br()
        };
    }

    /**
     * @brief Computes position of given points after transformation.
     * 
     * @param src Vector of given points.
     * @param rotation_mat Matrix defining given rotation.
     * @returns vector of points after rotation specified by `rotation_mat`.
    */
    std::vector<cv::Point2d> transform(const std::vector<cv::Point2d> &src, const cv::Mat &rotation_mat) const {
        std::vector<cv::Point2d> res;
        cv::transform(src, res, rotation_mat);
        return res;
    }

    /**
     * @brief Updates matrix used for rotation.
     * 
     * Maximal angle of rotation is 180 degrees, degree is in linear correspondance with
     * duration spent in vault.
     * 
     * @param center Center of rotation.
    */
    void update_rotation_mat(const cv::Point2f &center) {
        double old = alpha;
        if ((double)current_frame / (double)fps <= vault_duration) {
            alpha = dir * 180.0 * (double)current_frame / (double)fps / vault_duration;
        }
        // Update rotation matrices if processing first frame or if rotation angle changes.
        if (!current_frame || (old != alpha)) {
            rotation = cv::getRotationMatrix2D(center, alpha, 1.0);
            rotation_back = cv::getRotationMatrix2D(center, -alpha, 1.0);
        }
    }

    /**
     * @brief Rotate frame as specified by `rotation` matrix.
     * 
     * @param frame Frame to be rotated.
     * @returns rotated frame.
     */
    cv::Mat rotate(const cv::Mat &frame) const {
        cv::Mat rotated;
        cv::warpAffine(frame, rotated, rotation, frame.size());
        return rotated;
    }

public:

    /**
     * @brief Determines size of bounding box where to detect body parts.
     * 
     * Ratio of size of bounding box used to detect body parts and size of
     * bounding box tracked by `tracker`.
     * 
     * @note Measures size linearly, not bounding box's surface.
     */
    const double scale_factor = 1.8;

    /**
     * @brief Default constructor.
     * 
     * @param fps Frame rate of processed video.
     * @param dir Direction of athlete's runup.
    */
    vault_body_detector(std::size_t fps, int dir) : dir(dir) {
        this->fps = fps;
    }

    /// @brief Updates runup direction.
    void update_direction(int dir) {
        this->dir = dir;
    }

    /**
     * @brief Process next frame.
     * 
     * Updates rotation angle, rotates frame and tracks athlete in given frame.
     * 
     * @param frame Frame to process.
     * @param tracker Tracker used to track athlete.
     * @param[out] res Result of tracking athlete in given frame (true if everything is OK,
     *     false if tracking failed).
     * @returns vector of points representing position of corners of athlete's tracked bounding box
     *     in unrotated frame.
    */
    std::vector<cv::Point2d> update(const cv::Mat &frame, cv::Ptr<cv::Tracker> &tracker, bool &res) {
        cv::Rect person;
        cv::Point2f center(frame.cols / 2, frame.rows / 2);
        update_rotation_mat(center);

        cv::Mat rotated = rotate(frame);

        // Track person.
        res = tracker->update(rotated, person);

        current_frame++;
        
        return transform(get_corners(person), rotation_back);
    }

    /**
     * @brief Crop frame so that it contains its whole body.
     * 
     * Frame must be rotated so that person's bounding box is not rotated and can
     * be cropped from frame.
     * 
     * @param corners Corners of person's unscaled bounding box in unrotated frame.
     * @param frame Frame that is supposed to be cropped.
     * @returns part of `frame`, that contains whole person.
    */
    cv::Mat get_person_frame(const std::vector<cv::Point2d> &corners, const cv::Mat &frame) {
        cv::Mat rotated = rotate(frame);
        std::vector<cv::Point2d> transformed = transform(corners, rotation);
        cv::Rect bbox(
            transformed[0], transformed[3]
        );
        cv::Rect2d scaled = scale(bbox, frame);
        return rotated(scaled).clone();
    }

    /// @brief Returns information if scaling was performed on last frame processed.
    bool was_scaling_performed() const {
        return scaling_performed;
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
        cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
        cv::Point diag = center - rect.tl();
        cv::Point tl = center - scale_factor * diag;
        cv::Point br = center + scale_factor * diag;

        if (tl.x >= 0 && tl.y >= 0 && br.x <= frame.cols && br.y <= frame.rows) {
            // Make sure it fits inside frame.
            scaling_performed = true; // Remember if scaling was performed.
            return cv::Rect(tl, br);
        } else {
            scaling_performed = false;
            return rect;
        }
    }

};