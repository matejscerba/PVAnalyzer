#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "movement_analyzer.hpp"

/**
 * @brief Represents person through whole video.
 * 
 * Holds information about a person through whole video (or its part
 * when certain person is in frame). Handles transitions between frames.
 * Handles rotations during vault as well.
*/
class person {

    /// @brief Bounding box corner indices.
    enum corner : int {
        /// Top left.
        tl = 0,
        /// Top right.
        tr,
        /// Bottom left.
        bl,
        /// Bottom right.
        br
    };

    /// @brief Expected vault duration in seconds.
    const double vault_duration = 0.8;

    /// @brief Number of frames during vault.
    const double vault_frames;

    /// @brief Holds information, whether scaling was performed.
    bool scaling_performed = true;

    /**
     * @brief Determines size of bounding box where to detect body parts.
     * 
     * Ratio of size of bounding box used to detect body parts and size of
     * bounding box tracked by `tracker`.
     * 
     * @note Measures size linearly, not bounding box's surface.
     */
    const double scale_factor = 1.8;

    /// @brief Number of body parts, that is being detected.
    const int npoints = 16;

    /// @brief Number of pairs of body parts (joined by line to form a stickman).
    const int npairs = 14;

    /**
     * @brief Body parts pairs specified by indices.
     * 
     * Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5,
     * Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11,
     * Left Knee – 12, Left Ankle – 13, Chest – 14, Background – 15.
     */
    const int pairs[14][2] = {
        {0,1}, {1,2}, {2,3},
        {3,4}, {1,5}, {5,6},
        {6,7}, {1,14}, {14,8}, {8,9},
        {9,10}, {14,11}, {11,12}, {12,13}
    };

    /// @brief Minimal probability value to mark body part as valid.
    const double probThreshold = 0.1;

    /// @brief Deep neural network used for detecting person's body parts.
    cv::dnn::Net net;

    /// @brief Tracker used to track person in frames.
    cv::Ptr<cv::Tracker> tracker;

    /// @brief Number of first frame in which person is detected.
    std::size_t first_frame_no;

    /// @brief Number of currently processed frame.
    std::size_t current_frame_no;

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
    std::vector<std::vector<cv::Point2d>> corners;

    /**
     * @brief Body parts of person in each frame.
     * 
     * Body parts are saved in similar way as corners, after accessing body parts
     * for certain frame, indices correspond to `pairs`.
     */
    std::vector<std::vector<cv::Point2d>> points;

    /// @brief Analyzes this person's movement.
    movement_analyzer move_analyzer;

    /**
     * @brief Extract corners from rectangle.
     * 
     * @param rect Rectangle to extract corners from.
     * @returns vector of points representing corners so that indices
     *     correspond to `enum corner`.
    */
    std::vector<cv::Point2d> get_corners(const cv::Rect &rect) const {
        return {
            rect.tl(), cv::Point2d(rect.br().x, rect.tl().y),
            cv::Point2d(rect.tl().x, rect.br().y), rect.br()
        };
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
    void extract_points(cv::Mat &output, std::size_t frame_no) {
        points.push_back(std::vector<cv::Point2d>(npoints));
        
        int h = output.size[2];
        int w = output.size[3];

        // Scale by `scaling_factor` if last rectangle was scaled.
        double factor = scaling_performed ? scale_factor : 1;
        double sx = factor * width(frame_no) / (double)w;
        double sy = factor * height(frame_no) / (double)h;

        // Get points from output.
        for (int n = 0; n < npoints; n++) {
            cv::Mat probMat(h, w, CV_32F, output.ptr(0, n));

            // Get point in output with maximum probability of "being point `n`".
            cv::Point2d p(-1, -1);
            cv::Point max;
            double prob;
            cv::minMaxLoc(probMat, 0, &prob, 0, &max);

            // Check point probability against a threshold
            if (prob > probThreshold) {
                p = max;
                p.x *= sx; p.y *= sy; // Scale point so it fits original frame.

                // Move point `p` so it is in correct position in frame.
                p = corners[frame_no - first_frame_no][corner::tl]
                    + (1.0 - scale_factor) * 0.5 * (corners[frame_no - first_frame_no][corner::br] - corners[frame_no - first_frame_no][corner::tl])
                    + p.x * (corners[frame_no - first_frame_no][corner::tr] - corners[frame_no - first_frame_no][corner::tl]) / width(frame_no)
                    + p.y * (corners[frame_no - first_frame_no][corner::bl] - corners[frame_no - first_frame_no][corner::tl]) / height(frame_no);
            }

            points[frame_no - first_frame_no][n] = p;
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
                                       std::size_t frame_no,
                                       bool back = false) const {
        double angle = get_angle(frame_no);
        if (back) angle *= -1;
        cv::Mat rotation = cv::getRotationMatrix2D(get_center(frame), angle, 1.0);
        std::vector<cv::Point2d> res;
        cv::transform(src, res, rotation);
        return res;
    }

    /**
     * @brief Crop frame so that it contains its whole body.
     * 
     * Frame must be rotated so that person's bounding box is not rotated and can
     * be cropped from frame.
     * 
     * @param corners Corners of person's unscaled bounding box in unrotated frame.
     * @param frame Frame that is supposed to be cropped.
     * @param frame_no Number of given frame.
     * @returns part of `frame`, that contains whole person.
    */
    cv::Mat get_person_frame(const std::vector<cv::Point2d> &corners, const cv::Mat &frame, std::size_t frame_no) {
        cv::Mat rotated = rotate(frame, frame_no);
        std::vector<cv::Point2d> transformed = transform(corners, frame, frame_no);
        cv::Rect bbox(
            transformed[corner::tl], transformed[corner::br]
        );
        cv::Rect2d scaled = scale(bbox, frame);
        return rotated(scaled).clone();
    }

    /**
     * @brief Compute angle from given number of frame.
     * 
     * @param frame_no Number of frame used to determine angle of rotation.
     */
    double get_angle(std::size_t frame_no) const {
        std::size_t frames = move_analyzer.vault_frames(frame_no);
        return (double)move_analyzer.get_direction() * 180.0 * std::min(1.0, (double)frames / (double)vault_frames);
    }

    /**
     * @brief Computes center of given frame.
     * 
     * @param frame Given frame used to compute its center.
     * @returns point in center of given frame.
     */
    cv::Point get_center(const cv::Mat &frame) const {
        return cv::Point(
            frame.cols / 2, frame.rows / 2
        );
    }

    /**
     * @brief Rotate frame so that person's angle of rotation is as low as possible.
     * 
     * @param frame Frame to be rotated.
     * @param frame_no Number of frame to be rotated.
     * @returns rotated frame.
     */
    cv::Mat rotate(const cv::Mat &frame, std::size_t frame_no) const {
        double angle = get_angle(frame_no);
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
    person(std::size_t frame_no, const cv::Mat &frame, std::size_t fps, const cv::Rect &box, cv::dnn::Net &net)
        : vault_frames((double)fps * vault_duration) {
            first_frame_no = frame_no;
            current_frame_no = frame_no;
            corners.push_back(get_corners(box));
            this->net = net;
            tracker = cv::TrackerCSRT::create();
            tracker->init(frame, box);
    }

    /**
     * @param frame_no Number of frame where to get bounding box.
     * @returns bounding box of this person in current frame.
     */
    cv::Rect bbox(std::size_t frame_no) const {
        // Check whether person exists in given frame.
        if (frame_no < first_frame_no) {
            std::cout << "Unable to extract bounding box from frame no. " << frame_no << std::endl;
            return cv::Rect();
        }
        std::size_t idx = frame_no - first_frame_no;
        return cv::Rect(
            cv::Point(corners[idx][corner::tl]),
            cv::Point(corners[idx][corner::br])
        );
    }

    /// @returns width of this person's bounding box in given frame.
    double width(std::size_t frame_no) const {
        return cv::norm(corners[frame_no - first_frame_no][corner::tr] - corners[frame_no - first_frame_no][corner::tl]);
    }

    /// @returns height of this person's bounding box in given frame.
    double height(std::size_t frame_no) const {
        return cv::norm(corners[frame_no - first_frame_no][corner::bl] - corners[frame_no - first_frame_no][corner::tl]);
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
        // Check whether person can be tracked in given frame
        // (person must exist at least one frame before the currently processed one).
        if (frame_no < first_frame_no + 1) {
            std::cout << "Unable to extract bounding box from frame before frame no. " << frame_no << std::endl;
            return false;
        }

        cv::Rect box;
        cv::Mat rotated = rotate(frame, frame_no);

        // Update tracker.
        if (tracker->update(rotated, box)) {
            // TODO: crop frame and save it for future deep detection.
            // Append person's bounding box corners.
            corners.push_back(transform(get_corners(box), frame, frame_no, true));
            return move_analyzer.update(frame, box, frame_no);
        }

        return false;
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
        // Check whether person exists in given frame.
        if (frame_no < first_frame_no) {
            std::cout << "Unable to detect body parts in frame no. " << frame_no << std::endl;
            return;
        }
        // Crop person from frame.
        cv::Mat input = get_person_frame(corners[frame_no - first_frame_no], frame, frame_no);
        // Process cropped frame.
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
        net.setInput(blob);
        cv::Mat output = net.forward();
        extract_points(output, frame_no);
    }

    /// @brief Returns centers of gravity of person in each frame.
    std::vector<cv::Point2d> get_centers_of_gravity() const {
        std::vector<cv::Point2d> res;
        for (const auto &p : points) {
            // Average of left and right hip (if possible).
            if ((p[8].y > 0) && (p[11].y > 0))
                res.emplace_back((p[8] + p[11]) / 2);
            else if (p[8].y > 0)
                res.emplace_back(p[8]);
            else if (p[11].y > 0)
                res.emplace_back(p[11]);
            else
                res.emplace_back();
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
        // Check whether person exists in given frame.
        if (frame_no < first_frame_no) {
            std::cout << "Unable to draw frame no. " << frame_no << std::endl;
            return;
        }
        std::size_t idx = frame_no - first_frame_no;
        
        if (corners.size() > idx) {
        // Rectangle which is tracked.
            cv::Point2d tl = corners[idx][corner::tl];
            cv::Point2d tr = corners[idx][corner::tr];
            cv::Point2d bl = corners[idx][corner::bl];
            cv::Point2d br = corners[idx][corner::br];

            cv::Scalar color(0, 0, 255);
            if (move_analyzer.vault_began(frame_no))
                color = cv::Scalar(0, 255, 0);

            cv::line(frame, tl, tr, color, 1);
            cv::line(frame, tr, br, color, 1);
            cv::line(frame, br, bl, color, 1);
            cv::line(frame, bl, tl, color, 1);

            // Scaled rectangle.
            tl = corners[idx][corner::tl]
                + (1.0 - scale_factor) * 0.5 * (corners[idx][corner::br] - corners[idx][corner::tl]);
            tr = corners[idx][corner::tr]
                + (1.0 - scale_factor) * 0.5 * (corners[idx][corner::bl] - corners[idx][corner::tr]);
            bl = corners[idx][corner::bl]
                - (1.0 - scale_factor) * 0.5 * (corners[idx][corner::bl] - corners[idx][corner::tr]);
            br = corners[idx][corner::br]
                - (1.0 - scale_factor) * 0.5 * (corners[idx][corner::br] - corners[idx][corner::tl]);

            color = cv::Scalar(0, 0, 127);
            if (move_analyzer.vault_began(frame_no))
                color = cv::Scalar(0, 127, 0);

            cv::line(frame, tl, tr, color, 1);
            cv::line(frame, tr, br, color, 1);
            cv::line(frame, br, bl, color, 1);
            cv::line(frame, bl, tl, color, 1);
        }

        // Body parts if they were detected.
        if (points.size() > idx) {
            for (int n = 0; n < npairs; n++) {
                cv::Point2d a = points[idx][pairs[n][0]];
                cv::Point2d b = points[idx][pairs[n][1]];

                // Check if points `a` and `b` are valid.
                if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                    continue;

                // Draw points representing joints and connect them with lines.
                cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
                cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Movement analyzer.
        move_analyzer.draw(frame);
    }

};