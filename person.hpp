#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "movement_analyzer.hpp"
#include "vault_body_detector.hpp"

/**
 * @brief Represents person through whole video.
 * 
 * Holds information about a person through whole video (or its part
 * when certain person is in frame). Handles transitions between frames.
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

    /// @brief Number of body parts, that is being detected.
    const int npoints = 16;

    /// @brief Number of pairs of body parts (joined by line to form a stickman).
    const int npairs = 14;

    /// @brief Body parts pairs specified by indices.
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

    /**
     * @brief Determines size of bounding box where to detect body parts.
     * 
     * Ratio of size of bounding box used to detect body parts and size of
     * bounding box tracked by `tracker`.
     * 
     * @note Measures size linearly, not bounding box's surface.
     */
    const double scale_factor = 1.8;

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

    /// @brief Detects this person during vault.
    vault_body_detector vb_detector;

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
    */
    void extract_points(cv::Mat &output) {
        points.push_back(std::vector<cv::Point2d>(npoints));
        
        int h = output.size[2];
        int w = output.size[3];

        // Scale by `scaling_factor` if last rectangle was scaled.
        double factor = vb_detector.was_scaling_performed() ? scale_factor : 1;
        double sx = factor * width() / (double)w;
        double sy = factor * height() / (double)h;

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
                p = corners.back()[corner::tl]
                    + (1.0 - scale_factor) * 0.5 * (corners.back()[corner::br] - corners.back()[corner::tl])
                    + p.x * (corners.back()[corner::tr] - corners.back()[corner::tl]) / width()
                    + p.y * (corners.back()[corner::bl] - corners.back()[corner::tl]) / height();
            }

            points.back()[n] = p;
        }
    }

    /**
     * @brief Crop frame so that it contains its whole body.
     * 
     * @param box Unscaled bounding box of this person.
     * @param frame Frame that is supposed to be cropped.
     * @returns part of `frame`, that contains whole person.
    */
    cv::Mat get_person_frame(const cv::Rect &box, const cv::Mat &frame) {
        return frame(vb_detector.scale(box, frame, scale_factor));
    }

public:

    /**
     * @brief Default constructor.
     * 
     * Detect body parts of this person in `frame` as well.
     * 
     * @param frame_no Number of frame, in which this person was detected for the first time.
     * @param frame Current frame, this person was already detected here.
     * @param fps Frame rate of processed video.
     * @param box Bounding box of this person in `frame`.
     * @param net Deep neural network used for detecting person's body parts.
    */
    person(std::size_t frame_no, const cv::Mat &frame, std::size_t fps, const cv::Rect &box, cv::dnn::Net &net) :
        vb_detector(fps, movement_analyzer::direction::unknown) {
            first_frame_no = frame_no;
            current_frame_no = frame_no;
            corners.push_back(get_corners(box));
            this->net = net;
            tracker = cv::TrackerCSRT::create();
            tracker->init(frame, box);
            cv::Mat person_frame = get_person_frame(box, frame);
            detect(person_frame);
    }

    /// @returns bounding box of this person in current frame.
    cv::Rect bbox() const {
        return cv::Rect(
            cv::Point(corners.back()[corner::tl]),
            cv::Point(corners.back()[corner::br])
        );
    }

    /// @returns width of this person's bounding box in current frame.
    double width() const {
        return cv::norm(corners.back()[corner::tr] - corners.back()[corner::tl]);
    }

    /// @returns height of this person's bounding box in current frame.
    double height() const {
        return cv::norm(corners.back()[corner::bl] - corners.back()[corner::tl]);
    }

    /**
     * @brief Track person in next frame.
     * 
     * Detect body parts of this person in next frame as well.
     * 
     * @param frame Next frame in which person should be tracked.
     * @returns true if detection was OK, false if an error occured.
     * @note If vault has begun, `vb_detector` takes care of person's tracking.
    */
    bool track(const cv::Mat &frame) {
        current_frame_no++;
        cv::Rect box = bbox();
        cv::Mat person_frame; // Scaled person's frame.

        bool res = false;
        if (move_analyzer.vault_began()) {
            // Append person's bounding box corners, assuming vault has began.
            corners.push_back(vb_detector.update(frame, tracker, res, person_frame, scale_factor));
        } else {
            // Update runup direction.
            vb_detector.update_direction(move_analyzer.get_direction());

            // Update tracker.
            if (tracker->update(frame, box)) {
                // Append person's bounding box corners.
                corners.push_back(get_corners(box));
                res = move_analyzer.update(frame, box);
                person_frame = get_person_frame(box, frame);
            }
        }

        // Detect person's body parts.
        if (res) detect(person_frame);

        return res;
    }

    // Detects person's points iside `bbox` and saves them in `points`.
    /**
     * @brief Detect body parts of this person.
     * 
     * Use deep neural network to detect body parts of this person inside `bbox`.
     * 
     * @param input Image in which body parts of this person should be detected.
    */
    void detect(const cv::Mat &input) {
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
        net.setInput(blob);
        cv::Mat output = net.forward();
        extract_points(output);
    }

    // Draws person in image `frame`.
    /**
     * @brief Draw this person in `frame`.
     * 
     * Draw this person's unscaled bounding box, scaled bounidng box
     * and connected body parts forming a "stickman" in `frame`.
     * 
     * @param frame Frame in which this person should be drawn.
    */
    void draw(cv::Mat &frame) const {
        // Rectangle which is tracked.
        cv::Point2d tl = corners.back()[corner::tl];
        cv::Point2d tr = corners.back()[corner::tr];
        cv::Point2d bl = corners.back()[corner::bl];
        cv::Point2d br = corners.back()[corner::br];

        cv::Scalar color(0, 0, 255);
        if (move_analyzer.vault_began())
            color = cv::Scalar(0, 255, 0);

        cv::line(frame, tl, tr, color, 1);
        cv::line(frame, tr, br, color, 1);
        cv::line(frame, br, bl, color, 1);
        cv::line(frame, bl, tl, color, 1);

        // Scaled rectangle.
        tl = corners.back()[corner::tl]
            + (1.0 - scale_factor) * 0.5 * (corners.back()[corner::br] - corners.back()[corner::tl]);
        tr = corners.back()[corner::tr]
            + (1.0 - scale_factor) * 0.5 * (corners.back()[corner::bl] - corners.back()[corner::tr]);
        bl = corners.back()[corner::bl]
            - (1.0 - scale_factor) * 0.5 * (corners.back()[corner::bl] - corners.back()[corner::tr]);
        br = corners.back()[corner::br]
            - (1.0 - scale_factor) * 0.5 * (corners.back()[corner::br] - corners.back()[corner::tl]);

        color = cv::Scalar(0, 0, 127);
        if (move_analyzer.vault_began())
            color = cv::Scalar(0, 127, 0);

        cv::line(frame, tl, tr, color, 1);
        cv::line(frame, tr, br, color, 1);
        cv::line(frame, br, bl, color, 1);
        cv::line(frame, bl, tl, color, 1);

        // Body parts.
        for (int n = 0; n < npairs; n++) {
            cv::Point2d a = points.back()[pairs[n][0]];
            cv::Point2d b = points.back()[pairs[n][1]];

            // Check if points `a` and `b` are valid.
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            // Draw points representing joints and connect them with lines.
            cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
        }

        // Movement analyzer.
        move_analyzer.draw(frame);
    }

};