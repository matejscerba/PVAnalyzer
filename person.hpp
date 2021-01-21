#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "movement_analyzer.hpp"
#include "vault_body_detector.hpp"

class person {

    enum corner : int { tl = 0, tr, bl, br };

    const int npoints = 16;
    const int npairs = 14;
    const int pairs[14][2] = {
        {0,1}, {1,2}, {2,3},
        {3,4}, {1,5}, {5,6},
        {6,7}, {1,14}, {14,8}, {8,9},
        {9,10}, {14,11}, {11,12}, {12,13}
    };
    const double probThreshold = 0.1;
    cv::dnn::Net net;
    cv::Ptr<cv::Tracker> tracker;
    const double scale_factor = 1.8;

    std::size_t first_frame_no;
    std::size_t current_frame_no;
    std::vector<std::vector<cv::Point2d>> corners;
    std::vector<std::vector<cv::Point2d>> points;

    vault_body_detector vb_detector;
    movement_analyzer move_analyzer;

    std::vector<cv::Point2d> get_corners(cv::Rect2d &rect) const {
        return {
            rect.tl(), cv::Point2d(rect.br().x, rect.tl().y),
            cv::Point2d(rect.tl().x, rect.br().y), rect.br()
        };
    }

    // Saves person's body parts positions into `points`.
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

    cv::Mat get_person_frame(cv::Rect2d &box, cv::Mat &frame) {
        return frame(vb_detector.scale(box, frame, scale_factor));
    }

public:

    person(std::size_t frame_no, cv::Mat &frame, std::size_t fps, cv::Rect2d box, cv::dnn::Net &net) :
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

    cv::Rect2d bbox() const {
        return cv::Rect2d(
            cv::Point2d(corners.back()[corner::tl]),
            cv::Point2d(corners.back()[corner::br])
        );
    }

    double width() const {
        return cv::norm(corners.back()[corner::tr] - corners.back()[corner::tl]);
    }

    double height() const {
        return cv::norm(corners.back()[corner::bl] - corners.back()[corner::tl]);
    }

    // Tracks person in `frame`.
    bool track(cv::Mat &frame) {
        current_frame_no++;
        cv::Rect2d box = bbox();
        cv::Mat person_frame;

        bool res = false;
        if (move_analyzer.vault_began()) {
            corners.push_back(vb_detector.update(frame, tracker, res, person_frame, scale_factor));
        } else {
            // Update runup direction.
            vb_detector.update_direction(move_analyzer.get_direction());

            // Update tracker.
            if (tracker->update(frame, box)) {
                corners.push_back(get_corners(box));
                res = move_analyzer.update(frame, box);
                person_frame = get_person_frame(box, frame);
            }
        }

        if (res) detect(person_frame);

        return res;
    }

    // Detects person's points iside `bbox` and saves them in `points`.
    void detect(cv::Mat &bbox) {
        cv::Mat blob = cv::dnn::blobFromImage(bbox, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
        net.setInput(blob);
        cv::Mat output = net.forward();
        extract_points(output);
    }

    // Draws person in image `frame`.
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
    }

};