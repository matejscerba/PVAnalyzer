#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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

    std::size_t frame_no;
    std::vector<cv::Point2d> corners;
    std::vector<cv::Point2d> points;

    // Saves person's body parts positions into `points`.
    void extract_points(cv::Mat &output) {
        points = std::vector<cv::Point2d>(npoints);
        
        int h = output.size[2];
        int w = output.size[3];

        double sx = width() / (double)w;
        double sy = height() / (double)h;

        // Get points from output.
        for (int n = 0; n < npoints; n++) {
            cv::Mat probMat(h, w, CV_32F, output.ptr(0, n));

            // Get point in output with maximum probability of "being point `n`".
            cv::Point p(-1, -1), max;
            double prob;
            cv::minMaxLoc(probMat, 0, &prob, 0, &max);

            // Check point probability against a threshold
            if (prob > probThreshold) {
                p = max;
                p.x *= sx; p.y *= sy; // Scale point so it fits original frame.
            }

            points[n] = p;
        }
    }

    
    void transform(cv::Point2d &p) const {
        p = corners[corner::tl]
            + p.x * (corners[corner::tr] - corners[corner::tl]) / width()
            + p.y * (corners[corner::bl] - corners[corner::tl]) / height();
    }

public:

    person(std::size_t frame_no, cv::Rect2d rect, cv::dnn::Net &net) {
        this->frame_no = frame_no;
        this->corners = {
            rect.tl(), cv::Point2d(rect.br().x, rect.tl().y),
            cv::Point2d(rect.tl().x, rect.br().y), rect.br()
        };
        this->net = net;
    }

    person(std::size_t frame_no, std::vector<cv::Point2d> corners, cv::dnn::Net &net) {
        this->frame_no = frame_no;
        this->corners = corners;
        this->net = net;
    }

    cv::Rect2d bbox() const {
        return cv::Rect2d(
            cv::Point2d(corners[corner::tl]),
            cv::Point2d(corners[corner::br])
        );
    }

    double width() const {
        return cv::norm(corners[corner::tr] - corners[corner::tl]);
    }

    double height() const {
        return cv::norm(corners[corner::bl] - corners[corner::tl]);
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
        for (int n = 0; n < npairs; n++) {
            cv::Point2d a = points[pairs[n][0]];
            cv::Point2d b = points[pairs[n][1]];

            // Check if points `a` and `b` are valid.
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            // Move points `a` and `b` so they are in correct position in `frame`.
            transform(a); transform(b);

            // Draw points representing joints and connect them with lines.
            cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

};