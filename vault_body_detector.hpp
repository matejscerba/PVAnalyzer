#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

#include "movement_analyzer.hpp"

class vault_body_detector {

    const double vault_duration = 0.8;

    std::size_t fps;
    std::size_t current_frame = 0;
    int dir;

    cv::Mat rotation;
    cv::Mat rotation_back;
    double alpha;

    std::vector<cv::Rect2d> bboxes;

    // Computes position of person in frame before rotation, returns vector of corner points.
    std::vector<cv::Point2d> transform_back(cv::Rect2d person) {
        std::vector<cv::Point2d> src {
            person.tl(), cv::Point2d(person.br().x, person.tl().y),
            cv::Point2d(person.tl().x, person.br().y), person.br()
        };
        std::vector<cv::Point2d> res;
        cv::transform(src, res, rotation_back);

        return res;
    }

    // Updates rotation matrix used for transformation based on approximate position during vault.
    void update_rotation_mat(cv::Point2f center) {
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

public:

    vault_body_detector(std::size_t fps, int dir) : dir(dir) {
        this->fps = fps;
    }

    void update_direction(int dir) {
        this->dir = dir;
    }

    std::vector<cv::Point2d> update(cv::Mat &frame, cv::Rect2d person, cv::Ptr<cv::Tracker> tracker, cv::Mat &tracked) {
        cv::Rect2d prev = person;
        if (bboxes.size())
            prev = bboxes.back();
        cv::Point2f center(frame.cols / 2, frame.rows / 2);
        update_rotation_mat(center);

        // Rotate frame.
        cv::Mat rotated;
        cv::warpAffine(frame, rotated, rotation, frame.size());

        // Track athlete.
        tracker->update(rotated, prev);
        bboxes.push_back(prev);
        tracked = rotated(prev).clone();

        // Draw rectangle.
        cv::rectangle(rotated, prev.tl(), prev.br(), cv::Scalar(0, 255, 0), 2);
        cv::warpAffine(rotated, frame, rotation_back, frame.size());

        current_frame++;
        
        return transform_back(prev);
    }

};