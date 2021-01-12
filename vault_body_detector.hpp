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

    std::vector<cv::Rect2d> people;

    // Computes position of person in frame before rotation.
    cv::Rect2d transform_back(cv::Rect2d person) {
        std::vector<cv::Point2f> src {
            cv::Point2f(person.tl()),
            cv::Point2f(person.br().x, person.tl().y),
            cv::Point2f(person.tl().x, person.br().y),
            cv::Point2f(person.br())
        };
        std::vector<cv::Point2f> res;
        cv::transform(src, res, rotation_back);

        // Make sure person in rotated frame fits inside rectangle which I want to return.
        float x_min = std::min_element(res.begin(), res.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; })->x;
        float y_min = std::min_element(res.begin(), res.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; })->y;
        float x_max = std::max_element(res.begin(), res.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; })->x;
        float y_max = std::max_element(res.begin(), res.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; })->y;
        return cv::Rect2d(x_min, y_min, x_max - x_min, y_max - y_min);
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

    void change_direction(int dir) {
        this->dir = dir;
    }

    cv::Rect2d update(cv::Mat &frame, cv::Rect2d person, cv::Ptr<cv::Tracker> tracker) {
        cv::Rect2d prev = person;
        if (people.size())
            prev = people.back();
        cv::Point2f center(frame.cols / 2, frame.rows / 2);
        update_rotation_mat(center);

        // Rotate frame.
        cv::Mat res;
        cv::warpAffine(frame, res, rotation, frame.size());

        // Track athlete.
        tracker->update(res, prev);
        people.push_back(prev);

        // Draw rectangle.
        cv::rectangle(res, prev.tl(), prev.br(), cv::Scalar(0, 255, 0), 2);
        cv::imshow("rot", res);

        // Compute position before rotation.
        cv::Rect2d transformed = transform_back(prev);
        current_frame++;
        
        return transformed;
    }

};