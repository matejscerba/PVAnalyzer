#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

class vault_body_detector {

    const double vault_duration = 0.8;

    std::size_t fps;
    std::size_t current_frame = 0;
    int dir;

    cv::Mat rotation;
    cv::Mat rotation_back;
    double alpha;

    bool scaling_performed = true;

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

    std::vector<cv::Point2d> update(cv::Mat &frame, cv::Ptr<cv::Tracker> &tracker, bool &res, cv::Mat &person_frame, double scale_factor) {
        cv::Rect2d person;
        cv::Point2f center(frame.cols / 2, frame.rows / 2);
        update_rotation_mat(center);

        // Rotate frame.
        cv::Mat rotated;
        cv::warpAffine(frame, rotated, rotation, frame.size());

        // Track athlete.
        if (tracker->update(rotated, person)) {
            cv::Rect2d scaled = scale(person, frame, scale_factor);
            person_frame = rotated(scaled).clone();
            res = true;
        }

        current_frame++;
        
        return transform_back(person);
    }

    bool was_scaling_performed() const {
        return scaling_performed;
    }

    // Scale rectange `rect`'s size by `scale_factor`, keep center on the same position.
    cv::Rect2d scale(cv::Rect2d &rect, cv::Mat &frame, double scale_factor) {
        cv::Point2d center(rect.x + rect.width / 2, rect.y + rect.height / 2);
        cv::Point2d diag = center - rect.tl();
        cv::Point2d tl = center - scale_factor * diag;
        cv::Point2d br = center + scale_factor * diag;

        if (tl.x >= 0 && tl.y >= 0 && br.x <= frame.cols && br.y <= frame.rows) {
            // Make sure it fits inside frame.
            scaling_performed = true; // Remember if scaling was performed.
            return cv::Rect2d(tl, br);
        } else {
            scaling_performed = false;
            return rect;
        }
    }

};