#pragma once

#include <opencv2/opencv.hpp>

#include "forward.hpp"

class parts_detector {
public:

    parts_detector() {
        net = cv::dnn::readNet(protofile, caffemodel);
    }

    frame_body deep_detect(const cv::Mat &frame) noexcept {
        center = get_center(frame);
        // Crop person from frame.
        // Process cropped frame.
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size(), cv::Scalar(), false, false, CV_32F);
        net.setInput(blob);
        cv::Mat output = net.forward();
        return extract_points(frame, output);
    }

    cv::Point2d get_last_center() const noexcept {
        return center;
    }

    cv::Point2d last_body_size() const noexcept {
        return last_size;
    }

private:

    cv::dnn::Net net;

    cv::Point2d center;

    cv::Point2d last_size;

    frame_body extract_points(const cv::Mat &frame, cv::Mat &output) {
        frame_body res(npoints, std::nullopt);
        int h = output.size[2];
        int w = output.size[3];

        double sx = (double)frame.cols / (double)w;
        double sy = (double)frame.rows / (double)h;

        // Get points from output.
        for (int n = 0; n < npoints; ++n) {
            cv::Mat probMat(h, w, CV_32F, output.ptr(0, n));

            // Get point in output with maximum probability of "being point `n`".
            frame_part p = std::nullopt;
            cv::Point max;
            double prob;
            cv::minMaxLoc(probMat, 0, &prob, 0, &max);

            // Check point probability against a threshold
            if (prob > detection_threshold) {
                p = max;
                p->x *= sx; p->y *= sy; // Scale point so it fits original frame.
            }

            res[n] = p;
        }
        bool valid = false;
        double top;
        double right;
        double bottom;
        double left;
        for (const auto &p : res) {
            if (p) {
                if (valid) {
                    top = std::min(top, p->y);
                    right = std::max(right, p->x);
                    bottom = std::max(bottom, p->y);
                    left = std::min(left, p->x);
                } else {
                    top = p->y;
                    right = p->x;
                    bottom = p->y;
                    left = p->x;
                    valid = true;
                }
            }
        }
        if (valid) {
            last_size = cv::Point2d(right - left, bottom - top);
        }
        return res;
    }

};