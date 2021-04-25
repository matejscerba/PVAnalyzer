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

private:

    cv::dnn::Net net;

    cv::Point2d center;

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
        return res;
    }

};