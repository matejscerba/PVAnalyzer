#pragma once

#include <opencv2/opencv.hpp>

#include "forward.hpp"

class drawer {
public:

    static void draw(cv::Mat &frame, const model_body &body) noexcept {
        for (int n = 0; n < npairs; n++) {
            std::size_t a_idx = pairs[n][0];
            std::size_t b_idx = pairs[n][1];
            model_point a = body[a_idx];
            model_point b = body[b_idx];

            // Check if points `a` and `b` are valid.
            if (a && b) {
                cv::Point2d ap(a->x, a->z);
                cv::Point2d bp(b->x, b->z);
                cv::Scalar color(0, 255, 255);
                if ((a_idx == body_part::l_ankle) || (a_idx == body_part::l_knee) || (a_idx == body_part::l_hip) ||
                    (a_idx == body_part::l_wrist) || (a_idx == body_part::l_elbow) || (a_idx == body_part::l_shoulder) ||
                    (b_idx == body_part::l_ankle) || (b_idx == body_part::l_knee) || (b_idx == body_part::l_hip) ||
                    (b_idx == body_part::l_wrist) || (b_idx == body_part::l_elbow) || (b_idx == body_part::l_shoulder)) {
                        color = cv::Scalar(255, 0, 255);
                } else if ((a_idx == body_part::r_ankle) || (a_idx == body_part::r_knee) || (a_idx == body_part::r_hip) ||
                    (a_idx == body_part::r_wrist) || (a_idx == body_part::r_elbow) || (a_idx == body_part::r_shoulder) ||
                    (b_idx == body_part::r_ankle) || (b_idx == body_part::r_knee) || (b_idx == body_part::r_hip) ||
                    (b_idx == body_part::r_wrist) || (b_idx == body_part::r_elbow) || (b_idx == body_part::r_shoulder)) {
                        color = cv::Scalar(255, 255, 0);
                }
                // Draw points representing joints and connect them with lines.
                cv::line(frame, ap, bp, color, 2);
                cv::circle(frame, ap, 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(frame, bp, 2, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
};