#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

class body_detector {

    cv::dnn::Net net;
    const int max_size = 256;
    const int npoints = 16;
    const int npairs = 14;
    const int pairs[14][2] = {
        {0,1}, {1,2}, {2,3},
        {3,4}, {1,5}, {5,6},
        {6,7}, {1,14}, {14,8}, {8,9},
        {9,10}, {14,11}, {11,12}, {12,13}
    };
    const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

public:

    body_detector() {
        net = cv::dnn::readNet(protofile, caffemodel);
    }

    // Detects athlete in frame.
    void detect(cv::Mat &frame) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size(368, 368), cv::Scalar(0, 0, 0), false, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        draw_frame(frame, output);

        cv::imshow("frame", frame);
        cv::waitKey();
    }

    void draw_frame(cv::Mat &frame, cv::Mat &output) const {
        int h = output.size[2];
        int w = output.size[3];

        std::vector<cv::Point> points(npoints);
        for (int n = 0; n < npoints; n++) {
            cv::Mat probMat(h, w, CV_32F, output.ptr(0, n));

            cv::Point p(-1, -1), max;
            double prob;
            cv::minMaxLoc(probMat, 0, &prob, 0, &max);

            if (prob > 0.1) {
                p = max;
            }

            points[n] = p;
        }
        float sx = (float)frame.cols / w;
        float sy = (float)frame.rows / h;
        for (int n = 0; n < npairs; n++) {
            cv::Point2f a = points[pairs[n][0]];
            cv::Point2f b = points[pairs[n][1]];

            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            a.x *= sx;
            a.y *= sy;
            b.x *= sx;
            b.y *= sy;

            cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

};