#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

class body_detector {

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
    const float probThreshold = 0.1;

    cv::HOGDescriptor hog;
    cv::dnn::Net net;
    bool validPerson = false;
    cv::Rect lastPerson;

    // Updates `lastPerson` rectangle if possible and returns it.
    cv::Rect selectRectangle(const std::vector<cv::Rect> &detections) {
        if (detections.size()) {
            validPerson = true;
            lastPerson = detections[0];
        }
        // Returns last rectangle where person was.
        return lastPerson;
    }

public:

    body_detector() {
        hog = cv::HOGDescriptor(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
        hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
        net = cv::dnn::readNet(protofile, caffemodel);
    }

    // Detects athlete in frame.
    void detect(cv::Mat &frame) {

        // Detect person rectangle in frame.
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(8, 8), cv::Size(), 1.05, 2, true);

        // Select valid rectangle.
        cv::Rect person = selectRectangle(detections);

        if (!validPerson) return;
        
        // Get person's points.
        cv::Mat personFrame(frame, person);
        cv::Mat blob = cv::dnn::blobFromImage(personFrame, 1.0 / 255, cv::Size(368, 368), cv::Scalar(0, 0, 0), false, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        // Draw person into frame.
        drawFrame(frame, person, output);

        cv::imshow("frame", frame);
        cv::waitKey();
    }

    // Draw person in image `frame` based on `output`, it should be in rectangle `rect`.
    void drawFrame(cv::Mat &frame, const cv::Rect &rect, cv::Mat &output) const {
        int h = output.size[2];
        int w = output.size[3];

        // Draw rectangle where person is supposed to be.
        cv::rectangle(frame, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 2);

        std::vector<cv::Point> points(npoints);
        
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
            }

            points[n] = p;
        }

        // Scale factors.
        float sx = (float)rect.width / w;
        float sy = (float)rect.height / h;

        // Draw pairs of points and connect them with lines.
        for (int n = 0; n < npairs; n++) {
            cv::Point2f a = points[pairs[n][0]];
            cv::Point2f b = points[pairs[n][1]];

            // Check if points `a` and `b` are valid.
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            // Scale points so they are in correct position.
            a.x *= sx; a.x += rect.x;
            a.y *= sy; a.y += rect.y;
            b.x *= sx; b.x += rect.x;
            b.y *= sy; b.y += rect.y;

            // Draw points representing joints and connect them with lines.
            cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

};