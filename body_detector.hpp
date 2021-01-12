#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "movement_analyzer.hpp"
#include "vault_body_detector.hpp"

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
    cv::Ptr<cv::Tracker> tracker;

    std::size_t current_frame = 0;
    std::size_t person_frame;
    const cv::Point person_position;

    std::vector<cv::Rect2d> people;

    movement_analyzer move_analyzer;
    vault_body_detector vb_detector;

    // Comparator for rectangles by distance from `person_position`.
    bool distance_compare(const cv::Rect &lhs, const cv::Rect &rhs) const {
        cv::Point l(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point r(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        double lhsDist = std::sqrt(
            (l.x - person_position.x) * (l.x - person_position.x) +
            (l.y - person_position.y) * (l.y - person_position.y)
        );
        double rhsDist = std::sqrt(
            (r.x - person_position.x) * (r.x - person_position.x) +
            (r.y - person_position.y) * (r.y - person_position.y)
        );
        return lhsDist < rhsDist;
    }

    // Updates `people` with rectangle from detections closest to `person_position`.
    bool select_rectangle(std::vector<cv::Rect> &detections, const cv::Mat &frame) {
        if (detections.size()) {
            cv::Rect r = *std::min_element(
                detections.begin(), detections.end(),
                [this](const cv::Rect &a, const cv::Rect &b) { return distance_compare(a, b); }
            );
            people.emplace_back(r.x, r.y, r.width, r.height);
        }
        return detections.size();
    }

    // Detect person's bounding rectangle in frame.
    bool detect_current(cv::Mat &frame) {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);

        // Select valid rectangle.
        if (select_rectangle(detections, frame)) {
            // Valid rectangle selected, initialize tracker.
            tracker->init(frame, people.back());
            return true;
        }

        return false;
    }

    // Tracks athlete in current frame.
    bool track_current(cv::Mat &frame) {
        cv::Rect2d person = people.back();

        if (move_analyzer.vault_began()) {
            people.push_back(vb_detector.update(frame, person, tracker));
            return true;
        } else {
            // Update runup direction.
            vb_detector.change_direction(move_analyzer.get_direction());

            // Update tracker.
            if (tracker->update(frame, person)) {
                people.push_back(person);
                return move_analyzer.update(frame, person);
            }
        }
        return false;
    }

    // Draw person in image `frame` based on `output`, it is in rectangle `people.back()`.
    void draw(cv::Mat &frame, cv::Mat &output) const {
        int h = output.size[2];
        int w = output.size[3];

        cv::Rect last_person = people.back();

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
        float sx = (float)last_person.width / w;
        float sy = (float)last_person.height / h;

        // Draw pairs of points and connect them with lines.
        for (int n = 0; n < npairs; n++) {
            cv::Point2f a = points[pairs[n][0]];
            cv::Point2f b = points[pairs[n][1]];

            // Check if points `a` and `b` are valid.
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;

            // Scale points so they are in correct position.
            a.x *= sx; a.x += last_person.x;
            a.y *= sy; a.y += last_person.y;
            b.x *= sx; b.x += last_person.x;
            b.y *= sy; b.y += last_person.y;

            // Draw points representing joints and connect them with lines.
            cv::line(frame, a, b, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, a, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, b, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw rectangle `people.back()` in `frame`.
    void draw(cv::Mat &frame) const {
        cv::Scalar color(0, 0, 255);
        if (move_analyzer.vault_began())
            color = cv::Scalar(0, 255, 0);
        cv::rectangle(frame, people.back().tl(), people.back().br(), color, 2);
        move_analyzer.draw(frame);
    }

public:

    enum result { skip, ok, error };

    body_detector(std::size_t frame, const cv::Point &position, std::size_t fps) :
        hog(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9),
        person_position(position), vb_detector(fps, movement_analyzer::direction::unknown) {
            person_frame = frame;
            hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
            net = cv::dnn::readNet(protofile, caffemodel);
            tracker = cv::TrackerCSRT::create();
    }

    // Detects athlete in frame.
    result detect(cv::Mat &frame) {
        if (current_frame < person_frame) {
            current_frame++;
            return skip;
        } else if (current_frame == person_frame) {
            if (!detect_current(frame)) {
                std::cout << "detection failed" << std::endl;
                return error;
            }
        } else if (current_frame > person_frame) {
            if (!track_current(frame)) {
                std::cout << "tracking failed" << std::endl;
                return error;
            }
        }

        // Update frame counter.
        current_frame++;

        // Draw person into frame.
        draw(frame);

        return ok;
    }

};