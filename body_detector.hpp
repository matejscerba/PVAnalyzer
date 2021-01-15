#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "movement_analyzer.hpp"
#include "vault_body_detector.hpp"
#include "person.hpp"

class body_detector {

    cv::HOGDescriptor hog;
    cv::dnn::Net net;
    const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";
    cv::Ptr<cv::Tracker> tracker;

    std::size_t current_frame = 0;
    std::size_t person_frame;
    const cv::Point person_position;

    std::vector<person> people;

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
            people.emplace_back(current_frame, r, net);
        }
        return detections.size();
    }

    // Detect person's bounding rectangle in frame.
    bool detect_current(cv::Mat &frame, cv::Mat &person_mat) {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);

        for (auto &d : detections) {
            cv::rectangle(frame, d.tl(), d.br(), cv::Scalar(0, 255, 0), 2);
        }

        // Select valid rectangle.
        if (select_rectangle(detections, frame)) {
            // Valid rectangle selected, initialize tracker.
            tracker->init(frame, people.back().bbox());
            person_mat = frame(people.back().bbox()).clone();
            return true;
        }

        return false;
    }

    // Tracks athlete in current frame.
    bool track_current(cv::Mat &frame, cv::Mat &person_mat) {
        cv::Rect2d bbox = people.back().bbox();

        bool res = false;
        if (move_analyzer.vault_began()) {
            people.emplace_back(current_frame, vb_detector.update(frame, bbox, tracker, person_mat), net);
            res = true;
        } else {
            // Update runup direction.
            vb_detector.update_direction(move_analyzer.get_direction());

            // Update tracker.
            if (tracker->update(frame, bbox)) {
                people.emplace_back(current_frame, bbox, net);
                res = move_analyzer.update(frame, bbox);
                person_mat = frame(bbox).clone();
            }

            draw(frame);
        }
        return res;
    }

    // Draw rectangle `people.back().bbox()` in `frame`.
    void draw(cv::Mat &frame) const {
        cv::Scalar color(0, 0, 255);
        if (move_analyzer.vault_began())
            color = cv::Scalar(0, 255, 0);
        cv::rectangle(frame, people.back().bbox().tl(), people.back().bbox().br(), color, 2);
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
        cv::Mat person_mat;
        if (current_frame < person_frame) {
            current_frame++;
            return skip;
        } else if (current_frame == person_frame) {
            if (!detect_current(frame, person_mat)) {
                std::cout << "detection failed" << std::endl;
                return error;
            }
        } else if (current_frame > person_frame) {
            if (!track_current(frame, person_mat)) {
                std::cout << "tracking failed" << std::endl;
                return error;
            }
        }

        people.back().detect(person_mat);
        people.back().draw(frame);

        // Update frame counter.
        current_frame++;

        // Draw person into frame.
        // draw(frame);

        return ok;
    }

};