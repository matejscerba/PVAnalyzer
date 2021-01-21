#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <list>

#include "person.hpp"

class body_detector {

    cv::HOGDescriptor hog;
    cv::dnn::Net net;
    const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

    std::size_t current_frame = 0;
    std::size_t person_frame;
    std::size_t fps;
    const cv::Point person_position;

    std::list<person> people;

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
    bool select_rectangle(std::vector<cv::Rect> &detections, cv::Mat &frame) {
        if (detections.size()) {
            cv::Rect r = *std::min_element(
                detections.begin(), detections.end(),
                [this](const cv::Rect &a, const cv::Rect &b) { return distance_compare(a, b); }
            );
            people.push_back(person(current_frame, frame, fps, r, net));
        }
        return detections.size();
    }

    // Detect person's bounding rectangle in frame.
    bool detect_current(cv::Mat &frame) {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);

        // for (auto &d : detections) {
        //     cv::rectangle(frame, d.tl(), d.br(), cv::Scalar(0, 255, 0), 2);
        // }

        // Select valid rectangle.
        return select_rectangle(detections, frame);
    }

public:

    enum result { skip, ok, error };

    body_detector(std::size_t frame, const cv::Point &position, std::size_t fps) :
        hog(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9),
        person_position(position) {
            person_frame = frame;
            hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
            net = cv::dnn::readNet(protofile, caffemodel);
            this->fps = fps;
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
        } else {
            // Try to track every person in frame, if it fails, remove such person from `people`.
            people.remove_if([&frame](person &p){ return !p.track(frame); });
            if (people.empty()) return error;
        }

        for (auto &p : people) {
            p.draw(frame);
        }

        // Update frame counter.
        current_frame++;

        return ok;
    }

};