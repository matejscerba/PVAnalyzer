#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iterator>
#include <algorithm>

class body_detector {

    enum direction { up, down, none };

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

    bool valid_person = false;
    bool tracker_isInit = false;
    std::size_t current_frame = 0;
    std::size_t person_frame;
    const cv::Point person_position;
    std::vector<cv::Rect2d> people;

    direction person_direction = none;
    std::vector<double> position_offsets;
    double position_delta = 0;
    cv::Rect2d position_rect;
    cv::Ptr<cv::Tracker> position_tracker;

    bool vault_began = false;
    const std::size_t vault_check_frames = 6;
    const double vault_threshold = -0.55 / 720;
    std::vector<cv::Mat> frames;

    double dir = -1;

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
    void select_rectangle(std::vector<cv::Rect> &detections, const cv::Mat &frame) {
        if (detections.size()) {
            valid_person = true;
            cv::Rect r = *std::min_element(
                detections.begin(), detections.end(),
                [this](const cv::Rect &a, const cv::Rect &b) { return distance_compare(a, b); }
            );
            people.emplace_back(r.x, r.y, r.width, r.height);
            if (r.x + r.width / 2 > frame.cols / 2)
                dir = 1;
        }
    }

    void detect_current(cv::Mat &frame) {
        // Detect person rectangle in frame.
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);

        // Select valid rectangle.
        select_rectangle(detections, frame);

        // Get person's points.
        // cv::Mat personFrame(frame, person);
        // cv::Mat blob = cv::dnn::blobFromImage(personFrame, 1.0 / 255, cv::Size(368, 368), cv::Scalar(0, 0, 0), false, false);
        // net.setInput(blob);
        // cv::Mat output = net.forward();
        // cv::Mat output;
    }

    // Tracks `last_person` object in current frame.
    bool track_current(cv::Mat &frame) {
        cv::Rect2d last = people.back();

        // Initialize tracker.
        if (!tracker_isInit) {
            tracker->init(frame, last);
            tracker_isInit = true;
        }

        update_position_rect(frame, last);

        if (vault_began) {
            // TODO: Process vault differently.
            if (tracker->update(frame, last)) {
                people.push_back(last);
                if (position_tracker->update(frame, position_rect)) {
                    position_offsets.push_back(get_offset(last));
                    check_vault_beginning((double)frame.rows);
                }
                return true;
            }
        } else {
            // Update tracker.
            if (tracker->update(frame, last)) {
                people.push_back(last);
                if (position_tracker->update(frame, position_rect)) {
                    position_offsets.push_back(get_offset(last));
                    check_vault_beginning((double)frame.rows);
                }
                return true;
            }
        }
        return false;
    }

    // Check if `position_rect` is inside frame, create new one if neccessary.
    void update_position_rect(cv::Mat &frame, cv::Rect2d last) {
        if ((position_rect.x <= 0) || (position_rect.y <= 0) ||
            (position_rect.x + position_rect.width >= (double)frame.cols) ||
            (position_rect.y + position_rect.height >= (double)frame.rows)) {
                position_rect = last + cv::Point2d(dir * last.width, 0);
                position_tracker = cv::TrackerCSRT::create();
                position_tracker->init(frame, position_rect);
                if (position_offsets.size())
                    position_delta = position_offsets.back();
        }
    }

    // Calculate verical offset of initial `position_rect`.
    double get_offset(cv::Rect2d person_rect) {
        return position_delta + (person_rect.y + person_rect.height / 2) - (position_rect.y + position_rect.height / 2);
    }

    // Check if vault is beginning.
    void check_vault_beginning(double height) {
        if ((!vault_began) && (position_offsets.size() > vault_check_frames)) {
            double size = people.back().height / height;
            double runup_mean_delta = count_mean_delta(position_offsets.begin(), position_offsets.end() - vault_check_frames);
            double vault_mean_delta = count_mean_delta(position_offsets.end() - vault_check_frames, position_offsets.end());

            if ((vault_mean_delta - runup_mean_delta) * size / height < vault_threshold)
                vault_began = true;

            // std::cout << "size: " << size << std::endl;
            // std::cout << "runup:" << runup_mean_delta << std::endl;
            // std::cout << "vault:" << vault_mean_delta << std::endl << std::endl;
        }
    }

    // Count mean of difference of consecutives values given by iterators.
    double count_mean_delta(std::vector<double>::const_iterator begin, std::vector<double>::const_iterator end) {
        double sum = *(--end) - *begin;
        std::ptrdiff_t n = end - begin;
        if (n)
            return sum / n;
        return 0;
        // TODO: Check...
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
        if (vault_began)
            color = cv::Scalar(0, 255, 0);
        cv::rectangle(frame, people.back().tl(), people.back().br(), color, 2);
        cv::rectangle(frame, position_rect.tl(), position_rect.br(), cv::Scalar(255, 0, 0), 2);
    }

public:

    body_detector(std::size_t frame, const cv::Point &position) :
        hog(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9), person_position(position) {
            person_frame = frame;
            hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
            net = cv::dnn::readNet(protofile, caffemodel);
            tracker = cv::TrackerCSRT::create();
    }

    // Detects athlete in frame.
    void detect(cv::Mat &frame) {
        if (current_frame == person_frame) {
            detect_current(frame);
        }
        else if (current_frame > person_frame)
            if (!track_current(frame))
                std::cout << "tracking failed" << std::endl;

        // Update frame counter.
        current_frame++;

        // Check if `last_person` is correctly assigned.
        if (!valid_person) return;

        // TODO: detect body parts.

        // Draw person into frame.
        draw(frame);
        // TODO: Wrap person's info into some struct/class.

        frames.push_back(frame.clone());
        // Display current person in frame.
        cv::imshow("frame", frame);
        cv::waitKey();
    }

    void write(const std::string &&filename) {
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('D','I','V','X'), 30, cv::Size(frames.back().cols, frames.back().rows));
        for (auto &f : frames)
            writer.write(f);
        writer.release();
    }

};