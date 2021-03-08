#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <list>
#include <fstream>

#include "person.hpp"

/**
 * @brief Detects athlete in each frame of a video.
 * 
 * Detects all people in given frame and tracks athlete in next frames.
 * Athlete is selected based on given position. Tracking in next frames
 * is performed by instance of class person.
*/
class body_detector {

    /// @brief Holds instance, which takes care of detecting people in frame.
    cv::HOGDescriptor hog;

    /// @brief Deep neural network used for detecting athlete's body parts.
    cv::dnn::Net net;

    /// @brief Path to protofile to be used for deep neural network intialization.
    const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

    /// @brief Path to caffe model to be used for deep neural network intialization.
    const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

    /// @brief Number of frame, where person is supposed to be detected for the first time.
    std::size_t person_frame;

    /// @brief Point in frame where athlete is expected during first detection.
    const cv::Point person_position;

    /// @brief Frame rate of processed video.
    std::size_t fps;

    /**
     * @brief List of detected people.
     * 
     * Person is removed from list if it is clear that the person is not the athlete.
     * Only the athlete should be in this list when video is coming to an end.
    */
    std::list<person> people; // Only athlete is currently added to this list.

    /**
     * @brief Binary comparison function.
     * 
     * Compare rectangles based on distance of their centers to `person_position`.
     * 
     * @param lhs, rhs The rectangles to be compared.
     * @returns true if the first parameter is closer to `person_position` than the second parameter.
     */
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

    /**
     * @brief Select rectangle closest to `person_position`.
     * 
     * @param detections Vector of rectangles, from which to select athlete's bounding box.
     * @param[out] bbox Rectangle, that holds athlete's bounding box.
     * @returns true if `detections` is not empty (athlete's bounding box was assigned).
     * @note Distance from `person_position` is measured from rectangle's center.
    */
    bool select_rectangle(const std::vector<cv::Rect> &detections, cv::Rect &bbox) {
        if (detections.size()) {
            bbox = *std::min_element(
                detections.begin(), detections.end(),
                [this](const cv::Rect &a, const cv::Rect &b) { return distance_compare(a, b); }
            );
        }
        return detections.size();
    }

    /**
     * @brief Detect bounding box of athlete in frame.
     * 
     * Detects bounding boxes of all people in frame, selects
     * only athlete's bounding box and creates instance of class
     * person representing athlete.
     * 
     * @param frame Frame in which athlete should be detected.
     * @param frame_no Number of given frame.
     * @returns false if no person in frame was found, true otherwise.
    */
    bool detect_current(const cv::Mat &frame, std::size_t frame_no) {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);

        cv::Rect bbox;
        if (select_rectangle(detections, bbox)) {
            people.push_back(person(frame_no, frame, fps, bbox, net));
            return true;
        } else {
            return false;
        }
    }

    void write_params(const person &p) const {
        std::vector<cv::Point2d> cogs = p.get_centers_of_gravity();
        std::ofstream file;
        file.open("cogs.csv");
        for (const auto &cog : cogs) {
            if (cog.y != 0)
                file << cog.y << std::endl;
            else
                file << std::endl;
        }
        file.close();
    }

public:

    /// @brief Supported return values for function `detect`.
    enum class result {
        /// Processed frame was skipped.
        skip,
        /// Frame was processed correctly (athlete was detected).
        ok,
        /// An error occured when frame was being processed.
        error
    };

    /**
     * @brief Default constructor.
     * 
     * @param frame Number of frame when detection should begin, frames before are supposed
     *     to be skipped, computed from 0.
     * @param position Point in frame (specified by first parameter), where athlete is expected.
     * @param fps Frame rate of processed video.
     */
    body_detector(std::size_t frame, const cv::Point &position, std::size_t fps) :
        hog(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9),
        person_position(position) {
            person_frame = frame;
            hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
            net = cv::dnn::readNet(protofile, caffemodel);
            this->fps = fps;
    }

    /**
     * @brief Detect athlete in given frame or track it if it was detected earlier.
     * 
     * @param frame Frame, in which athlete should be detected.
     * @param frame_no Number of given frame.
     * @returns result of detection, whether frame was skipped, detected correctly or an error occured.
     */
    result detect(const cv::Mat &frame, std::size_t frame_no) {
        result res = result::ok;
        if (frame_no < person_frame) {
            res = result::skip;
        } else if (frame_no == person_frame) {
            // Detect person in frame.
            if (!detect_current(frame, frame_no)) {
                std::cout << "detection failed" << std::endl;
                res = result::error;
            }
        } else if (frame_no > person_frame) {
            // Try to track every person in frame, if it fails, remove such person from `people`.
            // people.remove_if([&frame, frame_no](person &p){ return !p.track(frame, frame_no); });
            // if (people.empty()) res = result::error;
            if (!people.front().track(frame, frame_no)) {
                res = result::error;
                write_params(people.front());
            }
        }

        // Detect body parts of all people in frame.
        if (frame_no >= person_frame) {
            for (auto &p : people) {
                p.detect(frame, frame_no);
            }
        }

        return res;
    }

    /**
     * @brief Draw each person in frame.
     * 
     * @param frame Frame where to draw people.
     * @param frame_no Number of given frame.
     */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        for (auto &p : people) {
            p.draw(frame, frame_no);
        }
    }

};