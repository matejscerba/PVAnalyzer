#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <list>
#include <optional>

#include "person.hpp"

/**
 * @brief Detects athlete in each frame of a video.
 * 
 * Detects all people in given frame and tracks athlete in next frames.
 * Athlete is selected based on given position. Tracking in next frames
 * is performed by instance of class person.
*/
class body_detector {
public:

    /// @brief Supported return values for function `detect`.
    enum class result {
        /// Processed frame was skipped.
        skip,
        /// Frame was processed correctly (athlete was detected).
        ok,
        /// An error occured when frame was being processed.
        error,
        /// No information about result is available.
        unknown
    };

    /**
     * @brief Default constructor.
     * 
     * @param fps Frame rate of processed video.
     */
    body_detector(double fps) {
        hog = cv::HOGDescriptor();
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        net = cv::dnn::readNet(protofile, caffemodel);
        this->fps = fps;
        athlete = std::nullopt;
    }

    /**
     * @brief Find athlete in given frame.
     * 
     * If athlete was not found, update people detected in previous frames, remove 
     * invalid people. Detect people in frame if all people were removed, select
     * person, whose vault has begun and mark that person as athlete.
     * 
     * @param frame Frame to be processed.
     * @param frame_no Number of frame to be processed.
     */
    void find(const cv::Mat &frame, std::size_t frame_no) noexcept {
        std::cout << "Finding athlete in frame " << frame_no << std::endl;
        if (!athlete) {
            std::cout << people.size() << " ";
            if (people.size()) {
                people.remove_if([&frame, frame_no](person &p){ return !p.track(frame, frame_no); });
            }
            std::cout << people.size() << " ";
            if (!people.size()) {
                find_people(frame, frame_no);
            }
            std::cout << people.size() << " ";
            auto found = std::find_if(people.begin(), people.end(), [frame_no](const person &p){
                return p.vault_began(frame_no);
            });
            std::cout << people.size() << " ";
            if (found != people.end()) {
                athlete = std::make_optional(*found);
                people.clear();
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief Check whether athlete was found.
     * 
     * @returns true if athlete was found.
     */
    bool is_found() const noexcept {
        return athlete != std::nullopt;
    }

    void setup() noexcept {
        person_frame = athlete->get_first_frame();
        person_position = *get_center(athlete->bbox(person_frame));
    }

    /**
     * @brief Detect athlete in given frame or track it if it was detected earlier.
     * 
     * @param frame Frame, in which athlete should be detected.
     * @param frame_no Number of given frame.
     * @returns result of detection, whether frame was skipped, detected correctly or an error occured.
     */
    result detect(const cv::Mat &frame, std::size_t frame_no) {
        if (frame_no < person_frame) {
            return result::skip;
        } else if (frame_no == person_frame) {
            // Detect person in frame.
            if (detect_current(frame, frame_no)) {
                athlete->detect(frame, frame_no);
            } else {
                std::cout << "detection failed" << std::endl;
                return result::error;
            }
        } else if (frame_no > person_frame) {
            if (athlete->track(frame, frame_no)) {
                athlete->detect(frame, frame_no);
            } else {
                return result::error;
            }
        }

        return result::ok;
    }

    /**
     * @brief Get valid athlete detected in processed video, invalid optional otherwise.
     * 
     * @returns person representing athlete.
     */
    person get_athlete() const {
        return *athlete;
    }

    /**
     * @brief Draw each person in frame.
     * 
     * @param frame Frame where to draw people.
     * @param frame_no Number of given frame.
     */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        for (const auto &p : people)
            p.draw(frame, frame_no);
        if (athlete)
            athlete->draw(frame, frame_no);
    }

private:

    /// @brief Holds instance, which takes care of detecting people in frame.
    cv::HOGDescriptor hog;

    /// @brief Deep neural network used for detecting athlete's body parts.
    cv::dnn::Net net;

    /// @brief Number of frame, where person is supposed to be detected for the first time.
    std::size_t person_frame;

    /// @brief Point in frame where athlete is expected during first detection.
    cv::Point person_position;

    /// @brief Frame rate of processed video.
    double fps;

    std::optional<person> athlete;

    /**
     * @brief List of detected people.
     * 
     * Person is removed from list if it is clear that the person is not the athlete.
     * Only the athlete should be in this list when video is coming to an end.
    */
    std::list<person> people;

    /**
     * @brief Binary comparison function.
     * 
     * Compare rectangles based on distance of their centers to `person_position`.
     * 
     * @param lhs, rhs The rectangles to be compared.
     * @returns true if the first parameter is closer to `person_position` than the second parameter.
     */
    bool distance_compare(const cv::Rect &lhs, const cv::Rect &rhs) {
        cv::Point l = get_center(lhs);
        cv::Point r = get_center(rhs);
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

    void find_people(const cv::Mat &frame, std::size_t frame_no) noexcept {
        std::cout << "a";
        std::vector<cv::Rect> detections;
        std::cout << "a";
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);
        std::cout << "a";
        for (const auto &detection : detections) {
            people.emplace_back(frame_no, frame, fps, detection, net);
        }
        std::cout << "a";
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
            athlete = person(frame_no, frame, fps, bbox, net);
            return true;
        } else {
            return false;
        }
    }

};