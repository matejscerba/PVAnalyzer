#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <list>
#include <optional>

#include "person_checker.hpp"
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
            if (people.size()) {
                people.remove_if([&frame, frame_no](person_checker &p){ return !p.track(frame, frame_no); });
            }
            if (!people.size()) {
                find_people(frame, frame_no);
            }
            auto found = std::find_if(people.begin(), people.end(), [frame_no](const person_checker &p){
                return p.vault_began(frame_no);
            });
            if (found != people.end()) {
                athlete = person(found->get_first_frame(), found->get_first_bbox(), fps, net);
                people.clear();
            }
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

    /**
     * @brief Detect athlete in given frame or track it if it was detected earlier.
     * 
     * @param frame Frame, in which athlete should be detected.
     * @param frame_no Number of given frame.
     * @returns result of detection, whether frame was skipped, detected correctly or an error occured.
     */
    result detect(const cv::Mat &frame, std::size_t frame_no) {
        if (athlete->track(frame, frame_no)) {
            athlete->detect(frame, frame_no);
        } else {
            return result::error;
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
    std::list<person_checker> people;

    void find_people(const cv::Mat &frame, std::size_t frame_no) noexcept {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);
        for (const auto &detection : detections) {
            people.emplace_back(frame_no, frame, fps, detection);
        }
    }

};