#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <list>
#include <optional>

#include "movement_analyzer.hpp"
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

    /**
     * @brief Default constructor.
     * 
     * @param fps Frame rate of processed video.
     */
    body_detector(double fps) {
        hog = cv::HOGDescriptor();
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        this->fps = fps;
        athlete = std::nullopt;
    }

    std::optional<person> find_athlete(const std::vector<cv::Mat> &raw_frames, std::vector<cv::Mat> &found_frames) {
        cv::Mat found_frame;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            found_frame = raw_frames[frame_no].clone();

            // Try to find athlete in current frame and draw detections.
            find(found_frame, frame_no);
            draw(found_frame, frame_no);

            // Save unmodified and modified frames.
            found_frames.push_back(found_frame);
        }
        return athlete;
    }

private:

    class person_candidate {
    public:

        person_candidate(std::size_t frame_no, const cv::Mat &frame, double fps, const cv::Rect &box)
            : move_analyzer(frame_no, frame, box, fps) {
                this->fps = fps;
                first_frame = frame_no;
                tracker = cv::TrackerCSRT::create();
                tracker->init(frame, box);
                bboxes.push_back(box);
        }

        bool track(const cv::Mat &frame, std::size_t frame_no) {
            cv::Rect box;
            // Update tracker.
            if (tracker->update(frame, box)) {
                bboxes.push_back(box);
                return is_inside(get_corners(box), frame)
                    && is_moving(frame_no)
                    && move_analyzer.update(frame, box, frame_no);
            }

            return false;
        }

        std::size_t get_first_frame() const noexcept {
            return first_frame;
        }

        cv::Rect get_first_bbox() const noexcept {
            return bboxes.front();
        }

        bool vault_began(std::size_t frame_no) const noexcept {
            return move_analyzer.vault_frames(frame_no);
        }

        void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
            if (bboxes.size() > frame_no) {
                cv::Scalar color(0, 0, 255);
                if (move_analyzer.vault_frames(frame_no))
                    color = cv::Scalar(0, 255, 0);
                
                cv::rectangle(frame, bboxes[frame_no].tl(), bboxes[frame_no].br(), cv::Scalar(255, 0, 0), 2);
                move_analyzer.draw(frame, frame_no);
            }
        }

    private:

        double fps;

        std::size_t first_frame;

        cv::Ptr<cv::Tracker> tracker;

        std::vector<cv::Rect> bboxes;

        movement_analyzer move_analyzer;

        bool is_moving(std::size_t frame_no) const noexcept {
            if ((double)frame_no - (double)first_frame < fps / 3.0) {
                return true;
            } else {
                return move_analyzer.get_direction() != direction::unknown;
            }
        }

    };

    /// @brief Holds instance, which takes care of detecting people in frame.
    cv::HOGDescriptor hog;

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
    */
    std::list<person_candidate> people;

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
                people.remove_if([&frame, frame_no](person_candidate &p){ return !p.track(frame, frame_no); });
            }
            if (!people.size()) {
                find_people(frame, frame_no);
            }
            auto found = std::find_if(people.begin(), people.end(), [frame_no](const person_candidate &p){
                return p.vault_began(frame_no);
            });
            if (found != people.end()) {
                athlete = person(found->get_first_frame(), found->get_first_bbox(), fps);
                people.clear();
            }
        }
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
    }

    void find_people(const cv::Mat &frame, std::size_t frame_no) noexcept {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(), 1.05, 2, true);
        for (const auto &detection : detections) {
            people.emplace_back(frame_no, frame, fps, detection);
        }
    }

};