#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <list>
#include <optional>
#include <ostream>
#include <vector>

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
    body_detector(double fps) noexcept {
        hog = cv::HOGDescriptor();
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        this->fps = fps;
        // refresh = 0.3 * fps;
        // last_finding = 0;
        athlete.reset();
    }

    /**
     * @brief Find athlete in video given by frames.
     * 
     * Find athlete in `raw_frames`, draw detections to `found_frames`.
     * 
     * @param raw_frames Frames representing video to be processed.
     * @param[out] found_frames Modified frames will be written here.
     * @returns person representing athlete or invalid value if athlete could not be found.
     */
    std::optional<person> find_athlete(const std::vector<cv::Mat> &raw_frames, std::vector<cv::Mat> &found_frames) noexcept {
        found_frames.clear();
        cv::Mat found_frame;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Finding athlete in frame " << frame_no << "/" << raw_frames.size() - 1 << std::endl;

            found_frame = raw_frames[frame_no].clone();

            // Try to find athlete in current frame and draw detections.
            find(found_frame, frame_no);
            draw(found_frame, frame_no);

            // cv::imshow("frame", found_frame);
            // cv::waitKey();

            // Save modified frames.
            found_frames.push_back(found_frame);
        }
        return athlete;
    }

private:

    /**
     * @brief Represents person, can represent someone else than athlete.
     */
    class person_candidate {
    public:

        /**
         * @brief Default constructor.
         * 
         * @param frame_no Frame number in which this person was detected.
         * @param frame Frame in which this person was detected.
         * @param fps Frame rate of processed video.
         * @param bbox Bounding box of person in given frame.
         */
        person_candidate(std::size_t frame_no, const cv::Mat &frame, double fps, const cv::Rect &bbox) noexcept
            : move_analyzer(frame_no, frame, bbox, fps) {
                this->fps = fps;
                first_frame = frame_no;
                tracker = cv::TrackerCSRT::create();
                tracker->init(frame, bbox);
                bboxes.push_back(bbox);
        }

        /**
         * @brief Track person in given frame.
         * 
         * @param frame Frame in which person should be tracked.
         * @param frame_no Number of processed frame.
         * @returns true if tracking was alright and this person can still represent athlete,
         * false otherwise.
         */
        bool track(const cv::Mat &frame, std::size_t frame_no) noexcept {
            cv::Rect bbox;
            if (tracker->update(frame, bbox)) {
                bboxes.push_back(bbox);
                return is_inside(bbox, frame)
                    && is_moving(frame_no)
                    && move_analyzer.update(frame, bbox, frame_no);
            }

            return false;
        }

        /**
         * @brief Get first frame number in which person was detected.
         * 
         * @returns first frame number in which person was detected.
         */
        std::size_t get_first_frame() const noexcept {
            return first_frame;
        }

        /**
         * @brief Get bounding box in first frame in which person was detected.
         * 
         * @returns bounding box in first frame in which person was detected.
         */
        cv::Rect get_first_bbox() const noexcept {
            return bboxes.front();
        }

        // void update_first_bbox(std::size_t frame_no, const cv::Rect &bbox) noexcept {
        //     cv::Point2d diag = *(bbox.br() - get_center(bboxes[frame_no - first_frame]));
        //     double scale = cv::norm(bboxes.front().tl() - bboxes.front().br()) /
        //         cv::norm(bboxes[frame_no - first_frame].tl() - bboxes[frame_no - first_frame].br());
        //     cv::Point2d center = get_center(bboxes.front());
        //     bboxes.front() = cv::Rect(center - scale * diag, center + scale * diag);
        //     std::cout << "updated" << std::endl;
        // }

        cv::Rect bbox(std::size_t frame_no) const noexcept {
            return bboxes[frame_no - first_frame];
        }

        /**
         * @brief Check if this person took off before given frame.
         * 
         * @param frame_no Frame number to be checked.
         * @returns true if person took off before `frame_no`, false otherwise.
         */
        bool vault_began(std::size_t frame_no) const noexcept {
            // return false;
            return move_analyzer.vault_frames(frame_no);
        }

        /**
         * @brief Draw person's bounding box in given frame.
         * 
         * @param frame Frame to be drawn in.
         * @param frame_no Number of given frame.
         */
        void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
            if (bboxes.size() > frame_no - first_frame) {
                cv::Scalar color(0, 0, 255);
                if (move_analyzer.vault_frames(frame_no))
                    color = cv::Scalar(0, 255, 0);
                
                cv::rectangle(frame, bboxes[frame_no - first_frame].tl(), bboxes[frame_no - first_frame].br(), color, 2);
            }
            move_analyzer.draw(frame, frame_no);
        }

    private:

        /**
         * @brief Frame rate of processed video.
         */
        double fps;

        /**
         * @brief Frame number of first frame in which person was detected.
         */
        std::size_t first_frame;

        /**
         * @brief Tracker used to track person in video.
         */
        cv::Ptr<cv::Tracker> tracker;

        /**
         * @brief Bounding boxes of person in frames since first detection.
         */
        std::vector<cv::Rect> bboxes;

        /**
         * @brief Analyzes this person's movement.
         */
        movement_analyzer move_analyzer;

        /**
         * @brief Checks if this person is moving.
         * 
         * @param frame_no Number of frame for which to check if person has moved
         * since first detection.
         * @returns true if person has moved since first detection or if not enough
         * time for movement has passed.
         */
        bool is_moving(std::size_t frame_no) const noexcept {
            if ((double)frame_no - (double)first_frame < fps / 3.0) {
                return true;
            } else {
                return move_analyzer.get_direction() != direction::unknown;
            }
        }

    };

    /**
     * @brief Holds instance, which takes care of detecting people in frame.
     */
    cv::HOGDescriptor hog;

    /**
     * @brief Number of frame, where person is supposed to be detected for the first time.
     */
    std::size_t person_frame;

    /**
     * @brief Point in frame where athlete is expected during first detection.
     */
    cv::Point person_position;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Holds valid person representing athlete if athlete was found in video.
     */
    std::optional<person> athlete;

    /**
     * @brief List of currently valid detected people.
    */
    std::list<person_candidate> people;

    // std::size_t refresh;

    // std::size_t last_finding;

    std::vector<cv::Rect> nms(const std::vector<cv::Rect> &detections) const noexcept {
        std::vector<cv::Rect> res;
        std::vector<cv::Rect> sorted = detections;
        std::sort(sorted.begin(), sorted.end(), [](const auto &r, const auto &s){ return area(r) < area(s); });
        while (sorted.size()) {
            res.push_back(sorted.back());
            sorted.erase(std::remove_if(sorted.begin(), sorted.end(), [&res](const auto &r){
                int top = std::max(res.back().y, r.y);
                int right = std::min(res.back().x + res.back().width, r.x + r.width);
                int bottom = std::min(res.back().y + res.back().height, r.y + r.height);
                int left = std::max(res.back().x, r.x);
                cv::Rect intersection(
                    left, top, right - left, bottom - top
                );
                return area(intersection) / area(r) >= 0.5;
            }));
        }
        return res;
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
        if (!athlete) {
            if (people.size()) {
                people.remove_if([&frame, frame_no](person_candidate &p){ return !p.track(frame, frame_no); });
            }
            // if (!people.size() || frame_no - last_finding >= refresh) {
            if (!people.size()) {
                find_people(frame, frame_no);
                // last_finding = frame_no;
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
     * @param frame Frame in which to draw people.
     * @param frame_no Number of given frame.
     */
    void draw(cv::Mat &frame, std::size_t frame_no) const noexcept {
        for (const auto &p : people)
            p.draw(frame, frame_no);
    }

    /**
     * @brief Detect all people in given frame.
     * 
     * @param frame Frame in which to detect people.
     * @param frame_no Number of processed frame.
     */
    void find_people(const cv::Mat &frame, std::size_t frame_no) noexcept {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(16, 16), 1.05, 2, false);

        detections = nms(detections);

        // std::vector<cv::Rect> new_dets;
        // for (const auto &detection : detections) {
        //     auto closest = std::find_if(people.begin(), people.end(), [frame_no, &detection](const person_candidate &p){ return is_inside(get_center(p.bbox(frame_no)), detection); });
        //     if (closest != people.end()) {
        //         if (is_inside(closest->bbox(frame_no), detection))
        //             closest->update_first_bbox(frame_no, detection);
        //     } else {
        //         new_dets.push_back(detection);
        //     }
        // }
        for (const auto &detection : detections) {
            people.emplace_back(frame_no, frame, fps, detection);
        }

    }

};