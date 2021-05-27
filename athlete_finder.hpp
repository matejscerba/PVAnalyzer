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
 * @brief Detects athlete in video.
*/
class athlete_finder {
public:

    /**
     * @brief Default constructor.
     * 
     * @param fps Frame rate of processed video.
     */
    athlete_finder(double fps) {
        hog = cv::HOGDescriptor();
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        this->fps = fps;
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
    std::optional<person> find_athlete(const std::vector<cv::Mat> &raw_frames, std::vector<cv::Mat> &found_frames) {
        found_frames.clear();
        cv::Mat found_frame;
        for (std::size_t frame_no = 0; frame_no < raw_frames.size(); ++frame_no) {
            std::cout << "Finding athlete in frame " << frame_no + 1 << "/" << raw_frames.size() << std::endl;

            found_frame = raw_frames[frame_no].clone();

            // Try to find athlete in current frame and draw detections.
            find(found_frame, frame_no);
            draw(found_frame, frame_no);

            // Save modified frames.
            found_frames.push_back(found_frame);
        }
        return athlete;
    }

    /**
     * @brief Let user select athlete's bounding box in video.
     * 
     * @param frames Frames of video to select athlete from.
     * @param fps Frame rate of processed video.
     * @returns athlete if user selected valid bounding box.
     */
    std::optional<person> select_athlete(const std::vector<cv::Mat> &frames, double fps) const {
        cv::Rect bbox;
        std::optional<person> athlete = std::nullopt;
        for (std::size_t frame_no = 0; frame_no < frames.size(); ++frame_no) {
            std::cout << "Skip to next frame by pressing the c button!" << std::endl;
            bbox = cv::selectROI("Select athlete", frames[frame_no], false, false);
            if (bbox.width > 0 && bbox.height > 0) {
                athlete = person(frame_no, bbox, fps);
                break;
            }
        }
        cv::waitKey(1);
        cv::destroyAllWindows();
        cv::waitKey(1);
        return athlete;
    }

private:

    /**
     * @brief Represents person, can represent someone else than athlete.
     */
    class athlete_candidate {
    public:

        /**
         * @brief Default constructor.
         * 
         * @param frame_no Frame number in which this person was detected.
         * @param frame Frame in which this person was detected.
         * @param fps Frame rate of processed video.
         * @param bbox Bounding box of person in given frame.
         */
        athlete_candidate(std::size_t frame_no, const cv::Mat &frame, double fps, const cv::Rect &bbox) noexcept
            : move_analyzer(frame_no, frame, std::vector<cv::Rect>{ bbox }, fps) {
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
        bool track(const cv::Mat &frame, std::size_t frame_no) {
            cv::Rect bbox;
            if (tracker->update(frame, bbox)) {
                bboxes.push_back(bbox);
                return is_inside(bbox, frame)
                    && is_moving(frame_no)
                    && move_analyzer.update(frame, std::vector<cv::Rect>{ bbox }, frame_no);
            }

            return false;
        }

        /**
         * @brief Get first frame number in which person was detected.
         * 
         * @returns first frame number in which person was detected.
         */
        std::size_t get_first_frame() const {
            return first_frame;
        }

        /**
         * @brief Get bounding box in first frame in which person was detected.
         * 
         * @returns bounding box in first frame in which person was detected.
         */
        cv::Rect get_first_bbox() const {
            return bboxes.front();
        }

        /**
         * @brief Check if this person took off before given frame.
         * 
         * @param frame_no Frame number to be checked.
         * @returns true if person took off before `frame_no`, false otherwise.
         */
        bool is_athlete(std::size_t frame_no) const {
            return move_analyzer.get_direction() != direction::unknown;
        }

        /**
         * @brief Draw person's bounding box in given frame.
         * 
         * @param frame Frame to be drawn in.
         * @param frame_no Number of given frame.
         */
        void draw(cv::Mat &frame, std::size_t frame_no) const {
            if (bboxes.size() > frame_no - first_frame) {
                cv::rectangle(frame, bboxes[frame_no - first_frame].tl(), bboxes[frame_no - first_frame].br(), cv::Scalar(0, 0, 255), 2);
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
         * @brief Check if enough time passed since first detection.
         * 
         * @param frame_no Number of currently processed frame.
         * @returns true if enough time has already passed.
         */
        bool time_exceeded(std::size_t frame_no) const {
            return (double)frame_no - (double)first_frame >= PERSON_CHECK_TIME * fps;
        }

        /**
         * @brief Checks if this person is moving.
         * 
         * @param frame_no Number of frame for which to check if person has moved
         *      since first detection.
         * @returns true if person has moved since first detection or if not enough
         *      time for movement has passed.
         */
        bool is_moving(std::size_t frame_no) const {
            if (time_exceeded(frame_no)) {
                return move_analyzer.get_direction() != direction::unknown;
            } else {
                return true;
            }
        }

    };

    /**
     * @brief Holds instance, which takes care of detecting people in frame.
     */
    cv::HOGDescriptor hog;

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
    std::list<athlete_candidate> people;

    /**
     * Non-maximum suppression.
     * 
     * Filter rectangles based on their intersections so that they intersect as little as
     * possible and hold different objects.
     * 
     * @param rects Rectangles to be filtered.
     * @returns filtered rectangles.
     */
    std::vector<cv::Rect> nms(const std::vector<cv::Rect> &rects) const {
        std::vector<cv::Rect> res;
        std::vector<cv::Rect> sorted = rects;
        std::sort(sorted.begin(), sorted.end(), [](const auto &r, const auto &s){ return area(r) < area(s); });
        while (sorted.size()) {
            res.push_back(sorted.back());
            sorted.erase(std::remove_if(sorted.begin(), sorted.end(), [&res](const auto &r){
                cv::Rect intersection = res.back() & r;
                return area(intersection) / area(r) >= NMS_THRESHOLD;
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
    void find(const cv::Mat &frame, std::size_t frame_no) {
        if (!athlete) {
            people.remove_if([&frame, frame_no](athlete_candidate &p){ return !p.track(frame, frame_no); });
            auto found = std::find_if(people.begin(), people.end(), [frame_no](const athlete_candidate &p){
                return p.is_athlete(frame_no);
            });
            if (found != people.end()) {
                athlete = person(found->get_first_frame(), found->get_first_bbox(), fps);
                people.clear();
                return;
            }
            if (!people.size()) {
                find_people(frame, frame_no);
            }
        }
    }

    /**
     * @brief Draw each person in frame.
     * 
     * @param frame Frame in which to draw people.
     * @param frame_no Number of given frame.
     */
    void draw(cv::Mat &frame, std::size_t frame_no) const {
        for (const auto &p : people)
            p.draw(frame, frame_no);
    }

    /**
     * @brief Detect all people in given frame.
     * 
     * @param frame Frame in which to detect people.
     * @param frame_no Number of processed frame.
     */
    void find_people(const cv::Mat &frame, std::size_t frame_no) {
        std::vector<cv::Rect> detections;
        hog.detectMultiScale(frame, detections, 0, cv::Size(4, 4), cv::Size(16, 16), 1.05, 2, false);

        detections = nms(detections);

        for (const auto &detection : detections) {
            people.emplace_back(frame_no, frame, fps, detection);
        }

    }

};