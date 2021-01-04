#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <vector>
#include <algorithm>

class vertical_movement_analyzer {

    double delta = 0;
    std::vector<double> offsets;
    cv::Rect2d background;
    cv::Ptr<cv::Tracker> tracker;

    bool _vault_began = false;
    const std::size_t vault_check_frames = 6;
    const double vault_threshold = -0.55 / 720;

    double direction = -1;

    // Check if `background` is inside frame, create new one if neccessary.
    void update_position_rect(cv::Mat &frame, cv::Rect2d person) {
        if ((background.x <= 0) || (background.y <= 0) ||
            (background.x + background.width >= (double)frame.cols) ||
            (background.y + background.height >= (double)frame.rows)) {
                double new_x = person.x + direction * person.width;
                // Make sure `background` fits inside frame.
                background = cv::Rect2d(
                    std::max(new_x, 0.0), person.y,
                    std::min(person.width, (double)frame.cols - new_x), person.height
                );
                
                tracker = cv::TrackerCSRT::create();
                tracker->init(frame, background);

                if (offsets.size())
                    delta = offsets.back();
        }
    }

    // Calculate verical offset of initial `background`.
    double get_offset(cv::Rect2d person) {
        return delta + (person.y + person.height / 2) - (background.y + background.height / 2);
    }

    // Check if vault is beginning.
    void check_vault_beginning(double height, cv::Rect2d person) {
        if ((!_vault_began) && (offsets.size() > vault_check_frames)) {
            double size = (double)person.height / height;
            double runup_mean_delta = count_mean_delta(offsets.begin(), offsets.end() - vault_check_frames);
            double vault_mean_delta = count_mean_delta(offsets.end() - vault_check_frames, offsets.end());

            if ((vault_mean_delta - runup_mean_delta) * size / height < vault_threshold)
                _vault_began = true;

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
    }

public:

    vertical_movement_analyzer(double dir) : direction(dir) {}

    void change_direction(double dir) {
        direction = dir;
    }

    bool update(cv::Mat &frame, cv::Rect2d person) {
        update_position_rect(frame, person);
        if (tracker->update(frame, background)) {
            offsets.push_back(get_offset(person));
            check_vault_beginning((double)frame.rows, person);
            return true;
        }
        return false;
    }

    bool vault_began() const {
        return _vault_began;
    }

    void draw(cv::Mat &frame) const {
        // cv::rectangle(frame, background.tl(), background.br(), cv::Scalar(255, 0, 0), 2);
    }

};