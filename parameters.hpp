#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "forward.hpp"

/**
 * @brief Base struct representing vault-related parameter, containing its values and operating with them.
 */
struct parameter {
public:

    /**
     * @brief Part of vault.
     */
    enum class vault_part {
        runup,
        takeoff,
        vault,
        invalid
    };

    /**
     * @brief This parameter's name.
     */
    const std::string name;

    /**
     * @brief Part of vault, for which this parameter is valid.
     */
    const vault_part part;

    /**
     * @brief Compute parameter's values based on given body parts detected in whole video.
     * 
     * @param points Detected body parts to be processed.
     * 
     * @note Saves corresponding values to member propoerty defined in derived structs.
     */
    virtual void compute(const video_body &points) noexcept = 0;

    /**
     * @brief Write value into given stream for given frame number.
     * 
     * @param os Stream in which this parameter's value for given frame number should be written.
     * @param frame_no Number of frame.
     * @param write_unit Specifies whether to write this parameter's unit as well.
     * 
     * @note `frame_no` is typically ignored for not frame-wise parameters.
     * @see frame_wise_parameter
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept = 0;

    /**
     * @brief Returns number of values computed for this parameter.
     * 
     * @returns number of values computed for this parameter.
     */
    virtual std::size_t size() const noexcept = 0;

protected:

    /**
     * @brief This parameter's unit in format to be written immediately after value.
     * 
     * @note If this parameter is computed in meters, then value " m" will be written
     * as e.g. "5.67 m".
     */
    const std::string unit;

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault, for which this parameter is valid.
     * @param unit This parameter's unit.
     * 
     * @note This constructor can be called by derived structs only.
     */
    parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : name(name), part(part), unit(unit) {}

};

//////////////////////////////////////////////////////////////////////////////////////
// single values parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Represents single-valued parameter, e.g. vault height.
 */
struct single_value_parameter : parameter {
public:

    /**
     * @see parameter::size
     */
    virtual std::size_t size() const noexcept {
        return 1;
    }

protected:

    /**
     * @brief Value for this parameter.
     */
    std::optional<double> value;

};

//////////////////////////////////////////////////////////////////////////////////////
// multiple values parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Represents multiple-valued parameter, e.g. length of each step during runup.
 * 
 * Does not represent parameter computed for each frame of video.
 */
struct multiple_values_parameter : parameter {
public:

    /**
     * @see parameter::size
     */
    virtual std::size_t size() const noexcept {
        return values.size();
    }

protected:

    /**
     * @brief Values of this parameter.
     */
    std::vector<std::optional<double>> values;

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault for which this parameter is designated.
     * @param unit String representation of this parameter's unit.
     * 
     * @note This constructor can be called by derived structs only.
     */
    multiple_values_parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : parameter(name, part, std::move(unit)) {}

};

/**
 * @brief Duration of each step in seconds.
 */
struct steps_duration : multiple_values_parameter {
public:

    /**
     * @brief Default constructor
     * 
     * Specifies parameter's name, vault part and unit as well as frame rate of processed video.
     * 
     * @param fps Frame rate of processed video.
     */
    steps_duration(double fps)
        : multiple_values_parameter("Steps duration", vault_part::runup, " s"), fps(fps) {}

    /**
     * @brief Compute duration of each step based on detected body parts in whole video.
     * 
     * Get numbers of frames where the lower ankle leaves locally lowest point and where
     * the higher ankle leaves locally highest point. Keep low points, which are below
     * center of previous low point and high point (the first one is kept every time).
     * Compute duration between subsequent low points based on their frame numbers and
     * frame rate of processed video.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * 
     * @note Ankle must leave local point of interest by more than 1 px (vertically).
     */
    virtual void compute(const video_body &points) noexcept {
        values.clear();
        step_frames.clear();
        std::greater<double> low;
        std::vector<std::size_t> lows = get_frames(points, low);
        std::less<double> high;
        std::vector<std::size_t> highs = get_frames(points, high);
        double center = 0;
        for (std::size_t i = 0; i < std::min(lows.size(), highs.size()); i++) {
            frame_body l_body = points[lows[i]];
            frame_body h_body = points[highs[i]];
            double l = *get_height(l_body[body_part::l_ankle], l_body[body_part::r_ankle], low);
            double h = *get_height(h_body[body_part::l_ankle], h_body[body_part::r_ankle], high);
            if (i > 0) {
                if (high(l, center)) continue; // Low point is above center of previous points.
                if (low(h, center)) continue;  // High point is below center of previous points.
            }
            center = (l + h) / 2.0;
            step_frames.push_back(lows[i]);
        }
        for (std::size_t i = 1; i < step_frames.size(); i++) {
            values.push_back((step_frames[i] - step_frames[i - 1]) / fps);
        }
    }

    /**
     * @brief Write duration of step ending in frame number `frame_no`.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        auto found = std::find(step_frames.begin(), step_frames.end(), frame_no);
        if (found != step_frames.end()) {
            std::ptrdiff_t idx = found - step_frames.begin();
            if (idx) os << *values[idx - 1] << (write_unit ? unit : "");
        }
    }

    /**
     * @brief Get number of frame in which athlete takes off.
     * 
     * @returns number of frame in which last step ends, no value if no step was detected.
     */
    std::optional<std::size_t> get_takeoff() const noexcept {
        if (step_frames.size())
            return step_frames.back();
        return std::nullopt;
    }

private:

    /**
     * @brief Numbers of frames in which the lower ankle leaves locally lowest point.
     */
    std::vector<std::size_t> step_frames;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Get numbers of frames in which ankle specified by `compare` reaches local point of interest.
     * 
     * `compare` returns true if given values are in correct order based on `compare`. Only ankle satisfying
     * `compare` for each frame is taken into consideration, once it leaves point of interest (next ankle in
     * next frame is vertically in the opposit relation than `compare` by more than 1 px) the previous frame
     * number is added to vector that will be returned.
     * 
     * @param points Detected body parts of athlete in the whole video.
     * @param compare Binary comparison function.
     * @returns vector of frame numbers where ankles leave specified local point of interest.
     */
    std::vector<std::size_t> get_frames(const video_body &points, std::function<bool (double, double)> compare) noexcept {
        std::vector<std::size_t> res;
        std::optional<double> last_height = std::nullopt;
        std::size_t index = 0;
        bool correct_diff = false;
        for (const auto &body : points) {
            std::optional<double> height = get_height(body[body_part::l_ankle], body[body_part::r_ankle], compare);
            if (height && last_height) {
                // Current and last value is valid.
                if (correct_diff && !compare(*height, *last_height) && std::abs(*height - *last_height) > 1) {
                    // Value was changing in the right direction, it stopped changing and is changing in the wrong direction.
                    correct_diff = false;
                    res.push_back(index - 1);
                }
                if (compare(*height, *last_height)) {
                    correct_diff = true;
                }
            }
            last_height = height;
            index++;
        }
        return res;
    }

};

//////////////////////////////////////////////////////////////////////////////////////
// frame-wise parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Represents frame-wise parameter, e.g. position of left ankle throughout whole processed video.
 * 
 * Holds this parameter's value for each frame.
 */
struct frame_wise_parameter : multiple_values_parameter {
public:

    /**
     * @brief Write parameter's value for given frame number.
     * 
     * If value is valid, write it to output stream followed by unit if it is supposed to be written.
     * Write nothing if unit is not supposed to be written and value is invalid. 
     * Write question mark followed by unit if unit is supposed to be written and value is invalid.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     * 
     * @note Writes message to standard output if `frame_no` is out of range of values' indices.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        if (frame_no < values.size()) {
            if (values[frame_no]) {
                os << *values[frame_no] << (write_unit ? unit : "");
            } else if (write_unit) {
                os << "?" << unit;
            }
        } else {
            std::cout << "Unable to write parameter \"" << name << "\" for frame number " << frame_no << std::endl;
        }
    }

    /**
     * @brief Compute this parameter's value for each frame and save them into `values`.
     * 
     * @param points Detected body parts to be processed.
     */
    virtual void compute(const video_body &points) noexcept {
        values.clear();
        std::transform(points.begin(), points.end(), std::back_inserter(values),
                       [this](const frame_body &body){ return this->extract_value(body); }
        );
    }

protected:

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault for which this parameter is designated.
     * @param unit String representation of this parameter's unit.
     * 
     * @note This constructor can be called by derived structs only.
     */
    frame_wise_parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : multiple_values_parameter(name, part, std::move(unit)) {}

    /**
     * @brief Extract value from detected body parts in a single frame.
     * 
     * @param body Detected body parts in a singe frame.
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept = 0;

};

/**
 * @brief Parameter specifying y-coordinate of center of athlete's hips in each frame.
 */
struct hips_height : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * Specifies parameter's name, vault part and unit.
     */
    hips_height() noexcept
        : frame_wise_parameter("Hips height", vault_part::runup, " px") {}

private:
    
    /**
     * @brief Compute hips height from given body points in a single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of hips in frame given by body points.
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        // Average of left and right hip's y-coordinate (if possible), otherwise take one hip if possible.
        if (body[body_part::r_hip] && body[body_part::l_hip]) {
            return ((*body[body_part::r_hip] + *body[body_part::l_hip]) / 2).y;
        } else if (body[body_part::r_hip]) {
            return body[body_part::r_hip]->y;
        } else if (body[body_part::l_hip]) {
            return body[body_part::l_hip]->y;
        } else {
            return std::nullopt;
        }
    }

};

/**
 * @brief Parameter specifying y-coordinate of a body part in each frame.
 */
struct body_part_height : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param b_part Which body part to be processed.
     * 
     * Specifies parameter's name, vault part and unit.
     */
    body_part_height(body_part b_part) noexcept
        : frame_wise_parameter(body_part_name(b_part) + " height", vault_part::runup, " px"), b_part(b_part) {}

private:

    /**
     * @brief Which body part should be used.
     */
    body_part b_part;
    
    /**
     * @brief Compute height of `b_part` from given body points in a single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of `b_part` in frame given by body points.
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        if (body[b_part]) return body[b_part]->y;
        return std::nullopt;
    }

};

/**
 * @brief Parameter specifying angle between vertical axis and ray given by body parts in each frame.
 * 
 * Ray is given by two pairs of body parts, so that the pairs can be centered.
 * If line should be given by only two body parts, pass each part twice.
 */
struct vertical_tilt : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param a1, a2 First pair of body parts, their center specifies origin of ray.
     * @param b1, b2 Second pair of body parts, their center specifies a point ray passes through.
     * @param dir Athlete's runup direction.
     */
    vertical_tilt(std::string name, body_part a1, body_part a2, body_part b1, body_part b2, direction dir) noexcept
        : frame_wise_parameter(name, vault_part::runup, "°"), a1_part(a1), a2_part(a2), b1_part(b1), b2_part(b2), dir(dir) {}

protected:
    
    /**
     * @brief Compute angle between ray and vertical axis.
     * 
     * @param body Body from which to extract value.
     * @returns angle between ray specified by body parts and vertical axis.
     * 
     * @note Negative angle represents ray facing in the opposite direction than athlete's runup.
     * @note Angle == 0 if ray faces directly upwards, degrees range is [-180,180].
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        std::optional<cv::Point2d> a = (body[a1_part] + body[a2_part]) / 2.0;
        std::optional<cv::Point2d> b = (body[b1_part] + body[b2_part]) / 2.0;
        if (a && b) {
            double y = a->y - b->y;
            double d = 0;
            if (dir == direction::left)
                d = 1;
            else if (dir == direction::right)
                d = -1;
            double x = d * (a->x - b->x);
            return std::atan(x / y) * 180.0 / M_PI;
        }
        return std::nullopt;
    }

private:

    /**
     * @brief Body parts specifying ray.
     * 
     * \f$\frac{a1\_part+a2\_part}{2}\f$ represents ray's origin, \f$\frac{b1\_part+b2\_part}{2}\f$
     * a point it passes through.
     */
    body_part a1_part, a2_part, b1_part, b2_part;

    /**
     * @brief Direction of athlete's runup.
     */
    direction dir;

};

/**
 * @brief Parameter specifying angle between horizontal axis and ray given by body parts in each frame.
 * 
 * @see vertical_tilt_parameter.
 */
struct horizontal_tilt : vertical_tilt {
public:

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param a1, a2 First pair of body parts, their center specifies origin of ray.
     * @param b1, b2 Second pair of body parts, their center specifies a point ray passes through.
     * 
     * @note Direction passes to vertical_tilt_parameter's constructor is irrelevant.
     */
    horizontal_tilt(std::string name, body_part a1, body_part a2, body_part b1, body_part b2) noexcept
        : vertical_tilt(name, a1, a2, b1, b2, direction::left) {}

private:

    /**
     * @brief Compute angle between ray and horizontal axis using angle between ray and vertical axis.
     * 
     * @param body Body from which to extract value.
     * @returns angle between ray specified by body parts and horizontal axis.
     * 
     * @note Negative angle represents ray facing downwards.
     * @note Angle's degrees range is [-90,90].
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        std::optional<double> vertical_tilt = vertical_tilt::extract_value(body);
        if (vertical_tilt) return 90.0 - std::abs(*vertical_tilt);
        return std::nullopt;
    }

};
