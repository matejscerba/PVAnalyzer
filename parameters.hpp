#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
        /// Whole vault including runup, takeoff and vault.
        whole,
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
     * @brief Values for this parameter.
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
struct hips_height_parameter : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * Specifies parameter's name, vault part and unit.
     */
    hips_height_parameter() noexcept
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
struct body_part_height_parameter : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param b_part Which body part to be processed.
     * 
     * Specifies parameter's name, vault part and unit.
     */
    body_part_height_parameter(body_part b_part) noexcept
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
struct vertical_tilt_parameter : frame_wise_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param a1, a2 First pair of body parts, their center specifies origin of ray.
     * @param b1, b2 Second pair of body parts, their center specifies a point ray passes through.
     * @param direction Athlete's runup direction.
     */
    vertical_tilt_parameter(std::string name, body_part a1, body_part a2, body_part b1, body_part b2, int direction) noexcept
        : frame_wise_parameter(name, vault_part::runup, "°"), a1_part(a1), a2_part(a2), b1_part(b1), b2_part(b2), dir(direction) {}

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
            double x = (double)dir * (a->x - b->x);
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
    int dir;

};

/**
 * @brief Parameter specifying angle between horizontal axis and ray given by body parts in each frame.
 * 
 * @see vertical_tilt_parameter.
 */
struct horizontal_tilt_parameter : vertical_tilt_parameter {
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
    horizontal_tilt_parameter(std::string name, body_part a1, body_part a2, body_part b1, body_part b2) noexcept
        : vertical_tilt_parameter(name, a1, a2, b1, b2, direction::left) {}

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
        std::optional<double> vertical_tilt = vertical_tilt_parameter::extract_value(body);
        if (vertical_tilt) return 90.0 - std::abs(*vertical_tilt);
        return std::nullopt;
    }

};
