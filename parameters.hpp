#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "forward.hpp"

/**
 * @brief
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

    parameter() noexcept : part(vault_part::invalid) {}

    virtual ~parameter() noexcept = default;

    virtual void compute(const video_body &points) = 0;

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

    virtual std::size_t size() const noexcept = 0;

protected:

    /**
     * @brief This parameter's unit in format to be written immediately after value.
     */
    const std::string unit;

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault, for which this parameter is valid.
     * @param unit This parameter's unit.
     * 
     * @note This class is abstract, so constructor is protected.
     */
    parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : name(name), part(part), unit(unit) {}

};

//////////////////////////////////////////////////////////////////////////////////////
// single values parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief
 */
struct single_value_parameter : parameter {
public:



private:

    std::optional<double> value;

};

//////////////////////////////////////////////////////////////////////////////////////
// multiple values parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief 
 */
struct multiple_values_parameter : parameter {
public:

    virtual std::size_t size() const noexcept {
        return values.size();
    }

private:

    std::vector<std::optional<double>> values;

};

//////////////////////////////////////////////////////////////////////////////////////
// frame-wise parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief 
 */
struct frame_wise_parameter : parameter {
public:

    /**
     * @brief Write parameter's value for given frame number.
     * 
     * If value is valid, write it to output stream followed by unit if it is supposed to be written.
     * Write nothing if unit is not supposed to be written. 
     * Write question mark followed by unit if unit is supposed to be written.
     * 
     * @param os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     * 
     * @note Writes message to standard output if `frame_no` is out of range of values' indices.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        if (frame_no < values.size()) {
            if (write_unit) {
                if (values[frame_no]) {
                    os << *values[frame_no];
                } else {
                    os << "?";
                }
                os << unit;
            } else if (values[frame_no]) {
                os << *values[frame_no];
            }
        } else {
            std::cout << "Unable to write parameter \"" << name << "\" for frame number " << frame_no << std::endl;
        }
    }

    virtual std::size_t size() const noexcept {
        return values.size();
    }

    virtual void compute(const video_body &points) noexcept {
        values.clear();
        std::transform(points.begin(), points.end(), std::back_inserter(values),
                       [this](const frame_body &body){ return this->extract_value(body); }
        );
    }

protected:

    std::vector<std::optional<double>> values;

    frame_wise_parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : parameter(name, part, std::move(unit)) {}

    /**
     * @brief
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept = 0;

};

/**
 * @brief 
 */
struct hips_height_parameter : frame_wise_parameter {
public:

    hips_height_parameter() noexcept
        : frame_wise_parameter("Hips height", vault_part::runup, " px") {}

private:
    
    /**
     * @brief Compute hips height from given body points in single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of hips in frame given by body points.
     * 
     * @see parameter::extract_value
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        // Average of left and right hip (if possible).
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
 * @brief
 */
struct body_part_height_parameter : frame_wise_parameter {
public:

    body_part_height_parameter(body_part b_part) noexcept
        : frame_wise_parameter(body_part_name(b_part) + " height", vault_part::runup, " px"), b_part(b_part) {}

private:

    body_part b_part;
    
    /**
     * @brief Compute body part's height from given body points in single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of `b_part` in frame given by body points.
     * 
     * @see parameter::extract_value
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        if (body[b_part]) return body[b_part]->y;
        return std::nullopt;
    }

};

/**
 * @brief
 */
struct vertical_tilt_parameter : frame_wise_parameter {
public:

    vertical_tilt_parameter(std::string name, body_part a1, body_part a2, body_part b1, body_part b2, int direction) noexcept
        : frame_wise_parameter(name, vault_part::runup, "°"), a1_part(a1), a2_part(a2), b1_part(b1), b2_part(b2), dir(direction) {}

protected:
    
    /**
     * @brief Compute body part's height from given body points in single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of `b_part` in frame given by body points.
     * 
     * @see parameter::extract_value
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

    body_part a1_part, a2_part, b1_part, b2_part;
    int dir;

};

/**
 * @brief
 */
struct horizontal_tilt_parameter : vertical_tilt_parameter {
public:

    horizontal_tilt_parameter(std::string name, body_part a1, body_part a2, body_part b1, body_part b2, int direction) noexcept
        : vertical_tilt_parameter(name, a1, a2, b1, b2, direction) {}

private:

    /**
     * @brief Compute body part's height from given body points in single frame.
     * 
     * @param body Body from which to extract value.
     * @returns y-coordinate of `b_part` in frame given by body points.
     * 
     * @see parameter::extract_value
     */
    virtual std::optional<double> extract_value(const frame_body &body) const noexcept {
        std::optional<double> vertical_tilt = vertical_tilt_parameter::extract_value(body);
        if (vertical_tilt) return 90.0 - std::abs(*vertical_tilt);
        return std::nullopt;
    }

};
