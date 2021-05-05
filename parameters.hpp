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
    virtual void compute(const model_video_points &points) noexcept = 0;

    /**
     * @brief Write value into given stream for given frame number.
     * 
     * @param os Stream in which this parameter's value for given frame number should be written.
     * @param frame_no Number of frame for which to write value.
     * @param write_unit Specifies whether to write this parameter's unit as well.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept = 0;

    /**
     * @brief Write value into given stream for given frame number.
     * 
     * @param os Stream in which this parameter's value for given frame number should be written.
     * @param value_no Number of value to write.
     */
    virtual void write_param(std::ostream &os, std::size_t value_no) const noexcept = 0;

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

/**
 * @brief Comparison operator to be used for sorting parameters.
 */
bool operator<(const std::shared_ptr<parameter> &lhs, const std::shared_ptr<parameter> &rhs) noexcept {
    return (lhs && rhs && (lhs->name < rhs->name));
}

//////////////////////////////////////////////////////////////////////////////////////
// single values parameters
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Represents single-valued parameter, e.g. vault height.
 */
struct single_value_parameter : parameter {
public:

    /**
     * @see parameter::write_param
     */
    virtual void write_param(std::ostream &os, std::size_t value_no) const noexcept {
        if (value) os << *value;
    }

    /**
     * @see parameter::size
     */
    virtual std::size_t size() const noexcept {
        return 1;
    }

protected:

    /**
     * @brief Value of this parameter.
     */
    std::optional<double> value;

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault for which this parameter is designated.
     * @param unit String representation of this parameter's unit.
     * 
     * @note This constructor can be called by derived structs only.
     */
    single_value_parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : parameter(name, part, std::move(unit)) {}

};

/**
 * @brief Single valued parameter, whose value is associated with takeoff.
 */
struct takeoff_parameter : single_value_parameter {
protected:

    /**
     * @brief Number of frame in which athlete takes off.
     */
    const std::size_t takeoff;

    /**
     * @brief Default constructor.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault for which this parameter is designated.
     * @param unit String representation of this parameter's unit.
     * 
     * @note This constructor can be called by derived structs only.
     */
    takeoff_parameter(std::size_t takeoff, const std::string name, const vault_part part, const std::string &&unit) noexcept
        : single_value_parameter(name, part, std::move(unit)), takeoff(takeoff) {}

};

/**
 * @brief Velocity loss of center of given body parts during takeoff.
 */
struct velocity_loss : takeoff_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param takeoff Number of frame in which athlete takes off.
     * @param name Name of this parameter.
     * @param a, b Body parts whose velocity loss should be computed.
     */
    velocity_loss(std::size_t takeoff, const std::string name, body_part a, body_part b) noexcept
        : takeoff_parameter(takeoff, name, vault_part::takeoff, " %"), a(a), b(b) {}

    /**
     * @brief Compute horizontal velocity loss during takeoff of given body parts.
     * 
     * Velocity is computed for last `takeoff_parameter_frames` before takeoff and for
     * first `takeoff_parameter_frames` after takeoff. Those two values determine horizontal
     * velocity loss.
     * 
     * @param points Athlete's body parts detected in the whole video.
     */
    virtual void compute(const model_video_points &points) noexcept {
        if ((takeoff >= TAKEOFF_PARAM_FRAMES) && (points.size() > takeoff + TAKEOFF_PARAM_FRAMES)) {
            model_points before = points[takeoff - TAKEOFF_PARAM_FRAMES];
            model_points during = points[takeoff];
            model_points after = points[takeoff + TAKEOFF_PARAM_FRAMES];
            model_point p_before = (before[a] + before[b]) / 2.0;
            model_point p_during = (during[a] + during[b]) / 2.0;
            model_point p_after = (after[a] + after[b]) / 2.0;
            if (p_before && p_during && p_after) {
                value = (1.0 - (p_after - p_during)->x / (p_during - p_before)->x) * 100.0;
            } else {
                value = std::nullopt;
            }
        } else {
            value = std::nullopt;
        }
    }

    /**
     * @brief Write velocity loss if `frame_no` is the number of frame where athlete takes off.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        if ((frame_no == takeoff) && value) {
            os << *value << (write_unit ? unit : "");
        }
    }

private:

    /**
     * @brief Body parts to compute velocity loss from.
     */
    body_part a, b;

};

/**
 * @brief Horizontal velocity loss of center of hips.
 */
struct hips_velocity_loss : velocity_loss {
public:

    /**
     * @brief Default constructor.
     * 
     * @param takeoff Number of frame in which athlete takes off.
     */
    hips_velocity_loss(std::size_t takeoff) noexcept
        : velocity_loss(takeoff, "Hips velocity loss", body_part::l_hip, body_part::r_hip) {}

};

/**
 * @brief Horizontal velocity loss of center of shoulders.
 */
struct shoulders_velocity_loss : velocity_loss {
public:

    /**
     * @brief Default constructor.
     * 
     * @param takeoff Number of frame in which athlete takes off.
     */
    shoulders_velocity_loss(std::size_t takeoff) noexcept
        : velocity_loss(takeoff, "Shoulders velocity loss", body_part::l_shoulder, body_part::r_shoulder) {}

};

/**
 * @brief Athlete's takeoff angle.
 */
struct takeoff_angle : takeoff_parameter {
public:

    /**
     * @brief Default constructor.
     * 
     * @param takeoff Number of frame in which athlete takes off.
     */
    takeoff_angle(std::size_t takeoff) noexcept
        : takeoff_parameter(takeoff, "Takeoff angle", vault_part::takeoff, "°") {}

    /**
     * @brief Compute athlete's takeoff angle (hips are used to compute angle).
     * 
     * Angle is computed for last `takeoff_parameter_frames` before takeoff and for
     * first `takeoff_parameter_frames` after takeoff. Those two values determine
     * resulting angle.
     * 
     * @param points Athlete's body parts detected in the whole video.
     */
    virtual void compute(const model_video_points &points) noexcept {
        if ((takeoff >= TAKEOFF_PARAM_FRAMES) && (points.size() > takeoff + TAKEOFF_PARAM_FRAMES)) {
            model_points first = get_first_hips(points);
            model_points during = points[takeoff];
            model_points after = points[takeoff + TAKEOFF_PARAM_FRAMES];
            model_point p_first = (first[body_part::l_hip] + first[body_part::r_hip]) / 2.0;
            model_point p_during = (during[body_part::l_hip] + during[body_part::r_hip]) / 2.0;
            model_point p_after = (after[body_part::l_hip] + after[body_part::r_hip]) / 2.0;
            if (p_first && p_during && p_after) {
                double runup_angle = std::atan((p_first - p_during)->z / std::abs((p_first - p_during)->x));
                double takeoff_angle = std::atan((p_during - p_after)->z / std::abs((p_during - p_after)->x));
                value = (takeoff_angle - runup_angle) * 180.0 / M_PI;
            } else {
                value = std::nullopt;
            }
        } else {
            value = std::nullopt;
        }
    }

    /**
     * @brief Write takeoff angle if `frame_no` is the number of frame where athlete takes off.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        if ((frame_no == takeoff) && value) {
            os << *value << (write_unit ? unit : "");
        }
    }

private:

    /**
     * @brief Get first frame_body in which both hips are valid.
     * 
     * @param points Athlete's body parts detected in the whole video.
     * 
     * @returns first frame_body in which both hips are valid.
     */
    model_points get_first_hips(const model_video_points &points) const noexcept {
        auto found = std::find_if(points.begin(), points.end(), [](const model_points &body){
            return body[body_part::l_hip] && body[body_part::r_hip];
        });
        if (found != points.end()) {
            return *found;
        } else {
            return points[0];
        }
    }

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

    /**
     * @see parameter::write_param
     */
    virtual void write_param(std::ostream &os, std::size_t value_no) const noexcept {
        if (values[value_no]) os << *values[value_no];
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
 * @brief Represents parameter with value for each step.
 */
struct steps_parameter : multiple_values_parameter {
protected:

    /**
     * @brief Numbers of frames where step happens.
     */
    std::vector<std::size_t> step_frames;

    /**
     * @brief Frame rate of processed video.
     */
    double fps;

    /**
     * @brief Default constructor
     * 
     * Specifies parameter's name, vault part and unit as well as frame rate of processed video.
     * 
     * @param name Name of this parameter.
     * @param part Part of vault for which this parameter is designated.
     * @param unit String representation of this parameter's unit.
     * @param fps Frame rate of processed video.
     * 
     * @note This constructor can be called by derived structs only.
     */
    steps_parameter(const std::string name, const vault_part part, const std::string &&unit, double fps = 30) noexcept
        : multiple_values_parameter(name, vault_part::runup, " s"), fps(fps) {}

};

/**
 * @brief Duration of each step in seconds.
 */
struct steps_duration : steps_parameter {
public:

    /**
     * @brief Default constructor
     * 
     * Specifies parameter's name, vault part and unit as well as frame rate of processed video.
     * 
     * @param fps Frame rate of processed video.
     */
    steps_duration(double fps) noexcept
        : steps_parameter("Step duration", vault_part::runup, " s", fps) {}

    /**
     * @brief Compute duration of each step based on detected body parts in whole video.
     * 
     * Compute duration between subsequent low points based on their frame numbers and
     * frame rate of processed video.
     * 
     * @param points Athlete's body parts detected in the whole video.
     */
    virtual void compute(const model_video_points &points) noexcept {
        values.clear();
        step_frames = get_step_frames(points);
        for (std::size_t i = 1; i < step_frames.size(); ++i) {
            values.push_back((step_frames[i] - step_frames[i - 1]) / fps);
        }
    }

    /**
     * @brief Write duration of step happening during frame `frame_no`.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        std::size_t before = 0;
        for (; before < step_frames.size(); ++before) {
            if (step_frames[before] >= frame_no) {
                // `frame_no` is before `step_frames[before]`.
                break;
            }
        }
        if (before && (frame_no <= step_frames.back())) {
            os << *values[before - 1] << (write_unit ? unit : "");
        }
    }

};

/**
 * @brief Angle of ray from hips center to lower ankle and vertical axis when athlete's foot reaches ground.
 */
struct steps_angle : steps_parameter {
public:

    /**
     * @brief Default constructor
     * 
     * Specifies parameter's name, vault part and unit as well as frame rate of processed video.
     * 
     * @param fps Frame rate of processed video.
     */
    steps_angle(direction dir) noexcept
        : steps_parameter("Step angle", vault_part::runup, "°"), dir(dir) {}

    /**
     * @brief Compute leg angle for each step.
     * 
     * @param points Athlete's body parts detected in the whole video.
     */
    virtual void compute(const model_video_points &points) noexcept {
        std::vector steps = get_step_frames(points);
        model_video_points reversed(points.rbegin(), points.rend());
        std::vector<std::size_t> step_beg_frames = get_frame_numbers(reversed.begin(), reversed.end(), std::less<double>());
        for (auto &i : step_beg_frames) {
            i = points.size() - i - 1;
        }
        step_frames.clear();
        std::vector<std::size_t> res;
        for (std::size_t i = 0, j = steps.size(); i < step_beg_frames.size(); ++i) {
            if (!j) break;
            if (step_beg_frames[i] <= steps[j - 1]) {
                --j;
                step_frames.push_back(step_beg_frames[i]);
            }
        }
        values.clear();
        for (auto i : step_frames) {
            values.push_back(extract_value(points[i]));
        }
    }

    /**
     * @brief Write leg angle if step happens in frame `frame_no`.
     * 
     * @param[out] os Output stream.
     * @param frame_no Number of frame for which this parameter's value should be written.
     * @param write_unit True if unit is supposed to be written as well, false otherwise.
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        auto found = std::find(step_frames.begin(), step_frames.end(), frame_no);
        if (found != step_frames.end()) {
            os << *values[found - step_frames.begin()] << (write_unit ? unit : "");
        }
    }

private:

    /**
     * @brief Direction of athlete's runup.
     */
    direction dir;

    /**
     * @brief Compute angle between ray from center of hips to lower ankle and vertical axis.
     * 
     * @param body Body from which to extract value.
     * 
     * @returns angle between ray from center of hips to lower ankle and vertical axis.
     * 
     * @note Negative angle represents ray facing in the opposite direction than athlete's runup.
     */
    virtual std::optional<double> extract_value(const model_points &body) const noexcept {
        model_point a = (body[body_part::l_hip] + body[body_part::r_hip]) / 2.0;
        model_point b = get_part(body[body_part::l_ankle], body[body_part::r_ankle], std::less<double>());
        if (a && b) {
            double y = a->z - b->z;
            double d = 0;
            if (dir == direction::left)
                d = -1;
            else if (dir == direction::right)
                d = 1;
            double x = d * (a->x - b->x);
            return std::atan(x / y) * 180.0 / M_PI;
        }
        return std::nullopt;
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
     * @brief Compute this parameter's value for each frame and save them into `values`.
     * 
     * @param points Detected body parts to be processed.
     */
    virtual void compute(const model_video_points &points) noexcept {
        values.clear();
        std::transform(points.begin(), points.end(), std::back_inserter(values),
                       [this](const model_points &body){ return this->extract_value(body); }
        );
    }

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
     */
    virtual void write_value(std::ostream &os, std::size_t frame_no, bool write_unit) const noexcept {
        if (values[frame_no]) {
            os << *values[frame_no] << (write_unit ? unit : "");
        } else if (write_unit) {
            os << "?" << unit;
        }
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
    virtual std::optional<double> extract_value(const model_points &body) const noexcept = 0;

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
     * 
     * @returns y-coordinate of hips in frame given by body points.
     */
    virtual std::optional<double> extract_value(const model_points &body) const noexcept {
        // Average of left and right hip's y-coordinate (if possible), otherwise take one hip if possible.
        if (body[body_part::r_hip] && body[body_part::l_hip]) {
            return ((*body[body_part::r_hip] + *body[body_part::l_hip]) / 2).z;
        } else if (body[body_part::r_hip]) {
            return body[body_part::r_hip]->z;
        } else if (body[body_part::l_hip]) {
            return body[body_part::l_hip]->z;
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
     * 
     * @returns y-coordinate of `b_part` in frame given by body points.
     */
    virtual std::optional<double> extract_value(const model_points &body) const noexcept {
        if (body[b_part]) return body[b_part]->z;
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
     * 
     * @returns angle between ray specified by body parts and vertical axis.
     * 
     * @note Negative angle represents ray facing in the opposite direction than athlete's runup.
     * @note Angle == 0 if ray faces directly upwards, degrees range is [-180,180].
     */
    virtual std::optional<double> extract_value(const model_points &body) const noexcept {
        double d = 0;
        if (dir == direction::left)
            d = 1;
        else if (dir == direction::right)
            d = -1;
        return d * get_vertical_tilt_angle((body[a1_part] + body[a2_part]) / 2.0, (body[b1_part] + body[b2_part]) / 2.0);
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
     * 
     * @returns angle between ray specified by body parts and horizontal axis.
     * 
     * @note Negative angle represents ray facing downwards.
     * @note Angle's degrees range is [-90,90].
     */
    virtual std::optional<double> extract_value(const model_points &body) const noexcept {
        std::optional<double> vertical_tilt = vertical_tilt::extract_value(body);
        if (vertical_tilt) return 90.0 - std::abs(*vertical_tilt);
        return std::nullopt;
    }

};
