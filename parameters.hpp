#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <ostream>

/**
 * @brief
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

protected:

    /**
     * @brief Part of vault.
     */
    enum class vaut_part {
        runup,
        takeoff,
        vault
    }

    /**
     * @brief This parameter's unit in format to be written immediately after value.
     */
    const std::string unit;

    /**
     * @brief Default constructor.
     * 
     * @param part Part of vault, for which this parameter is valid.
     * @param unit This parameter's unit.
     * 
     * @note This class is abstract, so constructor is protected.
     */
    parameter(const std::string name, const vault_part part, const std::string &&unit) noexcept
        : name(name), part(part), unit(unit) {}

};

/**
 * @brief
 */
struct single_value_parameter : parameter {

};

/**
 * @brief 
 */
struct multiple_values_parameter : parameter {

};

/**
 * @brief 
 */
struct frame_wise_parameter : parameter {

};