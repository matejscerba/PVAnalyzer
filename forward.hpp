#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <ostream>
#include <ctime>
#include <sstream>

/**
 * @brief File containing definitions of type aliases, constants and helper functions.
 */

//////////////////////////////////////////////////////////////////////////////////////
// typedefs
//////////////////////////////////////////////////////////////////////////////////////

/// @brief Represents person's body part in frame, can be invalid.
typedef std::optional<cv::Point2d> frame_part;

/// @brief Represents person's body in frame.
typedef std::vector<frame_part> frame_body;

/// @brief Represents person's body in all frames where person was detected.
typedef std::vector<frame_body> video_body;

/// @brief Represents corners of person's bounding box in all frames where person was detected.
typedef std::vector<std::optional<std::vector<cv::Point2d>>> person_corners;

//////////////////////////////////////////////////////////////////////////////////////
// enums
//////////////////////////////////////////////////////////////////////////////////////

/// @brief Maps bounding box corners to indices.
enum corner : std::size_t {
    /// Top left.
    tl = 0,
    /// Top right.
    tr,
    /// Bottom left.
    bl,
    /// Bottom right.
    br
};

/**
 * @brief Maps body parts' names to correct indices.
 */
enum body_part : std::size_t {
    head = 0,
    neck = 1,
    r_shoulder = 2,
    r_elbow = 3,
    r_wrist = 4,
    l_shoulder = 5,
    l_elbow = 6,
    l_wrist = 7,
    r_hip = 8,
    r_knee = 9,
    r_ankle = 10,
    l_hip = 11,
    l_knee = 12,
    l_ankle = 13,
    chest = 14
};

/**
 * @brief Supported horizontal movement directions and their corresponding values.
 * 
 * @note This value multiplies angle so that video rotates the correct way during vault.
*/
enum class direction {
    right,
    unknown,
    left
};

/**
 * @brief Part of vault.
 */
enum class vault_part {
    runup,
    takeoff,
    vault,
    invalid
};

//////////////////////////////////////////////////////////////////////////////////////
// functions
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Add two optional points if both are valid.
 * 
 * @returns sum of two points if both are valid, empty optional otherwise.
 */
std::optional<cv::Point2d> operator+(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs);

std::optional<cv::Point2d> operator-(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs);

/**
 * @brief
 */
std::optional<cv::Point2d> operator/(const std::optional<cv::Point2d> &lhs, double rhs);

/**
 * @brief
 */
std::optional<double> operator*(double lhs, const std::optional<double> &rhs) noexcept;

/**
 * @brief Add two video bodies together.
 * 
 * @returns result of appending `rhs` to `lhs`.
 */
video_body operator+(const video_body &&lhs, const video_body &&rhs);

/**
 * @brief Extract corners from rectangle.
 * 
 * @param rect Rectangle to extract corners from.
 * @returns vector of points representing corners so that indices
 *     correspond to `enum corner`.
*/
std::vector<cv::Point2d> get_corners(const cv::Rect &rect);

/**
 * @brief Computes center of given frame.
 * 
 * @param frame Given frame used to compute its center.
 * @returns point in center of given frame.
 */
cv::Point get_center(const cv::Mat &frame);

/**
 * @brief Compute center of given rectangle.
 * 
 * @param rect Rectangle whose center should be computed.
 * @returns point in center of given rectangle.
 */
cv::Point2d get_center(const cv::Rect &rect);

std::optional<cv::Point2d> get_center(const std::optional<cv::Rect> &rect);

/**
 * @brief Count mean of offset of given consecutive values in vector.
 * 
 * @param begin Iterator specifying beginning of values to be processed.
 * @param end Iterator specifying end of values to be processed.
 * @returns mean of offsets of given consecutive values.
*/
std::optional<cv::Point2d> count_mean_delta(std::vector<std::optional<cv::Point2d>>::const_iterator begin, std::vector<std::optional<cv::Point2d>>::const_iterator end) noexcept;

/**
 * @brief
 */
std::string body_part_name(const body_part part);

std::optional<double> distance(const std::optional<cv::Point2d> &a, const std::optional<cv::Point2d> &b) noexcept;

frame_part get_part(const frame_part &a, const frame_part &b, std::function<bool (double, double)> compare) noexcept;

std::optional<double> get_height(const frame_part &a, const frame_part &b, std::function<bool (double, double)> compare) noexcept;

/**
 * @brief Create name for output file from current date.
 * 
 * @returns name for output file.
 */
std::string create_output_filename() noexcept;

/**
 * @brief Get numbers of frames in which ankle specified by `compare` reaches local point of interest.
 * 
 * `compare` returns true if given values are in correct order based on `compare`. Only ankle satisfying
 * `compare` for each frame is taken into consideration, once it leaves point of interest (next ankle in
 * next frame is vertically in the opposit relation than `compare` by more than 1 px) the previous frame
 * number is added to vector that will be returned.
 * 
 * @param begin Const iterator to beginning of detected body parts of athlete in the whole video.
 * @param end Const iterator to end of detected body parts of athlete in the whole video.
 * @param compare Binary comparison function.
 * @returns vector of frame numbers where ankles leave specified local point of interest.
 */
std::vector<std::size_t> get_frame_numbers( std::vector<frame_body>::const_iterator begin,
                                            std::vector<frame_body>::const_iterator end,
                                            std::function<bool (double, double)> compare) noexcept;
/**
 * @brief Get frame numbers in which athlete's foot leaves ground.
 * 
 * Get numbers of frames where the lower ankle leaves locally lowest point and where
 * the higher ankle leaves locally highest point. Keep low points, which are below
 * center of previous low point and high point (the first one is kept every time).
 * 
 * @param points Athlete's body parts detected in the whole video.
 * @returns numbers of frames in which athlete's lower foot leaves ground.
 */
std::vector<std::size_t> get_step_frames(const video_body &points) noexcept;

std::optional<double> get_vertical_tilt_angle(const frame_part &a, const frame_part &b) noexcept;

//////////////////////////////////////////////////////////////////////////////////////
// constants
//////////////////////////////////////////////////////////////////////////////////////

/// @brief Path to protofile to be used for deep neural network intialization.
extern const std::string protofile;

/// @brief Path to caffe model to be used for deep neural network intialization.
extern const std::string caffemodel;

/// @brief Expected vault duration in seconds.
const double vault_duration = 0.8;

/**
 * @brief Determines size of bounding box where to detect body parts.
 * 
 * Ratio of size of bounding box used to detect body parts and size of
 * bounding box tracked by `tracker`.
 * 
 * @note Measures size linearly, not bounding box's surface.
 */
const double scale_factor = 1.8;

/// @brief Number of body parts, that is being detected.
const int npoints = 16;

/// @brief Number of pairs of body parts (joined by line to form a stickman).
const int npairs = 14;

/**
 * @brief Body parts pairs specified by indices.
 * 
 * Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5,
 * Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11,
 * Left Knee – 12, Left Ankle – 13, Chest – 14, Background – 15.
 */
const std::size_t pairs[14][2] = {
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
};

/// @brief Minimal probability value to mark body part as valid.
const double detection_threshold = 0.1;

/// @brief How many seconds to check for vault beginning.
const double vault_check_time = 0.2;

/// @brief How much the person's coordinates must change in order to set `_vault_began` to true.
const double vault_threshold = -2.5 * 137.0;

const std::size_t takeoff_parameter_frames = 3;
