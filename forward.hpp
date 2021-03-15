#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <optional>

/**
 * @brief File containing definitions of type aliases, constants and helper functions.
 */

/// @brief Represents body part in frame, can be invalid.
typedef std::optional<cv::Point2d> frame_part;

/// @brief Represents whole body in frame.
typedef std::vector<frame_part> frame_body;

/// @brief Represents whole body in all frames where person was detected.
typedef std::vector<frame_body> video_body;

/// @brief Represents corners of person's bounding box in all frames where person was detected.
typedef std::vector<std::vector<cv::Point2d>> person_corners;

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
 * @brief Maps body parts' names to correct indices to access its points.
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
*/
enum direction : int {
    right = -1,
    unknown = 0,
    left = 1
};

/**
 * @brief Add two optional points if both are valid.
 * 
 * @returns sum of two points if both are valid, empty optional otherwise.
 */
std::optional<cv::Point2d> operator+(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs);

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
 * @returns center of `rect`.
 */
cv::Point2d get_center(const cv::Rect &rect);

/**
 * @brief Count mean of offset of given consecutive values in vector.
 * 
 * @param begin Iterator specifying beginning of values to be processed.
 * @param end Iterator specifying end of values to be processed.
 * @returns mean of offsets of given consecutive values.
*/
cv::Point2d count_mean_delta(std::vector<cv::Point2d>::const_iterator begin, std::vector<cv::Point2d>::const_iterator end);

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
const int pairs[14][2] = {
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
};

/// @brief Minimal probability value to mark body part as valid.
const double detection_threshold = 0.1;

/// @brief How many frames to check to determine if vault began.
const std::size_t vault_check_frames = 6;

/// @brief How much the person's coordinates must change in order to set `_vault_began` to true.
const double vault_threshold = -0.55 / 720;
