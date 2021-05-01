#pragma once

#include <opencv2/opencv.hpp>

#include <ctime>
#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

/**
 * @brief File containing definitions of type aliases, constants and helper functions.
 */

//////////////////////////////////////////////////////////////////////////////////////
// typedefs
//////////////////////////////////////////////////////////////////////////////////////

typedef std::optional<cv::Point3d> model_point;
typedef std::vector<model_point> model_body;
typedef std::vector<model_body> model_video_body;
typedef std::vector<model_point> model_offsets;

typedef std::optional<cv::Point2d> offset;
typedef std::vector<offset> offsets;

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

/**
 * @brief Maps corners to indices.
 */
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
 * @brief Maps body parts' names to indices.
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
 * @brief Supported horizontal movement directions.
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
 * @brief Add two frame_parts if both are valid, return invalid otherwise.
 */
frame_part operator+(const frame_part &lhs, const frame_part &rhs) noexcept;

/**
 * @brief Subtract two frame_parts if both are valid, return invalid otherwise.
 */
frame_part operator-(const frame_part &lhs, const frame_part &rhs) noexcept;

/**
 * @brief Divide frame_part by number.
 */
frame_part operator/(const frame_part &lhs, double rhs) noexcept;

/**
 * @brief Add two model_points if both are valid, return invalid otherwise.
 */
model_point operator+(const model_point &lhs, const model_point &rhs) noexcept;

/**
 * @brief Unary minus operator for model_point.
 */
model_point operator-(const model_point &p) noexcept;

/**
 * @brief Subtract two model_points if both are valid, return invalid otherwise.
 */
model_point operator-(const model_point &lhs, const model_point &rhs) noexcept;

/**
 * @brief Divide model_point by number.
 */
model_point operator/(const model_point &lhs, double rhs) noexcept;

/**
 * @brief Multiply number by optional number if it contains value, return invalid otherwise.
 */
std::optional<double> operator*(double lhs, const std::optional<double> &rhs) noexcept;

/**
 * @brief Write model_point to output stream.
 */
std::ostream& operator<<(std::ostream& os, const model_point &p) noexcept;

/**
 * @brief Compute center of given rectangle.
 * 
 * @param rect Rectangle whose center should be computed.
 * @returns point in center of given rectangle.
 */
cv::Point2d get_center(const cv::Rect &rect) noexcept;

/**
 * @brief Count mean offset of given points.
 * 
 * @param begin Iterator specifying beginning of values to be processed.
 * @param end Iterator specifying end of values to be processed.
 * @returns mean offset.
*/
offset count_mean_delta(offsets::const_iterator begin, offsets::const_iterator end) noexcept;

/**
 * @brief Get name of given body part.
 */
std::string body_part_name(const body_part part) noexcept;

/**
 * @brief Compute distance of two frame_parts.
 * 
 * If both frame_parts are valid, compute their distance,
 * otherwise return no value.
 */
std::optional<double> distance(const frame_part &a, const frame_part &b) noexcept;

/**
 * @brief Compute distance of two model_points.
 * 
 * If both model_points are valid, compute their distance,
 * otherwise return no value.
 */
std::optional<double> distance(const model_point &a, const model_point &b) noexcept;

/**
 * @brief Get model_point satisfying comparison function in z coordinate.
 */
model_point get_part(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) noexcept;

/**
 * @brief Get height of model_point satisfying comparison function.
 */
std::optional<double> get_height(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) noexcept;

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
std::vector<std::size_t> get_frame_numbers( std::vector<model_body>::const_iterator begin,
                                            std::vector<model_body>::const_iterator end,
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
std::vector<std::size_t> get_step_frames(const model_video_body &points) noexcept;

/**
 * @brief Compute angle of ray with origin in `a` going through `b` and vertical axis.
 * 
 * @note y coordinate is not used.
 */
std::optional<double> get_vertical_tilt_angle(const model_point &a, const model_point &b) noexcept;

/**
 * @brief Compute angle of ray with origin in `a` going through `b` and vertical axis.
 */
std::optional<double> get_vertical_tilt_angle(const frame_part &a, const frame_part &b) noexcept;

/**
 * @brief Create name for output file from current date.
 * 
 * @returns name for output file.
 */
std::string create_output_filename() noexcept;

/**
 * @brief Check if given rectangle is inside `frame`.
 */
bool is_inside(const cv::Rect &rect, const cv::Mat &frame) noexcept;

/**
 * @brief Draw body into given frame.
 */
void draw_body(cv::Mat &frame, const frame_body &body) noexcept;

/**
 * @brief Convert model_body to frame_body.
 * 
 * Leave out y coordinte of model_body.
 */
frame_body model_to_frame(const model_body &body) noexcept;

//////////////////////////////////////////////////////////////////////////////////////
// constants
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Path to protofile to be used for deep neural network intialization.
 */
extern const std::string PROTOFILE;

/**
 * @brief Path to caffe model to be used for deep neural network intialization.
 */
extern const std::string CAFFEMODEL;

/**
 * @brief Determines size ratio of window in which to detect body parts.
 * 
 * Ratio of window in which to detect body parts and initial athlete's
 * bounding box.
 * 
 * @note Measures size linearly, not bounding box's surface.
 */
const double BASE_SCALE_FACTOR = 1.8;

/**
 * @brief Number of body parts, that is being detected.
 */
const int NPOINTS = 16;

/**
 * @brief Number of pairs of body parts (joined by line to form a stickman).
 */
const int NPAIRS = 14;

/**
 * @brief Body parts pairs specified by indices.
 * 
 * Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5,
 * Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11,
 * Left Knee – 12, Left Ankle – 13, Chest – 14, Background – 15.
 */
const std::size_t PAIRS[NPAIRS][2] = {
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
};

/**
 * @brief Minimal probability value to mark body part as valid.
 */
const double DET_THRESHOLD = 0.1;

/**
 * @brief How many seconds to check for vault beginning.
 */
const double VAULT_CHECK_TIME = 0.2;

/**
 * @brief How much the person's coordinates must change in order to set `_vault_began` to true.
 */
const double VAULT_THRESHOLD = -2.5 * 137.0;

/**
 * @brief Number of frames to analyze for takeoff parameter.
 */
const std::size_t TAKEOFF_PARAM_FRAMES = 3;

/**
 * @brief Rotation shifts to use when detecting body parts.
 */
const std::vector<double> SHIFTS{ 0, -20, 20 };
