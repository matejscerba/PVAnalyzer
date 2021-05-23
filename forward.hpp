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

/**
 * @brief Position of athlete's body part in frame.
 */
typedef std::optional<cv::Point2d> frame_point;

/**
 * @brief Positions of all athlete's body parts in frame.
 */
typedef std::vector<frame_point> frame_points; 

/**
 * @brief Positions of all athlete's body parts in whole video.
 */
typedef std::vector<frame_points> frame_video_points;

/**
 * @brief Position of athlete's body part.
 */
typedef std::optional<cv::Point3d> model_point;

/**
 * @brief Positions of all athlete's body parts in one moment.
 */
typedef std::vector<model_point> model_points;

/**
 * @brief Positions of all athlete's body parts during whole vault.
 */
typedef std::vector<model_points> model_video_points;

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
    all,
    runup,
    takeoff,
    vault,
    invalid
};

/**
 * @brief Side of tilt angle.
 */
enum class tilt_angle_side {
    top,
    bottom
};

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
 * @brief Name of file, where model will be saved.
 * 
 * Including extension.
 */
extern const std::string MODEL_FILE;

/**
 * @brief Name of file, where parameters will be saved.
 * 
 * Including extension.
 */
extern const std::string PARAMETERS_FILE;

/**
 * @brief Determines size ratio of window in which to detect body parts.
 * 
 * Ratio of window in which to detect body parts and initial athlete's
 * bounding box.
 * 
 * @note Measures size linearly, not bounding box's surface.
 */
const double BASE_SCALE_FACTOR = 1.3;

/**
 * @brief Number of body parts, that is being detected.
 * 
 * @note Background is not detected.
 */
const int NPOINTS = 15;

/**
 * @brief Number of pairs of body parts (joined by line to form a stickman).
 */
const int NPAIRS = 14;

/**
 * @brief Body parts pairs specified by indices.
 * 
 * Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5,
 * Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11,
 * Left Knee – 12, Left Ankle – 13, Chest – 14.
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
 * @brief Time to analyze takeoff parameter for.
 */
const double TAKEOFF_PARAM_TIME = 0.1;

/**
 * @brief Rotation shifts to use when detecting body parts.
 */
const std::vector<double> SHIFTS{ 0, -20, 20 };

/**
 * @brief Time, for which to track person, which is not moving.
 */
const double PERSON_CHECK_TIME = 0.3;

/**
 * @brief Threshold to be used for non-maxima suppression algorithm.
 */
const double NMS_THRESHOLD = 0.5;

/**
 * @brief Number of rows and cols of trackers' grid.
 */
const int GRID_SIZE = 3;

/**
 * @brief Number of frames to update grid again.
 */
const int GRID_UPDATE_FRAMES = 2;

/**
 * @brief Video's frame size (smaller dimension) to be scaled to.
 */
const std::size_t FRAME_SIZE = 720;

/**
 * @brief Torso tilt's difference threshold for parts detection's angle update.
 */
const double ANGLE_DIFF = 30.0;

/**
 * @brief Threshold to filter step frames by.
 */
const double STEP_THRESHOLD = 1.38;

/**
 * @brief Size of window used for parts detection - smaller dimension.
 */
const std::size_t WINDOW_SIZE = 150;

//////////////////////////////////////////////////////////////////////////////////////
// functions
//////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Add two frame_points if both are valid, return invalid otherwise.
 */
frame_point operator+(const frame_point &lhs, const frame_point &rhs);

/**
 * @brief Subtract two frame_points if both are valid, return invalid otherwise.
 */
frame_point operator-(const frame_point &lhs, const frame_point &rhs);

/**
 * @brief Divide frame_point by number.
 */
frame_point operator/(const frame_point &lhs, double rhs);

/**
 * @brief Add two model_points if both are valid, return invalid otherwise.
 */
model_point operator+(const model_point &lhs, const model_point &rhs);

/**
 * @brief Unary minus operator for model_point.
 */
model_point operator-(const model_point &p);

/**
 * @brief Subtract two model_points if both are valid, return invalid otherwise.
 */
model_point operator-(const model_point &lhs, const model_point &rhs);

/**
 * @brief Divide model_point by number.
 */
model_point operator/(const model_point &lhs, double rhs);

/**
 * @brief Multiply number by optional number if it contains value, return invalid otherwise.
 */
std::optional<double> operator*(double lhs, const std::optional<double> &rhs);

cv::Size operator*(double lhs, const cv::Size &rhs);

/**
 * @brief Write model_point to output stream.
 */
std::ostream& operator<<(std::ostream& os, const model_point &p);

/**
 * @brief Write frame_point to output stream.
 */
std::ostream& operator<<(std::ostream& os, const frame_point &p);

/**
 * @brief Compute center of given rectangle.
 * 
 * @param rect Rectangle whose center should be computed.
 * @returns point in center of given rectangle.
 */
cv::Point2d get_center(const cv::Rect &rect);

/**
 * @brief Count offset of given points.
 * 
 * @param begin Iterator specifying beginning of values to be processed.
 * @param end Iterator specifying end of values to be processed.
 * @returns offset.
*/
frame_point count_offset(frame_points::const_iterator begin, frame_points::const_iterator end);

/**
 * @brief Get name of given body part.
 */
std::string body_part_name(const body_part part);

/**
 * @brief Compute distance of two frame_points.
 * 
 * If both frame_points are valid, compute their distance,
 * otherwise return no value.
 */
std::optional<double> distance(const frame_point &a, const frame_point &b);

/**
 * @brief Compute distance of two model_points.
 * 
 * If both model_points are valid, compute their distance,
 * otherwise return no value.
 */
std::optional<double> distance(const model_point &a, const model_point &b);

/**
 * @brief Get model_point satisfying comparison function in z coordinate.
 */
model_point get_part(const model_point &a, const model_point &b, std::function<bool (double, double)> compare);

/**
 * @brief Get height of model_point satisfying comparison function.
 */
std::optional<double> get_height(const model_point &a, const model_point &b, std::function<bool (double, double)> compare);

/**
 * @brief Skeleton's size.
 * 
 * Distance of two furthest points among points.
 */
double size(const model_points &body);

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
std::vector<std::size_t> get_frame_numbers( std::vector<model_points>::const_iterator begin,
                                            std::vector<model_points>::const_iterator end,
                                            std::function<bool (double, double)> compare,
                                            double fps);

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
std::vector<std::size_t> get_step_frames(const model_video_points &points, double fps);

/**
 * @brief Compute angle of ray with origin in `a` going through `b` and vertical axis.
 */
std::optional<double> get_vertical_tilt_angle(const model_point &a, const model_point &b, tilt_angle_side tilt_side);

/**
 * @brief Compute angle of ray with origin in `a` going through `b` and horizontal axis (growing).
 */
double get_horizontal_tilt_angle(const cv::Point3d &a, const cv::Point3d &b);

/**
 * @brief Compute angle of ray with origin in `a` going through `b` and vertical axis.
 */
std::optional<double> get_vertical_tilt_angle(const frame_point &a, const frame_point &b);

/**
 * @brief Get path to output directory.
 * 
 * Create new directory if it doesn't exist yet.
 * 
 * @param video_filename Path to processed video for which to create output directory.
 * @returns path to output directory.
 */
std::string get_output_dir(const std::string &video_filename);

/**
 * @brief Check if given rectangle is inside `frame`.
 */
bool is_inside(const cv::Rect &rect, const cv::Mat &frame);

/**
 * @brief Draw body into given frame.
 */
void draw_body(cv::Mat &frame, const frame_points &body);

/**
 * @brief Convert frame_points to frame_body.
 * 
 * Leave out y coordinte of frame_points.
 */
frame_points model_to_frame(const model_points &body);

/**
 * @brief Resize `frame` so that its smaller dimension is `height`.
 */
cv::Mat resize(const cv::Mat &frame, std::size_t height = FRAME_SIZE);

/**
 * @brief Compute area of given rectangle.
 */
double area(const cv::Rect &r);

/**
 * @brief Check if `p` is inside `r`.
 */
bool is_inside(const cv::Point2d &p, const cv::Rect &r);

/**
 * @brief Convert size of `frame` to rectangle.
 */
cv::Rect rect(const cv::Mat &frame);

/**
 * @brief Enclose rectangles into smallest possible rectangle.
 */
cv::Rect rect(const std::vector<cv::Rect> &rs);

/**
 * @brief Split bounding box into grid of smaller bounding boxes.
 */
std::vector<cv::Rect> split(const cv::Rect &bbox);

/**
 * @brief Compute average distance of centers of given rectangles.
 */
double average_dist(const std::vector<cv::Rect> &rs);
