#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

#include "forward.hpp"

const std::string PROTOFILE = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string CAFFEMODEL = "pose/mpi/pose_iter_160000.caffemodel";

const std::string MODEL_FILE = "model.txt";

const std::string PARAMETERS_FILE = "parameters.csv";

frame_point operator+(const frame_point &lhs, const frame_point &rhs) {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

frame_point operator-(const frame_point &lhs, const frame_point &rhs) {
    if (lhs && rhs) return *lhs - *rhs;
    return std::nullopt;
}

frame_point operator/(const frame_point &lhs, double rhs) {
    if (lhs) return *lhs / rhs;
    return std::nullopt;
}

model_point operator+(const model_point &lhs, const model_point &rhs) {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

model_point operator-(const model_point &p) {
    if (p) return -*p;
    return std::nullopt;
}

model_point operator-(const model_point &lhs, const model_point &rhs) {
    return lhs + (- rhs);
}

model_point operator/(const model_point &lhs, double rhs) {
    if (lhs) return *lhs / rhs;
    return std::nullopt;
}

std::optional<double> operator*(double lhs, const std::optional<double> &rhs) {
    if (rhs) return lhs * (*rhs);
    return std::nullopt;
}

cv::Size operator*(double lhs, const cv::Size &rhs) {
    return cv::Size(lhs * rhs.width, lhs * rhs.height);
}

std::ostream& operator<<(std::ostream& os, const model_point &p) {
    if (p) {
        os << p->x << "," << p->y << "," << p->z;
    } else {
        os << ",,";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const frame_point &p) {
    if (p) {
        os << p->x << "," << p->y;
    } else {
        os << ",";
    }
    return os;
}

cv::Point2d get_center(const cv::Rect &rect) {
    return cv::Point2d((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
}

frame_point count_offset(frame_points::const_iterator begin, frame_points::const_iterator end) {
    if (*(--end) && *begin) {
        return **end - **begin;
    }
    return std::nullopt;
}

std::string body_part_name(const body_part part) {
    switch (part) {
        case body_part::head:
            return "Head";
        case body_part::neck:
            return "Neck";
        case body_part::r_shoulder:
            return "Right shoulder";
        case body_part::r_elbow:
            return "Right elbow";
        case body_part::r_wrist:
            return "Right wrist";
        case body_part::l_shoulder:
            return "Left shoulder";
        case body_part::l_elbow:
            return "Left elbow";
        case body_part::l_wrist:
            return "Left wrist";
        case body_part::r_hip:
            return "Right hip";
        case body_part::r_knee:
            return "Right knee";
        case body_part::r_ankle:
            return "Right ankle";
        case body_part::l_hip:
            return "Left hip";
        case body_part::l_knee:
            return "Left knee";
        case body_part::l_ankle:
            return "Left ankle";
        case body_part::chest:
            return "Chest";
    }
    return "";
}

std::optional<double> distance(const frame_point &a, const frame_point &b) {
    if (a && b) {
        return cv::norm(*a - *b);
    }
    return std::nullopt;
}

std::optional<double> distance(const model_point &a, const model_point &b) {
    if (a && b) {
        return cv::norm(*a - *b);
    }
    return std::nullopt;
}

model_point get_part(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) {
    if (a && b) {
        if (compare(a->z, b->z))
            return a;
        return b;
    } else if (a) {
        return a;
    } else if (b) {
        return b;
    }
    return std::nullopt;
}

std::optional<double> get_height(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) {
    model_point p = get_part(a, b, compare);
    if (p) return p->z;
    return std::nullopt;
}

double size(const model_points &body) {
    double res = 0;
    for (const auto &p : body) {
        for (const auto &q : body) {
            if (p && q) {
                if (cv::norm(*p - *q) > res) {
                    res = cv::norm(*p - *q);
                }
            }
        }
    }
    return res;
}

std::vector<std::size_t> get_frame_numbers( std::vector<model_points>::const_iterator begin,
                                            std::vector<model_points>::const_iterator end,
                                            std::function<bool (double, double)> compare,
                                            double fps) {
    std::vector<std::size_t> res;
    std::optional<double> last_height = std::nullopt;
    std::size_t index = 0;
    bool correct_diff = false;
    double threshold = STEP_THRESHOLD / fps;
    for (; begin != end; ++begin, ++index) {
        std::optional<double> height = get_height((*begin)[body_part::l_ankle], (*begin)[body_part::r_ankle], compare);
        if (height && last_height) {
            // Current and last value is valid.
            if (correct_diff && !compare(*height, *last_height) && std::abs(*height - *last_height) > threshold * size(*begin)) {
                // Value was changing in the right direction, it stopped changing and is changing in the wrong direction.
                correct_diff = false;
                res.push_back(index - 1);
            }
            correct_diff = compare(*height, *last_height);
        }
        last_height = height;
    }
    return res;
}

std::vector<std::size_t> get_step_frames(const model_video_points &points, double fps) {
    std::vector<std::size_t> res;
    std::less<double> low;
    // Numbers of frames, in whose athlete's lower foot starts moving up.
    std::vector<std::size_t> lows = get_frame_numbers(points.begin(), points.end(), low, fps);
    std::greater<double> high;
    // Max height of lower foot in frames between `lows[i]` and `lows[i+1]` is stored at `highs[i]`.
    std::vector<double> highs;
    for (std::size_t i = 1; i < lows.size(); ++i) {
        std::size_t begin = lows[i - 1];
        std::size_t end = lows[i];
        model_points body = points[begin];
        double max = *get_height(body[body_part::l_ankle], body[body_part::r_ankle], low);
        for (++begin; begin < end; ++begin) {
            body = points[begin];
            std::optional<double> current = get_height(body[body_part::l_ankle], body[body_part::r_ankle], low);
            if (current && high(*current, max)) {
                max = *current;
            }
        }
        highs.push_back(max);
    }
    // Filter steps frames.
    if (lows.size()) {
        res.push_back(lows.front());
        double highest_possible;
        if (highs.size()) {
            highest_possible = highs.front();
        } else {
            model_points first = points[lows.front()];
            highest_possible = *get_height(first[body_part::l_ankle], first[body_part::r_ankle], low);
        }
        for (std::size_t i = 1; i < lows.size(); ++i) {
            model_points body = points[lows[i - 1]];
            double lower = *get_height(body[body_part::l_ankle], body[body_part::r_ankle], low);
            double higher = highs[i - 1];
            double center = (lower + higher) / 2.0;

            body = points[lows[i]];
            lower = *get_height(body[body_part::l_ankle], body[body_part::r_ankle], low);
            if (!low(lower, highest_possible)) break;
            if (low(lower, center)) {
                res.push_back(lows[i]);
            }
        }
    }
    return res;
}

std::optional<double> get_vertical_tilt_angle(const model_point &a, const model_point &b, tilt_angle_side tilt_side) {
    if (a && b) {
        double mult1 = 1;
        if (tilt_side == tilt_angle_side::bottom)
            mult1 = -1;
        cv::Point3d v1(0, 0, mult1 * 1); // Point straight up.
        cv::Point3d v2 = *b - *a;
        double mult2 = 1;
        if (v2.x != 0)
            mult2 = v2.x / std::abs(v2.x);
        double dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        double val = dot / (cv::norm(v1) * cv::norm(v2));
        return mult2 * std::acos(val) * 180.0 / M_PI;
    }
    return std::nullopt;
}

double get_horizontal_tilt_angle(const cv::Point3d &a, const cv::Point3d &b) {
    cv::Point3d v1(1, 0, 0);
    cv::Point3d v2 = b - a;
    double mult = 1;
    if (v2.x != 0)
        mult = v2.x / std::abs(v2.x);
    double dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    double val = dot / (cv::norm(v1) * cv::norm(v2));
    return mult * std::acos(val) * 180.0 / M_PI;
}

std::optional<double> get_vertical_tilt_angle(const frame_point &a, const frame_point &b) {
    if (a && b) {
        cv::Point2d v1(0, - 1); // Point straight up.
        cv::Point2d v2 = *b - *a;
        double dot = v1.x * v2.x + v1.y * v2.y;
        double val = dot / (cv::norm(v1) * cv::norm(v2));
        double mult = 1;
        if (v2.x != 0)
            mult = v2.x / std::abs(v2.x);
        return mult * std::acos(val) * 180.0 / M_PI;
    }
    return std::nullopt;
}

std::string get_output_dir(const std::string &video_filename) {
    std::stringstream path(video_filename);
    std::string part;
    std::vector<std::string> parts;
    std::string dir = "outputs";
    std::filesystem::create_directory(dir);
    dir += "/";
    while (std::getline(path, part, '/'))
        parts.push_back(part);
    if (parts.size()) {
        dir += parts.back().substr(0, parts.back().find("."));
    } else {
        // Could not parse parts of path to video filename.
        std::time_t now = std::time(nullptr);
        std::stringstream sstr;
        sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
        dir += sstr.str();
    }
    // Create directory.
    std::filesystem::create_directory(dir);
    return dir + "/";
}

bool is_inside(const cv::Rect &rect, const cv::Mat &frame) {
    return rect.tl().x >= 0 && rect.tl().y >= 0 &&
        rect.br().x <= frame.cols && rect.br().y <= frame.rows;
}

void draw_body(cv::Mat &frame, const frame_points &body) {
    for (std::size_t i = 0; i < NPAIRS; ++i) {
        std::size_t a_idx = PAIRS[i][0];
        std::size_t b_idx = PAIRS[i][1];
        frame_point a = body[a_idx];
        frame_point b = body[b_idx];

        // Check if points `a` and `b` are valid.
        if (a && b) {
            cv::Scalar c(0, 255, 255);
            if ((a_idx == body_part::l_ankle) || (a_idx == body_part::l_knee) || (a_idx == body_part::l_hip) ||
                (a_idx == body_part::l_wrist) || (a_idx == body_part::l_elbow) || (a_idx == body_part::l_shoulder) ||
                (b_idx == body_part::l_ankle) || (b_idx == body_part::l_knee) || (b_idx == body_part::l_hip) ||
                (b_idx == body_part::l_wrist) || (b_idx == body_part::l_elbow) || (b_idx == body_part::l_shoulder)) {
                    c = cv::Scalar(255, 0, 255);
            } else if ((a_idx == body_part::r_ankle) || (a_idx == body_part::r_knee) || (a_idx == body_part::r_hip) ||
                (a_idx == body_part::r_wrist) || (a_idx == body_part::r_elbow) || (a_idx == body_part::r_shoulder) ||
                (b_idx == body_part::r_ankle) || (b_idx == body_part::r_knee) || (b_idx == body_part::r_hip) ||
                (b_idx == body_part::r_wrist) || (b_idx == body_part::r_elbow) || (b_idx == body_part::r_shoulder)) {
                    c = cv::Scalar(255, 255, 0);
            }
            // Connect body parts with lines.
            cv::line(frame, *a, *b, c, 2);
        }
    }
    // Draw body parts.
    for (std::size_t i = 0; i < NPOINTS; ++i) {
        if (body[i])
            cv::circle(frame, *body[i], 2, cv::Scalar(0, 0, 255), -1);
    }
}

frame_points model_to_frame(const model_points &body) {
    frame_points res;
    for (const auto &p : body) {
        if (p) {
            res.push_back(cv::Point2d(p->x, p->z));
        } else {
            res.push_back(std::nullopt);
        }
    }
    return res;
}

cv::Mat resize(const cv::Mat &frame, std::size_t height) {
    cv::Mat res;
    std::size_t width = height * ((double)frame.cols / (double)frame.rows);
    cv::Size size(width, height);
    
    // Portrait mode.
    if (frame.cols < frame.rows) {
        size = cv::Size(height, height * ((double)frame.rows / (double)frame.cols));
    }
    
    cv::resize(frame, res, size);
    return res;
}

double area(const cv::Rect &r) {
    return r.width * r.height;
}

bool is_inside(const cv::Point2d &p, const cv::Rect &r) {
    return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}

cv::Rect rect(const cv::Mat &frame) {
    return cv::Rect(0, 0, frame.cols, frame.rows);
}

cv::Rect rect(const std::vector<cv::Rect> &rs) {
    if (!rs.size()) return cv::Rect();
    cv::Rect res = rs.front();
    for (const auto &r : rs) {
        res |= r;
    }
    return res;
}

std::vector<cv::Rect> split(const cv::Rect &bbox) {
    std::vector<cv::Rect> bboxes;
    cv::Point offset;
    cv::Rect smaller(bbox.tl(), bbox.size() / GRID_SIZE);
    for (std::size_t i = 0; i < GRID_SIZE; ++i) {
        offset = cv::Point(0, i * bbox.height / GRID_SIZE);
        for (std::size_t j = 0; j < GRID_SIZE; ++j) {
            offset = cv::Point(j * bbox.width / GRID_SIZE, offset.y);
            bboxes.push_back(smaller + offset);
        }
    }
    return bboxes;
}

double average_dist(const std::vector<cv::Rect> &rs) {
    int n_distances = 0;
    double res = 0.0;
    for (std::size_t i = 0; i < rs.size(); ++i)
        for (std::size_t j = i + 1; j < rs.size(); ++j, ++n_distances) {
            res += cv::norm(get_center(rs[i]) - get_center(rs[j]));
        }
    return (n_distances) ? res / n_distances : 0;
}