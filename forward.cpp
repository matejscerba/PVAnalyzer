#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

#include "forward.hpp"

const std::string PROTOFILE = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string CAFFEMODEL = "pose/mpi/pose_iter_160000.caffemodel";

frame_point operator+(const frame_point &lhs, const frame_point &rhs) noexcept {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

frame_point operator-(const frame_point &lhs, const frame_point &rhs) noexcept {
    if (lhs && rhs) return *lhs - *rhs;
    return std::nullopt;
}

frame_point operator/(const frame_point &lhs, double rhs) noexcept {
    if (lhs) return *lhs / rhs;
    return std::nullopt;
}

model_point operator+(const model_point &lhs, const model_point &rhs) noexcept {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

model_point operator-(const model_point &p) noexcept {
    if (p) return -*p;
    return std::nullopt;
}

model_point operator-(const model_point &lhs, const model_point &rhs) noexcept {
    return lhs + (- rhs);
}

model_point operator/(const model_point &lhs, double rhs) noexcept {
    if (lhs) return *lhs / rhs;
    return std::nullopt;
}

std::optional<double> operator*(double lhs, const std::optional<double> &rhs) noexcept {
    if (rhs) return lhs * (*rhs);
    return std::nullopt;
}

cv::Size operator*(double lhs, const cv::Size &rhs) noexcept {
    return cv::Size(lhs * rhs.width, lhs * rhs.height);
}

std::ostream& operator<<(std::ostream& os, const model_point &p) noexcept {
    if (p) {
        os << p->x << "," << p->y << "," << p->z;
    } else {
        os << ",,";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const frame_point &p) noexcept {
    if (p) {
        os << p->x << "," << p->y;
    } else {
        os << ",";
    }
    return os;
}

cv::Point2d get_center(const cv::Rect &rect) noexcept {
    return cv::Point2d((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
}

frame_point count_mean_delta(frame_points::const_iterator begin, frame_points::const_iterator end) noexcept {
    auto n = end - begin - 1;
    if (n > 0 && *(--end) && *begin) {
        cv::Point2d sum = **end - **begin;
        return sum / (double)n;
    }
    return std::nullopt;
}

std::string body_part_name(const body_part part) noexcept {
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
}

std::optional<double> distance(const frame_point &a, const frame_point &b) noexcept {
    if (a && b) {
        return cv::norm(*a - *b);
    }
    return std::nullopt;
}

std::optional<double> distance(const model_point &a, const model_point &b) noexcept {
    if (a && b) {
        return cv::norm(*a - *b);
    }
    return std::nullopt;
}

model_point get_part(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) noexcept {
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

std::optional<double> get_height(const model_point &a, const model_point &b, std::function<bool (double, double)> compare) noexcept {
    model_point p = get_part(a, b, compare);
    if (p) return p->z;
    return std::nullopt;
}

std::vector<std::size_t> get_frame_numbers( std::vector<model_points>::const_iterator begin,
                                            std::vector<model_points>::const_iterator end,
                                            std::function<bool (double, double)> compare) noexcept {
    std::vector<std::size_t> res;
    std::optional<double> last_height = std::nullopt;
    std::size_t index = 0;
    bool correct_diff = false;
    for (; begin != end; ++begin) {
        std::optional<double> height = get_height((*begin)[body_part::l_ankle], (*begin)[body_part::r_ankle], compare);
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
        ++index;
    }
    return res;
}

std::vector<std::size_t> get_step_frames(const model_video_points &points) noexcept {
    std::vector<std::size_t> res;
    std::less<double> low;
    std::vector<std::size_t> lows = get_frame_numbers(points.begin(), points.end(), low);
    std::greater<double> high;
    std::vector<std::size_t> highs = get_frame_numbers(points.begin(), points.end(), high);
    double center = 0;
    for (std::size_t i = 0; i < std::min(lows.size(), highs.size()); ++i) {
        model_points l_body = points[lows[i]];
        model_points h_body = points[highs[i]];
        double l = *get_height(l_body[body_part::l_ankle], l_body[body_part::r_ankle], low);
        double h = *get_height(h_body[body_part::l_ankle], h_body[body_part::r_ankle], high);
        if (i > 0) {
            if (high(l, center)) continue; // Low point is above center of previous points.
            if (low(h, center)) continue;  // High point is below center of previous points.
        }
        center = (l + h) / 2.0;
        res.push_back(lows[i]);
    }
    return res;
}

std::optional<double> get_vertical_tilt_angle(const model_point &a, const model_point &b) noexcept {
    if (a && b) {
        double z = a->z - b->z;
        double x = a->x - b->x;
        return std::atan(x / z) * 180.0 / M_PI;
    }
    return std::nullopt;
}

std::optional<double> get_vertical_tilt_angle(const frame_point &a, const frame_point &b) noexcept {
    if (a && b) {
        double y = a->y - b->y;
        double x = a->x - b->x;
        return std::atan(x / y) * 180.0 / M_PI;
    }
    return std::nullopt;
}

std::string get_output_dir(const std::string &video_filename) noexcept {
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
    return dir;
}

bool is_inside(const cv::Rect &rect, const cv::Mat &frame) noexcept {
    return rect.tl().x >= 0 && rect.tl().y >= 0 &&
        rect.br().x <= frame.cols && rect.br().y <= frame.rows;
}

void draw_body(cv::Mat &frame, const frame_points &body) noexcept {
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

frame_points model_to_frame(const model_points &body) noexcept {
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

cv::Mat resize(const cv::Mat &frame, std::size_t height) noexcept {
    cv::Mat res;
    std::size_t width = height * ((double)frame.cols / (double)frame.rows);
    cv::Size size(width, height);
    if (frame.cols < frame.rows) {
        size = cv::Size(height, height * ((double)frame.rows / (double)frame.cols));
    }
    cv::resize(frame, res, size);
    return res;
}

double area(const cv::Rect &r) noexcept {
    return r.width * r.height;
}

bool is_inside(const cv::Rect &r, const cv::Rect &s) noexcept {
    return cv::norm(s.tl() - s.br()) / cv::norm(r.tl() - r.br()) >= 1.1 &&
        r.tl().x >= s.tl().x && r.tl().y >= s.tl().y && r.br().x <= s.br().x && r.br().y <= s.br().y;
}

bool is_inside(const cv::Point2d &p, const cv::Rect &r) noexcept {
    return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}

cv::Rect rect(const cv::Mat &frame) noexcept {
    return cv::Rect(0, 0, frame.cols, frame.rows);
}

cv::Rect rect(const std::vector<cv::Rect> &rs) noexcept {
    if (!rs.size()) return cv::Rect();
    cv::Rect res = rs.front();
    for (const auto &r : rs) {
        res |= r;
    }
    return res;
}

std::vector<cv::Rect> split(const cv::Rect &bbox) noexcept {
    std::vector<cv::Rect> bboxes;
    cv::Point offset;
    cv::Rect smaller = bbox - cv::Size(2 * bbox.width / 3, 2 * bbox.height / 3);
    for (std::size_t i = 0; i < 3; ++i) {
        offset = cv::Point(0, i * bbox.height / 3);
        for (std::size_t j = 0; j < 3; ++j) {
            offset = cv::Point(j * bbox.width / 3, offset.y);
            bboxes.push_back(smaller + offset);
        }
    }
    return bboxes;
}

double average_dist(const std::vector<cv::Rect> &rs) noexcept {
    int n_distances = 0;
    double res = 0.0;
    for (std::size_t i = 0; i < rs.size(); ++i)
        for (std::size_t j = i + 1; j < rs.size(); ++j, ++n_distances) {
            res += cv::norm(get_center(rs[i]) - get_center(rs[j]));
        }
    return (n_distances) ? res / n_distances : 0;
}