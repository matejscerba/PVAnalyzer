#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include "forward.hpp"

const std::string PROTOFILE = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string CAFFEMODEL = "pose/mpi/pose_iter_160000.caffemodel";

frame_part operator+(const frame_part &lhs, const frame_part &rhs) noexcept {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

frame_part operator-(const frame_part &lhs, const frame_part &rhs) noexcept {
    if (lhs && rhs) return *lhs - *rhs;
    return std::nullopt;
}

frame_part operator/(const frame_part &lhs, double rhs) noexcept {
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

std::ostream& operator<<(std::ostream& os, const model_point &p) noexcept {
    if (p) {
        os << p->x << "," << p->y << "," << p->z;
    } else {
        os << ",,";
    }
    return os;
}

cv::Point2d get_center(const cv::Rect &rect) noexcept {
    return cv::Point2d((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
}

offset count_mean_delta(offsets::const_iterator begin, offsets::const_iterator end) noexcept {
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

std::optional<double> distance(const frame_part &a, const frame_part &b) noexcept {
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

std::vector<std::size_t> get_frame_numbers( std::vector<model_body>::const_iterator begin,
                                            std::vector<model_body>::const_iterator end,
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

std::vector<std::size_t> get_step_frames(const model_video_body &points) noexcept {
    std::vector<std::size_t> res;
    std::less<double> low;
    std::vector<std::size_t> lows = get_frame_numbers(points.begin(), points.end(), low);
    std::greater<double> high;
    std::vector<std::size_t> highs = get_frame_numbers(points.begin(), points.end(), high);
    double center = 0;
    for (std::size_t i = 0; i < std::min(lows.size(), highs.size()); ++i) {
        model_body l_body = points[lows[i]];
        model_body h_body = points[highs[i]];
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

std::optional<double> get_vertical_tilt_angle(const frame_part &a, const frame_part &b) noexcept {
    if (a && b) {
        double y = a->y - b->y;
        double x = a->x - b->x;
        return std::atan(x / y) * 180.0 / M_PI;
    }
    return std::nullopt;
}

std::string create_output_filename() noexcept {
    std::time_t now = std::time(nullptr);
    std::stringstream sstr;
    sstr << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
    return sstr.str();
}

bool is_inside(const cv::Rect &rect, const cv::Mat &frame) noexcept {
    return rect.tl().x >= 0 && rect.tl().y <= 0 &&
        rect.br().x <= frame.cols && rect.br().y <= frame.rows;
}

void draw_body(cv::Mat &frame, const frame_body &body) noexcept {
    for (std::size_t i = 0; i < NPAIRS; ++i) {
        std::size_t a_idx = PAIRS[i][0];
        std::size_t b_idx = PAIRS[i][1];
        std::optional<cv::Point2d> a = body[a_idx];
        std::optional<cv::Point2d> b = body[b_idx];

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

frame_body model_to_frame(const model_body &body) noexcept {
    frame_body res;
    for (const auto &p : body) {
        if (p) {
            res.push_back(cv::Point2d(p->x, p->z));
        }
        res.push_back(std::nullopt);
    }
    return res;
}
