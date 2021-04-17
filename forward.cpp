#include "forward.hpp"

const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

std::optional<cv::Point2d> operator+(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs) {
    if (lhs && rhs) return *lhs + *rhs;
    return std::nullopt;
}

std::optional<cv::Point2d> operator-(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs) {
    if (lhs && rhs) return *lhs - *rhs;
    return std::nullopt;
}

std::optional<cv::Point2d> operator/(const std::optional<cv::Point2d> &lhs, double rhs) {
    if (lhs) return *lhs / rhs;
    return std::nullopt;
}

video_body operator+(const video_body &&lhs, const video_body &&rhs) {
    video_body res = std::move(lhs);
    res.insert(res.end(), rhs.begin(), rhs.end());
    return res;
}

std::optional<double> operator*(double lhs, const std::optional<double> &rhs) noexcept {
    if (rhs) return lhs * (*rhs);
    return std::nullopt;
}

std::vector<cv::Point2d> get_corners(const cv::Rect &rect) {
    return {
        rect.tl(), cv::Point2d(rect.br().x, rect.tl().y),
        cv::Point2d(rect.tl().x, rect.br().y), rect.br()
    };
}

cv::Point get_center(const cv::Mat &frame) {
    return cv::Point(frame.cols / 2, frame.rows / 2);
}

cv::Point2d get_center(const cv::Rect &rect) {
    return cv::Point2d((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
}

std::optional<cv::Point2d> get_center(const std::optional<cv::Rect> &rect){
    if (rect) return get_center(*rect);
    return std::nullopt;
}

std::optional<cv::Point2d> count_mean_delta(std::vector<std::optional<cv::Point2d>>::const_iterator begin, std::vector<std::optional<cv::Point2d>>::const_iterator end) noexcept {
    auto n = end - begin - 1;
    if (n > 0 && *(--end) && *begin) {
        cv::Point2d sum = **end - **begin;
        return sum / (double)n;
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
}

std::optional<double> distance(const std::optional<cv::Point2d> &a, const std::optional<cv::Point2d> &b) noexcept {
    if (a && b) {
        return std::sqrt((a->x - b->x) * (a->x - b->x) + (a->y - b->y) * (a->y - b-> y));
    }
    return std::nullopt;
}

frame_part get_part(const frame_part &a, const frame_part &b, std::function<bool (double, double)> compare) noexcept {
    if (a && b) {
        if (compare(a->y, b->y))
            return a;
        return b;
    } else if (a) {
        return a;
    } else if (b) {
        return b;
    }
    return std::nullopt;
}

std::optional<double> get_height(const frame_part &a, const frame_part &b, std::function<bool (double, double)> compare) noexcept {
    frame_part p = get_part(a, b, compare);
    if (p) return p->y;
    return std::nullopt;
}

std::vector<std::size_t> get_frame_numbers( std::vector<frame_body>::const_iterator begin,
                                            std::vector<frame_body>::const_iterator end,
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

std::vector<std::size_t> get_step_frames(const video_body &points) noexcept {
    std::vector<std::size_t> res;
    std::greater<double> low;
    std::vector<std::size_t> lows = get_frame_numbers(points.begin(), points.end(), low);
    std::less<double> high;
    std::vector<std::size_t> highs = get_frame_numbers(points.begin(), points.end(), high);
    double center = 0;
    for (std::size_t i = 0; i < std::min(lows.size(), highs.size()); ++i) {
        frame_body l_body = points[lows[i]];
        frame_body h_body = points[highs[i]];
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

bool is_inside(const std::vector<cv::Point2d> &corners, const cv::Mat &frame) noexcept {
    for (const auto &corner : corners) {
        if (corner.x < 0.0 || corner.x > (double)frame.cols ||
            corner.y < 0.0 || corner.y > (double)frame.rows) {
                return false;
        }
    }
    return true;
}
