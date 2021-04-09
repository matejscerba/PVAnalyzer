#include "forward.hpp"

const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

std::optional<cv::Point2d> operator+(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs) {
    if (lhs && rhs) return *lhs + *rhs;
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

cv::Point2d count_mean_delta(std::vector<cv::Point2d>::const_iterator begin, std::vector<cv::Point2d>::const_iterator end) noexcept {
    auto n = end - begin - 1;
    if (n > 0) {
        cv::Point2d sum = *(--end) - *begin;
        return sum / (double)n;
    }
    return cv::Point2d();
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

std::optional<double> get_height(const frame_part &a, const frame_part &b, std::function<bool (double, double)> compare) noexcept {
    if (a && b) {
        if (compare(a->y, b->y))
            return a->y;
        return b->y;
    } else if (a) {
        return a->y;
    } else if (b) {
        return b->y;
    }
    return std::nullopt;
}
