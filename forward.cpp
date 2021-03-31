#include "forward.hpp"

const std::string protofile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";

const std::string caffemodel = "pose/mpi/pose_iter_160000.caffemodel";

std::string get_name(const parameter &p) {
    return std::get<0>(p);
}

std::vector<std::optional<double>> get_values(const parameter &p) {
    return std::get<1>(p);
}

std::optional<cv::Point2d> operator+(const std::optional<cv::Point2d> &lhs, const std::optional<cv::Point2d> &rhs) {
    if (lhs && rhs)
        return *lhs + *rhs;
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

cv::Point2d count_mean_delta(std::vector<cv::Point2d>::const_iterator begin, std::vector<cv::Point2d>::const_iterator end) {
    double n = end - begin - 1;
    if (n) {
        cv::Point2d sum = *(--end) - *begin;
        return sum / n;
    }
    return cv::Point2d();
}
