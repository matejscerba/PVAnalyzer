#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

class video_processor {

public:

    void process(std::string filename) {
        cv::Mat image = cv::imread(filename);
        std::cout << filename << std::endl;
    }

};