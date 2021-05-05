#include <cstddef>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>

#include "video_processor.hpp"

void write_usage_msg(char *name) noexcept {
    std::cout <<
    "usage: " << name << " " << "--help | vid1 ... | -m mod1 ..." << std::endl <<
    "  vid1: path to video" << std::endl <<
    "  mod1: path to model file" << std::endl;
}

int main(int argc, char **argv) {

    video_processor vp;

    if (argc > 1) {
        if (std::string(argv[1]) == "--help") {
            write_usage_msg(argv[0]);
        } else if (std::string(argv[1]) == "-m") {
            for (std::size_t i = 2; i < argc; ++i) {
                if (std::filesystem::exists(argv[i])) {
                    vp.process_model(argv[i]);
                } else {
                    std::cout << "File \"" << argv[i] << "\" does not exists." << std::endl;
                }
            }
        } else {
            for (std::size_t i = 1; i < argc; ++i) {
                if (std::filesystem::exists(argv[i])) {
                    vp.process_video(argv[i]);
                } else {
                    std::cout << "File \"" << argv[i] << "\" does not exists." << std::endl;
                }
            }
        }
    }

    return 0;
}
