#include <cstddef>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>

#include "video_processor.hpp"

/**
 * @brief Write usage message for user.
 * 
 * @param name Name of program.
 */
void write_usage_msg(char *name) {
    std::cout <<
    "usage: " << name << " " << "--help | [-f] vid1 ... | -m mod1 ..." << std::endl <<
    "  -f: if automatic athlete detection should be performed" << std::endl <<
    "  vid1: path to video" << std::endl <<
    "  mod1: path to model file" << std::endl;
}

/**
 * @brief Entry point of application.
 * 
 * Process parameters and pass arguments to analyzer.
 */
int main(int argc, char **argv) {

    video_processor vp;

    if (argc > 1) {
        if (std::string(argv[1]) == "--help") {
            write_usage_msg(argv[0]);
            return 0;
        } else if (std::string(argv[1]) == "-m") {
            for (std::size_t i = 2; i < argc; ++i) {
                if (std::filesystem::exists(argv[i])) {
                    vp.process_model(argv[i]);
                } else {
                    std::cout << "File \"" << argv[i] << "\" does not exists." << std::endl;
                }
            }
        } else {
            bool find = false;
            std::size_t begin = 1;
            if (std::string(argv[1]) == "-f") {
                find = true;
                ++begin;
            }
            for (std::size_t i = begin; i < argc; ++i) {
                if (std::filesystem::exists(argv[i])) {
                    vp.process_video(argv[i], find);
                } else {
                    std::cout << "File \"" << argv[i] << "\" does not exists." << std::endl;
                }
            }
        }
    }

    return 0;
}
