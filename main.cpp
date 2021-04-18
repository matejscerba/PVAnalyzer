#include <string>

#include "video_processor.hpp"

int main(int argc, char **argv) {

    video_processor vp;

    for (int i = 1; i < argc; ++i) {
        if ((std::string(argv[i]) == "--model") && (++i < argc)) {
            vp.process_model(argv[i]);
        } else {
            vp.process_video(argv[i]);
        }
    }

    return 0;
}
