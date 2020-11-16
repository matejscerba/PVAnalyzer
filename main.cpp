#include "video_processor.hpp"

int main(int argc, char **argv) {

    video_processor vp;

    for (int i = 1; i < argc; i++) {
        vp.process(argv[i]);
    }

    return 0;
}