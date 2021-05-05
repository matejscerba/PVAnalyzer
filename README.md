## Technologies
Project is created with:
* OpenCV version: 4.5.1

Correct functionality requires models for body parts detections. You can download them running script `models.sh`.
Note that `models.sh` uses `wget` package.

## Setup
To run this project, install it by running script `install.sh`.

This creates executable file `build/PVAnalyzer`.

## Usage
Program can be run in two modes:
* Video analyzer
* Model analyzer

To show help, use single argument `--help`.

### Video analyzer
To analyze videos, pass videos' paths as command line arguments:

```
$ ./build/PVAnalyzer vid1 vid2 vid3
```

Command above analyzes videos `vid1`, `vid2` and `vid3`. You can pas any number of videos to be analyzed sequentially.
Video analyzer mode creates output files in folder `outputs/X/`, where `X` is name of analyzed video (without path and extension).

### Model analyzer
To analyze saved models, pass models' paths as command line arguments:

```
$ ./build/PVAnalyzer -m mod1 mod2 mod3
```

Command above analyzes models `mod1`, `mod2` and `mod3`. You can pas any number of models to be analyzed sequentially.
Model analyzer mode does not create any output files.
