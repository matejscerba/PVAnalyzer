# Pole Vault Video Analyzer

## Table of contents
* [Technologies](#technologies)
* [Setup](#technologies)
* [Usage](#usage)
* [Plotting](#plotting)

## Technologies
Project is created with:
* OpenCV version: 4.5.1
* OpenCV_contrib modules: tracking (it is recommended to install all modules)
* cmake version: 3.1 or later
* python version: 3.8.2 (older versions should work as well)
* matplotlib version: 3.3.2 (older versions should work as well)
* argparse version: 1.1 (older versions should work as well)

[OpenCV installation guide for Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
[OpenCV installation guide for Windows](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html)
Note: Make sure to download `OpenCV_contrib` modules as well.

Correct functionality requires models for body parts detections. You can download them running script `models.sh`.
Note that `models.sh` uses `wget` package.

## Setup
To run this project, install it by running script `install.sh`.

This creates executable file `build/PVAnalyzer`.

## Usage
Program can be run in two modes:
* Video analyzer
* Model analyzer

After video or model is analyzed, you will be able to view all video frames with detections drawed in them.
Navigate through frames using left and right arrows, parameters for each frame will be written to console.
Note that only parameters valid for viewed frame will be written, for example velocity losses computed during
takeoff will be written only in frame showing sthlete's takeoff.
You can turn detections drawings off/on by pressing space bar. To end viewer, pres escape key.

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

## Plotting
Output parameters can be ploted as graphs using `plot_params.py` script.

To plot parameters saved to file `output.csv`, run the following command:

```
$ python3 plot_params.py --file output.csv
```

This script will show graphs of all parameters stored in file `output.csv`.