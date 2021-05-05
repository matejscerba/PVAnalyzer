#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse

# Prepare arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="Path to file containing parameters")
args = parser.parse_args()

def read_file(path):
    try:
        with open(path, "r") as file:
            lines = file.read().splitlines()
    except IOError:
        print("File {} could not be opened".format(args.file))
        return None

    if len(lines) < 2: return None

    video = lines[0]
    names = lines[1].split(',')
    values = [[] for _ in names]
    for i in range(2, len(lines)):
        row_vals = lines[i].split(',')
        for j in range(len(names)):
            if row_vals[j] == '':
                values[j].append(None)
            else:
                values[j].append(eval(row_vals[j]))
    return video, names, values

def get_valid_values(values):
    return None

def plot_bars(x, y, x_name, y_name):
    pass

def plot_lines(x, y, x_name, y_name):
    pass

def plot_params(vid_name, names, vals):
    for p_name, p_vals in zip(names[1:], vals[1:]):
        plt.title("{} - {}".format(vid_name, p_name))
        if p_name == "Hips velocity loss" or p_name == "Shoulders velocity loss" or p_name == "Takeoff angle":
            continue
        elif "Step" in p_name or "step" in p_name:
            # Bar chart.
            valid_vals = get_valid_values(p_vals)
            plot_bars(range(len(valid_vals)), valid_vals, "Step", p_name)
            pass
        elif "tilt" in p_name:
            # Frame-wise bar chart.
            plot_bars(vals[0], p_vals, names[0], p_name)
            pass
        else:
            # Frame-wise line segments chart.
            plot_lines(vals[0], p_vals, names[0], p_name)
            pass
        plt.show()
    pass

if args.file != None:
    data = read_file(args.file)
    if data != None:
        video, names, values = data
        plot_params(video, names, values)
