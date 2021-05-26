#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse

# Prepare arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="Path to file containing parameters")
args = parser.parse_args()

# Read all data from file, returns filename of video, names and values of parameters.
def read_file(path):
    try:
        with open(path, "r") as file:
            lines = file.read().splitlines()
    except IOError:
        print("File {} could not be opened".format(args.file))
        return None

    if len(lines) < 3: return None

    # File was correctly opened and contains data.
    video = lines[0]
    names = lines[1].split(',')
    units = lines[2].split(',')
    values = [[] for _ in names]
    for i in range(3, len(lines)):
        row_vals = lines[i].split(',')
        for j in range(len(names)):
            if row_vals[j] == '':
                values[j].append(None)
            else:
                values[j].append(eval(row_vals[j]))
    return video, names, units, values

# Plot parameters.
def plot_params(vid_name, names, units, vals):
    shown = 0
    for p_name, p_unit, p_vals in zip(names[1:], units[1:], vals[1:]):
        if p_name == "Hips velocity loss" or p_name == "Shoulders velocity loss" or p_name == "Takeoff angle":
            # Don't plot this parameter.
            continue
        elif "Step" in p_name or "step" in p_name:
            # Plot as bar chart.
            valid_vals = [x for x in p_vals if x != None]
            plot_func = plt.bar
            x = range(1, len(valid_vals) + 1)
            y = valid_vals
            x_label = "Step number"
            y_label = "{} [{}]".format(p_name, p_unit)
        else:
            # Plot as frame-wise line segments chart.
            plot_func = plt.plot
            x = vals[0]
            y = p_vals
            x_label = names[0]
            y_label = "{} [{}]".format(p_name, p_unit)

        # Plot graph.
        if shown > 0:
            plt.figure()
        plt.title("{} : {}".format(vid_name, p_name))
        plot_func(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        shown += 1    
    
    plt.show()

if args.file != None:
    data = read_file(args.file)
    if data != None:
        video, names, units, values = data
        plot_params(video, names, units, values)
