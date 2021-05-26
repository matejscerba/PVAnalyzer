#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse

# Prepare arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--fir", default=None, type=str, help="Path to file containing parameters")
parser.add_argument("--sec", default=None, type=str, help="Path to file containing parameters")
parser.add_argument("--thi", default=None, type=str, help="Path to file containing parameters")
parser.add_argument("--fou", default=None, type=str, help="Path to file containing parameters")
parser.add_argument("--fif", default=None, type=str, help="Path to file containing parameters")
args = parser.parse_args()

def read_file(path):
    try:
        with open(path, "r") as file:
            lines = file.read().splitlines()
    except IOError:
        print("File {} could not be opened".format(args.file))
        return None

    points = [[[],[]] for _ in range(15)]

    for i in range(0, len(lines) // 17):
        fx, _, fy = [eval(x) if x != "" else None for x in lines[1 + i * 17 + 1].split(',')]
        for j in range(2, 17):
            x, _, y = [eval(x) if x != "" else None for x in lines[1 + i * 17 + j].split(',')]
            if (x != None):
                points[j - 2][0].append(- fx - x)
                points[j - 2][1].append(- fy - y)
            else:
                points[j - 2][0].append(None)
                points[j - 2][1].append(None)
    
    return points

def plot(first, second, third, fourth, fifth):
    for i in range(len(first)):
        if i == 0:
            title = "Hlava"
        elif i == 1:
            title = "Krk"
        elif i == 2:
            title = "Pravé rameno"
        elif i == 3:
            title = "Pravý loket"
        elif i == 4:
            title = "Pravé zápěstí"
        elif i == 5:
            title = "Levé rameno"
        elif i == 6:
            title = "Levý loket"
        elif i == 7:
            title = "Levé zápěstí"
        elif i == 8:
            title = "Pravá kyčel"
        elif i == 9:
            title = "Pravé koleno"
        elif i == 10:
            title = "Pravý kotník"
        elif i == 11:
            title = "Levá kyčel"
        elif i == 12:
            title = "Levé koleno"
        elif i == 13:
            title = "Levý kotník"
        elif i == 14:
            title = "Hrudník"

        if i > 0:
            plt.figure()
        plt.title("Posun úvodního rámečku - data/23.MOV: {}".format(title))
        plt.plot(first[i][0], first[i][1], label="Bez posunu")
        if second != None:
            plt.plot(second[i][0], second[i][1], label="Posun nahoru o 25 % výšky")
        if third != None:
            plt.plot(third[i][0], third[i][1], label="Posun doprava o 25 % šířky")
        if fourth != None:
            plt.plot(fourth[i][0], fourth[i][1], label="Posun dolů o 25 % výšky")
        if fifth != None:
            plt.plot(fifth[i][0], fifth[i][1], label="Posun doleva o 25 % šířky")
        plt.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)
        plt.xlabel("x [px]")
        plt.ylabel("z [px]")
        plt.savefig("shift/{}.pdf".format(title))


    # plt.show()

if args.fir != None:
    first = read_file(args.fir)
    second = None
    third = None
    fourth = None
    first = [[
        [x -first[10][0][192] if x != None else None for x in first[i][0]],
        [y - first[10][1][192] if y != None else None for y in first[i][1]]
    ] for i in range(len(first))]
    if args.sec != None:
        second = read_file(args.sec)
        second = [[
            [x - second[10][0][193] if x != None else None for x in second[i][0]],
            [y - second[10][1][193] if y != None else None for y in second[i][1]]
        ] for i in range(len(second))]
        if args.thi != None:
            third = read_file(args.thi)
            third = [[
                [x - third[10][0][191] if x != None else None for x in third[i][0]],
                [y - third[10][1][191] if y != None else None for y in third[i][1]]
            ] for i in range(len(third))]
            if args.fou != None:
                fourth = read_file(args.fou)
                fourth = [[
                    [x - fourth[13][0][192] if x != None else None for x in fourth[i][0]],
                    [y - fourth[13][1][192] if y != None else None for y in fourth[i][1]]
                ] for i in range(len(fourth))]
                if args.fif != None:
                    fifth = read_file(args.fif)
                    fifth = [[
                        [x - fifth[13][0][190] if x != None else None for x in fifth[i][0]],
                        [y - fifth[13][1][190] if y != None else None for y in fifth[i][1]]
                    ] for i in range(len(fifth))]
    plot(first, second, third, fourth, fifth)

