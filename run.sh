#!/bin/bash
for FILE in data/BP_videos/*
do
./build/PVAnalyzer $FILE
done