#!/bin/bash
for FILE in data/*
do
./build/PVAnalyzer -f $FILE
done