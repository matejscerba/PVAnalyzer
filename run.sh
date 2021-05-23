#!/bin/bash
for FILE in data/*
do
./build/PVAnalyzer $FILE
done