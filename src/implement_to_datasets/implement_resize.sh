#!/bin/sh

for i in *.jpg
do
	python resize.py $i
done
