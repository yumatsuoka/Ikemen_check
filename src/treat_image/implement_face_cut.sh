#!/bin/sh

#n ${files[@]}
#do
#	python cut_face.py $file
#done
#exit 0

for i in *.jpg
do
	python cut_face.py $i
done
