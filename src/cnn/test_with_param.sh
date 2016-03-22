#!/bin/sh
#1
for i in `seq 1 $1` 
do
	 python $2 $3 $4
	 echo "shell step:"$i
done

