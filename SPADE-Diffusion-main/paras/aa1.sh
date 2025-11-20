#!/bin/sh
i=1
while [ "$i" -le 300 ]; do
    filename=$(printf "spatial%03d.txt" "$i")
    cp spatialbak.txt "$filename"
    i=$((i+1))
done
