#!/usr/bin/bash

SIZE="s m l"
BRO="1_01 1_05 1_10 1_50"
TEST="1 2 3 4 5"

for s in $SIZE; do
    for b in $BRO; do
        for t in $TEST; do
            echo "${s} ${b} ${t}"
        done
    done
done

