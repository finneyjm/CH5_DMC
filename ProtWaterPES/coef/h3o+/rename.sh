#!/bin/bash

for x in *.out
do
  base=${x%.out}
  cp $x $base.dat
done
echo rename
