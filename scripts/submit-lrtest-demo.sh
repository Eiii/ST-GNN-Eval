#!/usr/bin/env bash

NUMS=$(seq 5)
POST="demo rand old"

for c in $NUMS; do
  for p in $POST; do
    NAME=tr-tpc-med-lrtest-${p}
    scripts/run-experiment.sh $NAME $c
  done
done
