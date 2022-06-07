#!/usr/bin/env bash

NUMS=$(seq 3)
NETS="int tpc gc"
SIZES="med"
POST="onepred twopred"

for c in $NUMS; do
  for n in $NETS; do
    for p in $POST ; do
      NAME=sc-${n}-med-${p}
      scripts/run-experiment.sh $NAME $c
    done
  done
done
