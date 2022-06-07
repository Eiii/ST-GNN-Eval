#!/usr/bin/env bash

NUMS=$(seq 3)
DOMAINS="sc tr we"
SIZES="small med large"
NETS="tpc int gc"

for c in $NUMS; do
  for d in $DOMAINS; do
    for s in $SIZES ; do
      for n in $NETS; do
        NAME=${d}-${n}-${s}-lrtest
        scripts/run-experiment.sh $NAME $c
      done
    done
  done
done
