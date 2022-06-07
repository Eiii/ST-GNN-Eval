#!/usr/bin/env bash

NUMS=$(seq 5)
DOMAINS="tr we sc"
NETS="tpc int gc tpca"
SIZES="small med large"

for c in $NUMS; do
  for d in $DOMAINS; do
    for n in $NETS; do
      for s in $SIZES ; do
        NAME=${d}-${n}-${s}
        scripts/run-experiment.sh $NAME $c
      done
    done
  done
done
