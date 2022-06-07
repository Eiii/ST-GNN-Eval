#!/usr/bin/env bash

DATA=cluster/sc2-tpc-dist-*
TYPES="TPC Int GC"

for T in $TYPES ; do
  python -m pointnn.eval.sc2.loss plot $DATA --filter $T --out ${T}-dist.pdf
done


