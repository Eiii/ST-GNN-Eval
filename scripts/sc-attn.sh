#!/usr/bin/env bash

DATA=data/sc2scene/test
F="50 250"
NET=$(ls cluster/t-sc-final/TPCA-Med* | head -1)
FTYPE=pdf

for f in $F ; do
  OUT=sc2attn${f}
  python -m pointnn.eval.sc2.attn --data $DATA --net $NET --out $OUT --frame $f --ftype $FTYPE
done

