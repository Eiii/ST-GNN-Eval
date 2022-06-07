#!/usr/bin/env bash

DATA=/nfs/stak/users/merriler/cluster/hpc-share/data/sc2scene/test
NETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-sc-final
OUT=/nfs/stak/users/merriler/cluster/hpc-share/output

PREDS="$NETS/*OnePred* $NETS/*TwoPred*"
BASE="$NETS/TPC-Med\:9* $NETS/TInt-Med\:0* $NETS/TGC-Med\:5*"

echo $PREDS
echo $BASE

for t in $(seq 12) ; do
  echo $t
  python -m pointnn.eval.sc2.loss --data $DATA --bs 150 --net $PREDS $BASE --out $OUT/sc2-tpc-dist-${t}.pkl --t $t
done
