#!/usr/bin/env bash

DATA=/nfs/stak/users/merriler/cluster/hpc-share/data/weather
OUT=/nfs/stak/users/merriler/cluster/hpc-share/output/
TGTS="TPC-Med TGC-Med TInt-Med"  
BASE=/nfs/stak/users/merriler/cluster/hpc-share/output/t-we-final
NETS=""

for tgt in $TGTS ; do
  N=$(ls ${BASE}/${tgt}\:* | head -1)
  D=$(ls ${BASE}/${tgt}-Drop2\:* | head -1)
  NETS="${NETS} $N $D"
done

echo $NETS

for d in 0 0.1 0.2 0.5 ; do
  echo $d
  python -m pointnn.eval.weather.loss --data $DATA --bs 32 --net $NETS --out $OUT/weather-drop-${d/./}.pkl --drop $d
done
