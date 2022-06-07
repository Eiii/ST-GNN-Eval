#!/usr/bin/env bash

SC=cluster/t-sc-final
OUT=scweights
SIZES="Small Med Large"
FTYPE=png


for s in $SIZES; do
  NETS=($(ls $SC/TPC-$s\:*))
  NET=${NETS[0]}
  DEST=$OUT/$s
  mkdir -p $DEST
  for l in $(seq 0 2) ; do 
    python -m pointnn.eval.sc2.weight --net $NET --layer $l --out $DEST --ftype $FTYPE
  done
done

exit

