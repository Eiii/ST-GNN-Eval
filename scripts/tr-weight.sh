#!/usr/bin/env bash

NETS=cluster/t-tr-final
OUT=trweights
TPC=$(ls $NETS/TPC-Med\:* | head -1)

python -m pointnn.eval.traffic.weight --net $TPC --out $OUT
