#!/usr/bin/env bash

NETS=cluster/t-tr-final
OUT=day
TPC=$(ls $NETS/TPC-Med\:* | head -1)
INT=$(ls $NETS/TInt-Med\:* | head -1)
MONTH=5
DAY=20
FTYPE=pdf

python -m pointnn.eval.traffic.day --net $TPC $INT --month $MONTH --day $DAY --out $OUT --ftype $FTYPE
