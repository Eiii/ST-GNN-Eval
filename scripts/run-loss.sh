#!/usr/bin/env bash

OUT=/nfs/stak/users/merriler/cluster/hpc-share/output

SCDATA=/nfs/stak/users/merriler/cluster/hpc-share/data/sc2scene/test
SCNETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-sc-final
python -m pointnn.eval.sc2.loss --data $SCDATA --out $OUT/sc-loss.pkl --net $SCNETS/*.pkl --bs 32

WEDATA=/nfs/stak/users/merriler/cluster/hpc-share/data/weather
WENETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-we-final
python -m pointnn.eval.weather.loss --data $WEDATA --out $OUT/we-loss.pkl --nets $WENETS/*.pkl --bs 32

TRDATA=/nfs/stak/users/merriler/cluster/hpc-share/data/traffic/METR-LA
TRNETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-tr-final
python -m pointnn.eval.weather.loss --data $TRDATA --out $OUT/tr-loss.pkl --nets $TRNETS/*.pkl --bs 32
