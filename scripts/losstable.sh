#!/usr/bin/env bash

OUT=/nfs/stak/users/merriler/cluster/hpc-share/output
FIGURES=./figures

python -m pointnn.eval.loss --weather $OUT/we-loss.pkl --sc2 $OUT/sc-loss.pkl --traffic $OUT/tr-loss.pkl --out $OUT/all-loss.pkl
python -m pointnn.eval.loss table $OUT/all-loss.pkl > $FIGURES/losstable.tex
