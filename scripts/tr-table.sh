#!/usr/bin/env bash

OUT=cluster/

python -m pointnn.eval.traffic.table $OUT/traffic-loss.pkl
