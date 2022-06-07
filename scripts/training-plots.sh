#!/usr/bin/env bash

FTYPE=pdf

SC=cluster/t-sc-final
TR=cluster/t-tr-final
WE=cluster/t-we-final
SIZES="Small Med Large"

for size in $SIZES ; do 
  python -m pointnn.eval.training $SC --filter-all $size --filter-ignore OnePred,TwoPred --out sc-${size}.${FTYPE}
done

for size in $SIZES ; do
  python -m pointnn.eval.training $TR --filter-all $size --filter-ignore NoSpace --out tr-${size}.${FTYPE} --smooth 20
done

for size in $SIZES ; do
  python -m pointnn.eval.training $WE --filter-all $size --filter-ignore Drop2 --out we-${size}.${FTYPE} --smooth 10
done
