#!/usr/bin/env bash

# Run LR test for all normal networks
scripts/submit-lrtest.sh

# Run LR test for the Rand/Old LR test demo
scripts/submit-lrtest-demo.sh

# Run LR test for the OnePred/TwoPred SC networks
scripts/submit-sc-lrtest.sh

# Use results to determine appropriate LR ranges for each domain (and output plots)
python -m pointnn.eval.lr --output lr-sc.png --lroutput sc-lrs.json output/starcraft/*.pkl
python -m pointnn.eval.lr --output lr-we.png --lroutput we-lrs.json output/weather/*.pkl
python -m pointnn.eval.lr --output lr-tr.png --lroutput tr-lrs.json output/traffic/*.pkl

# Calculate LR ranges for each model, apply to each corresponding experiment file (to prep for training)
python -m pointnn.eval.lr apply --lrs sc-lrs.json experiments/sc-*.json
python -m pointnn.eval.lr apply --lrs we-lrs.json experiments/we-*.json
python -m pointnn.eval.lr apply --lrs tr-lrs.json experiments/tr-*.json
