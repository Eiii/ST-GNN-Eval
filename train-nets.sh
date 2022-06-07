#!/usr/bin/env bash

# Train all normal networks
scripts/submit.sh

# Train OnePred/TwoPred SC networks
scripts/submit-sc-preds.sh

# Train Drop20 weather networks
scripts/submit-drop.sh
