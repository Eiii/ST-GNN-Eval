#!/usr/bin/env sh

# Evaluate all models on the test data of all domains
scripts/run-loss.sh
# Use the results to generate the loss table 
# (Table 5)
scripts/losstable.sh

# Generate training performance plots for all domains+models 
# (Figures 3,7,9)
scripts/training-plots.sh

# Display weight functions learned by SC2 TPC models 
# (Figure 4)
scripts/sc-filters.sh

# Evaluate models on SC2 domains @ different prediction timesteps
scripts/run-sc-dist.sh
# Plot the resulting bar charts 
# (Figure 5)
scripts/sc-dist.sh

# Display attention weighting by SC2 TPCA model 
# (Figure 6)
scripts/sc-attn.sh

# Display traffic error table 
# (Table 6)
scripts/tr-table.sh

# Display traffic models' prediction over certain days 
# (Figure 8)
scripts/tr-day.sh

# Calculate weather models' dropout performance
# (Table 7)
scripts/we-ensemble.sh
