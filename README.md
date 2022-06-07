# Stake-free Evaluation of Graph Networks for Spatio-Temporal Processes

This is the code repo associated with the paper 'Stake-free Evaluation of Graph Networks for Spatio-Temporal Processes'.
All the models evaluated and figures included in that work are derived from this codebase.

A set of scripts is provided to recreate the experiments demonstrated in the paper. See **Scripts** below for more details.

## Scripts

A set of scripts is provided to configure the models, train the models, evaluate their performance, and generate figures demonstrating them. 
These scripts are organized hierarchically, and generally just execute individual experiments defined in the `scripts/` folder.

`run-all.sh` executes everything in order, specifically:
- `train-lrtests.sh` starts the learning rate test training runs, evaluates their results, and sets the learning rate for each experiment appropriately. 
- `train-nets.sh` starts the training runs for all models evaluated in this work.
_ `generate-plots.sh` uses the artifacts from the completed training runs to evaluate the models, and generate figures demonstrating them.

These scripts are set up to be as simple as possible, and execute each training / evaluation / plotting action sequentially.
Running them as provided will result in an unrealistically long runtime to train each network, etc.

Instead, you should open up the scripts to pull out the specific experiments you're interested in running, and execute them manually.
Additionally, you could modify `scripts/run-experiment.sh` so that it submits an appropriate job to a GPU cluster rather than training each model sequentially.

## Installation/Requirements

`requirements.txt` lists the python dependencies necessary to run this module.
Install them via `pip install -r requirements.txt`

## Python Module Layout

##### `pointnn/nets` 
Contains the implementations of each graph network primitive we evaluate. 
The folders `pointnn/nets/starcraft`, etc contain the network implementations applying these primitives to each problem domain.

##### `pointnn/data` 
Contains code to load + process the dataset associated with each problem domain.

##### `pointnn/problem`
Problem domain definitions.

##### `pointnn/eval`
Code to evaluate trained models' performance and generate figures for the paper.

