from . import train
from . import experiment

import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    for exp_args in experiment.read_experiment_json(args.config):
        train.train_single(**exp_args)
