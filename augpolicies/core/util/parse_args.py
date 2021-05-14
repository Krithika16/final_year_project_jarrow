import argparse
import os
import json
from augpolicies.core.util.parse_objects import parse_dataset, parse_list, parse_model, parse_aug



def get_args():
    parser = argparse.ArgumentParser(description='Script to run augpolicies')
    parser.add_argument('--hpc', action='store_true', help='Hypothesis testing aug comparison')
    parser.add_argument('--hpe', action='store_true', help='Hypothesis testing aug at end')
    parser.add_argument('--vis', action='store_true', help='Hypothesis testing visualise result')
    parser.add_argument('--rank', action='store_true', help='Ranks the full training sessions and gets the ranking through time')
    parser.add_argument('--data', default='fmnist', type=str, choices=['fmnist', 'cifar10'], help='Stores dataset.')
    parser.add_argument('-c', '--config', default='default.json', type=str, help='json config file')
    args = parser.parse_args()
    return args


def get_dataset_from_args():
    args = get_args()
    dataset = parse_dataset(args.data)
    return dataset


def get_config_json():
    args = get_args()
    config_path = os.path.join(os.getcwd(), 'data', 'configs', args.config)
    with open(config_path) as f:
        config = json.load(f)

    config['aug']['choices'] = parse_list(config['aug']['choices'], parse_aug)
    config['models'] = parse_list(config['models'], parse_model)
    return config
