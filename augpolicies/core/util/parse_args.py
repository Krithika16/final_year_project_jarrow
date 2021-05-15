import argparse
import os
import json
import random_name
from datetime import datetime
from augpolicies.core.util.parse_objects import parse_dataset, parse_list, parse_model, parse_aug, parse_strategy


def get_args():
    parser = argparse.ArgumentParser(description='Script to run augpolicies')
    parser.add_argument('--hpc', action='store_true', help='Hypothesis testing aug comparison')
    parser.add_argument('--hpe', action='store_true', help='Hypothesis testing aug at end')
    parser.add_argument('--vis', action='store_true', help='Hypothesis testing visualise result')
    parser.add_argument('--rank', action='store_true', help='Ranks the full training sessions and gets the ranking through time')
    parser.add_argument('--data', default='fmnist', type=str, choices=['fmnist', 'cifar10'], help='Stores dataset.')
    parser.add_argument('-c', '--config', default='default.json', type=str, help='json config file')
    parser.add_argument('--name', default=random_name.generate(1)[0], type=str, help='random name to store the logs')
    args = parser.parse_args()
    args.config_path = args.config
    args.config = get_config_json(args)
    args.dataset = get_dataset_from_args(args)
    return args


def get_dataset_from_args(args):
    dataset = parse_dataset(args.data)
    return dataset


def get_config_json(args):
    config_path = os.path.join(os.getcwd(), 'data', 'configs', args.config_path)
    with open(config_path) as f:
        config = json.load(f)
    config['start_time'] = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    config['aug']['choices'] = parse_list(config['aug']['choices'], parse_aug)
    config['models'] = parse_list(config['models'], parse_model)
    config['strategy'] = parse_strategy(config['strategy'])
    config['strategy_str'] = str(config['strategy'])
    config['num_replicas'] = config['strategy'].num_replicas_in_sync
    config['host_machine'] = os.getenv("HOST_HOSTNAME").upper()
    config['log_id'] = f'{config["start_time"]}-{args.name}'
    config['log_path'] = f'logs/{config["log_id"]}.log'
    return config
