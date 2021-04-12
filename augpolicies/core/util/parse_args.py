import argparse
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser(description='Script to run augpolicies')
    parser.add_argument('--hpc', action='store_true', help='Hypothesis testing aug comparison')
    parser.add_argument('--hpe', action='store_true', help='Hypothesis testing aug at end')
    parser.add_argument('--vis', action='store_true', help='Hypothesis testing visualise result')
    parser.add_argument('--rank', action='store_true', help='Ranks the full training sessions and gets the ranking through time')
    parser.add_argument('--data', default='fmnist', type=str, choices=['fmnist', 'cifar10'], help='Stores dataset.')
    args = parser.parse_args()
    return args


def get_dataset_from_args():
    args = get_args()
    if args.data == 'cifar10':
        dataset = tf.keras.datasets.cifar10
    elif args.data == 'fmnist':
        dataset = tf.keras.datasets.fashion_mnist
    return dataset
