#!/bin/bash

source env/bin/activate

python main.py --hpe --data cifar10
python main.py --hpc --data cifar10
