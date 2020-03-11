#!/usr/bin/env bash

# python week2.py --color rgb --channels 0 2 # 0.509
# python week2.py --color hsv --channels 0 2 # 0.201
# python week2.py --color hsv --channels 0   # 0.123
python week2.py --color ycrcb --channels 1 2 # 0.025
python week2.py --color lab --channels 1 2   # 0.003
