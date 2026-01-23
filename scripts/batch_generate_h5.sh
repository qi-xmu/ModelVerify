#!/bin/bash

uv run GenerateH5.py  -d ~/Resources/YT_server_200Hz/test/ -o ~/Resources/YT_server_200Hz/ --test_ratio 1 --train_ratio 0 --val_ratio 0
uv run GenerateH5.py  -d ~/Resources/YT_server_200Hz/train/ -o ~/Resources/YT_server_200Hz/ --test_ratio 0 --train_ratio 1 --val_ratio 0
uv run GenerateH5.py  -d ~/Resources/YT_server_200Hz/valid/ -o ~/Resources/YT_server_200Hz/ --test_ratio 0 --train_ratio 0 --val_ratio 1
