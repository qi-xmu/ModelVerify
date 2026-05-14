#!/usr/bin/env bash

uv run ./visualizers/DrawTrajCompare.py \
    -u /Users/qi/Resources/论文数据/E2自适应融合-固定噪声融合对比实验/2/S3 \
    -v -r \
    --session 4 \
    --device RM \
    --plot
