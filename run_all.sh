#!/usr/bin/env bash

# BASE="/Users/qi/Resources/论文数据"
BASE="/Users/qi/Resources/论文数据/E2自适应融合-固定噪声融合对比实验/2"

for scene in S1 S2 S3 S4; do
    uv run ./visualizers/DrawTrajCompare.py \
        -d "$BASE/$scene/" \
        -r
done
