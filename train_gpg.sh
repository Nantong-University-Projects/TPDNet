#!/bin/bash
# 训练脚本：AffinityAttention + GPG_FPN

CONFIG="configs/clrnet/clr_resnet18_culane_gpg.py"
GPU=${1:-0}
LOG_DIR="logs"
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR/gpg_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with GPG_FPN..."
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Log: $LOG_FILE"

nohup python main.py $CONFIG --gpus $GPU > $LOG_FILE 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo $PID > train_gpg.pid
echo "To view logs: tail -f $LOG_FILE"
echo "To stop: kill $PID"
echo "To check status: ps -p $PID"














