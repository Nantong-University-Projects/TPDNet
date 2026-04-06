#!/bin/bash
# 训练脚本：Directional Attention + Affinity Attention
# 使用nohup在后台运行，输出重定向到日志文件

CONFIG="configs/clrnet/clr_resnet18_culane_directional.py"
GPU=${1:-0}
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/directional_${TIMESTAMP}.log"
PID_FILE="train_directional.pid"

echo "=========================================="
echo "Starting training with Directional Attention..."
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Log: $LOG_FILE"
echo "=========================================="

# 使用nohup在后台运行，将stdout和stderr都重定向到日志文件
nohup python main.py $CONFIG --gpus $GPU > $LOG_FILE 2>&1 &
PID=$!

# 保存进程ID
echo $PID > $PID_FILE

echo "Training started with PID: $PID"
echo "PID saved to: $PID_FILE"
echo ""
echo "Useful commands:"
echo "  View logs:        tail -f $LOG_FILE"
echo "  View last 100:    tail -n 100 $LOG_FILE"
echo "  Stop training:    kill $PID"
echo "  Check status:     ps -p $PID"
echo "  Check GPU usage:  nvidia-smi"
echo "=========================================="










