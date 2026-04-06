#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：Directional Attention + Affinity Attention
支持后台运行和日志记录
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Train CLRNet with Directional Attention')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU IDs to use')
    parser.add_argument('--config', type=str, 
                       default='configs/clrnet/clr_resnet18_culane_directional.py',
                       help='Config file path')
    parser.add_argument('--background', action='store_true', 
                       help='Run in background (nohup)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'directional_{timestamp}.log')
    
    # 构建命令
    gpu_str = ' '.join(str(g) for g in args.gpus)
    cmd = f'python main.py {args.config} --gpus {gpu_str}'
    
    print("=" * 50)
    print("Starting training with Directional Attention...")
    print(f"Config: {args.config}")
    print(f"GPUs: {gpu_str}")
    print(f"Log: {log_file}")
    print("=" * 50)
    
    if args.background:
        # 后台运行
        print(f"\nRunning in background...")
        print(f"View logs: tail -f {log_file}")
        print(f"View last 100 lines: tail -n 100 {log_file}")
        
        # 使用nohup在后台运行
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd.split(),
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
        
        # 保存PID
        pid_file = 'train_directional.pid'
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"Training started with PID: {process.pid}")
        print(f"PID saved to: {pid_file}")
        print(f"To stop: kill {process.pid}")
    else:
        # 前台运行，同时输出到终端和文件
        print(f"\nRunning in foreground (logs also saved to {log_file})...")
        print("Press Ctrl+C to stop\n")
        
        with open(log_file, 'w') as log_f:
            try:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出到终端和文件
                for line in process.stdout:
                    print(line, end='')
                    log_f.write(line)
                    log_f.flush()
                
                process.wait()
                return_code = process.returncode
                
                if return_code == 0:
                    print("\n✓ Training completed successfully!")
                else:
                    print(f"\n✗ Training failed with return code {return_code}")
                    sys.exit(return_code)
                    
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user")
                process.terminate()
                process.wait()
                sys.exit(1)


if __name__ == '__main__':
    main()

