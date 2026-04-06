# AffinityAttention 模块集成说明

## 概述

本模块将师兄论文中的 `MLPAffinityAttention` 创新点集成到 CLRNet 中，用于增强 backbone 顶层特征的语义表示能力。

## 模块位置

- **实现文件**: `clrnet/models/aggregators/affinity_attention.py`
- **配置文件示例**: `configs/clrnet/clr_resnet18_culane_affinity.py`

## 工作原理

`AffinityAttention` 模块在 backbone 输出后、neck 之前应用，对顶层特征（通常是 stride=32 的 C5/P5）进行增强：

1. **Spatial Attention Block**: 通过像素间的亲和力矩阵增强空间关系
2. **Channel Attention Block**: 通过通道间的亲和力矩阵增强通道关系  
3. **CBAM 引导**: 使用 CBAM 模块的通道和空间注意力作为引导信号

最终输出：`out = sab + cab + x`，其中 `sab` 和 `cab` 都经过 CBAM 引导并与原始特征融合。

## 使用方法

### 1. 在配置文件中添加 aggregator

在您的配置文件中添加 `aggregator` 字段：

```python
# 对于 ResNet18/34: 最后一层输出 512 通道
aggregator = dict(
    type='AffinityAttention',
    in_channels=512,
)

# 对于 ResNet50/101: 最后一层输出 2048 通道（需要先降维或调整）
# 注意：ResNet50/101 使用 Bottleneck，expansion=4
# 如果直接使用，需要设置 in_channels=2048
# 但建议先用 1x1 conv 降维到 512，再应用 AffinityAttention
```

### 2. 不同 Backbone 的通道数

| Backbone | 最后一层通道数 | 配置示例 |
|----------|--------------|---------|
| ResNet-18 | 512 | `in_channels=512` |
| ResNet-34 | 512 | `in_channels=512` |
| ResNet-50 | 2048 | `in_channels=2048` (或先降维) |
| ResNet-101 | 2048 | `in_channels=2048` (或先降维) |
| DLA-34 | 512 | `in_channels=512` |

### 3. 运行训练

使用新的配置文件运行训练：

```bash
python main.py --config configs/clrnet/clr_resnet18_culane_affinity.py
```

## 预期效果

- **理论优势**: 
  - 增强顶层特征的全局语义表示
  - 通过亲和力矩阵建立像素/通道间的长距离依赖
  - 对车道线这种长条形结构特别有效

- **预期提升**:
  - CULane: mF1 和 F1@75 可能有 0.5-1.5% 的提升
  - TuSimple: F1 和 Acc 可能有小幅提升
  - LLAMAS: F1@50/75 可能有提升

## 注意事项

1. **计算开销**: AffinityAttention 会增加一定的计算量（主要在顶层特征，尺寸较小，开销可控）
2. **内存占用**: 亲和力矩阵的计算会占用一些显存，但通常在可接受范围内
3. **训练稳定性**: 建议使用与 baseline 相同的学习率和训练策略，如果出现不收敛，可以适当降低学习率

## 代码结构

```
clrnet/models/aggregators/
├── __init__.py
└── affinity_attention.py
    ├── SpatialAttentionBlock      # 空间注意力块
    ├── ChannelAttentionBlock      # 通道注意力块
    ├── CBAM_Module                # CBAM 引导模块
    └── AffinityAttention          # 主模块（已注册到 AGGREGATORS）
```

## 下一步优化方向

1. **简化版 GPG Neck**: 实现多尺度上下文增强的 neck 模块
2. **Cross Splicing Module**: 在 seg 分支中应用交叉融合模块
3. **自适应通道数**: 根据 backbone 自动推断通道数，减少配置负担

## 问题排查

如果遇到问题：

1. **通道数不匹配**: 检查 `in_channels` 是否与 backbone 最后一层输出匹配
2. **显存不足**: 可以尝试减小 batch size 或使用梯度累积
3. **训练不收敛**: 尝试降低学习率或增加 warmup 步数



