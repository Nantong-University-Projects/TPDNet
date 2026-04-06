# LaneGraphReasoner 模块集成说明

## 概述

本模块实现了三个"建模层"的改进，用于增强车道线检测中的prior-prior关系建模：

1. **模块A：Lane-Query Graph Reasoner** - 在head内对priors做关系推理
2. **模块B：Affinity-aware Matching** - 把关系项写进assign的cost
3. **模块C：Topology Regularization** - 用几何构造GT关系矩阵，监督A_pred

## 模块位置

- **LaneGraphReasoner类**: `clrnet/models/heads/clr_head.py`
- **GT关系矩阵构造**: `clrnet/models/utils/dynamic_assign.py` - `build_gt_affinity()`
- **关系损失**: `clrnet/models/utils/dynamic_assign.py` - `rel_loss()`
- **关系cost**: `clrnet/models/utils/dynamic_assign.py` - `relation_cost_matrix()`

## 模块A：Lane-Query Graph Reasoner

### 插入点

在 `clr_head.py` 的forward函数中，fc_features之后、reg_layers之前：

```python
# 原流程：
fc_features -> cls_modules/reg_modules -> cls_layers/reg_layers

# 新流程：
fc_features -> LaneGraphReasoner -> cls_modules/reg_modules -> cls_layers/reg_layers
```

### 实现

```python
class LaneGraphReasoner(nn.Module):
    def forward(self, h):  # h: (num_priors, D)
        q = self.q(h)
        k = self.k(h)
        att = (q @ k.t()) / sqrt(d_k)  # attention logits
        A = att.softmax(dim=-1)
        msg = A @ self.v(h)
        h2 = self.ln(h + self.proj(msg))
        return h2, att  # 返回更新后的特征和attention logits
```

### 输出

- `h2`: 更新后的特征 (num_priors, D)
- `att`: attention logits (num_priors, num_priors)，存储在 `output['affinity_lists']` 中

## 模块C：Topology Regularization

### GT关系矩阵构造

```python
def build_gt_affinity(targets, img_w, tau_px=20, sigma_px=30):
    # 从几何关系构造GT affinity矩阵
    # 对每条GT lane，计算与其他lanes的平均距离
    # 使用exp(-d/sigma)得到软关系权重
```

### 监督损失

```python
def rel_loss(att_logits_pred, A_gt, lam_sparse=0.01):
    # 对齐预测的affinity和GT affinity
    # 添加稀疏性正则化（鼓励稀疏连接）
```

### 使用方式

在 `loss()` 函数中，对matched的priors计算rel_loss：

```python
# 只监督matched的priors
att_pred_sub = att_logits[matched_row_inds][:, matched_row_inds]
A_gt_sub = A_gt[matched_col_inds][:, matched_col_inds]
rel_loss_total += rel_loss(att_pred_sub, A_gt_sub, lam_sparse=lam_sparse)
```

## 模块B：Affinity-aware Matching

### 关系cost计算

```python
def relation_cost_matrix(att_logits_pred, A_gt, topk=8):
    # 提取关系签名向量（TopK邻居强度）
    # 计算预测和GT的关系签名距离
```

### 集成到assign

在 `assign()` 函数中：

```python
# 如果启用rel_cost_weight > 0
A_gt = build_gt_affinity(targets, img_w)
rel_cost = relation_cost_matrix(pred_affinity_logits, A_gt, topk=8)
cost = cost + rel_cost_weight * rel_cost
```

## 配置参数

在配置文件中添加：

```python
# 模块C：关系损失权重
rel_loss_weight = 0.1  # 0表示禁用

# 模块B：关系cost权重（在matching中）
rel_cost_weight = 0.0  # 0表示禁用，建议先设为0，等模块C稳定后再启用

# 模块C：稀疏性正则化权重
lam_sparse = 0.01
```

## 实施顺序建议

按照用户建议的顺序实施，避免项目崩溃：

### 阶段1：只启用模块A（Reasoner）

```python
rel_loss_weight = 0.0  # 禁用
rel_cost_weight = 0.0  # 禁用
```

- 确保forward/inference不崩溃
- 验证affinity_lists正确输出

### 阶段2：启用模块C（监督）

```python
rel_loss_weight = 0.1  # 启用
rel_cost_weight = 0.0  # 仍禁用
lam_sparse = 0.01
```

- 观察rel_loss是否正常下降
- 这步提升通常更稳定

### 阶段3：启用模块B（matching）

```python
rel_loss_weight = 0.1  # 保持启用
rel_cost_weight = 0.5  # 启用（从0.1开始逐步增加）
lam_sparse = 0.01
```

- 这是最大提点，但最容易引入训练不稳定
- 建议从小的rel_cost_weight开始（如0.1），逐步增加

## 关键函数签名

### clr_head.py

```python
class LaneGraphReasoner(nn.Module):
    def __init__(self, d, d_k=64)
    def forward(self, h) -> (h2, att)

class CLRHead(nn.Module):
    def forward(self, batch_features, **kwargs) -> output
        # output包含: {'predictions_lists': [...], 'seg': ..., 'affinity_lists': [...]}
    
    def loss(self, output, batch, ..., rel_loss_weight=0.1, rel_cost_weight=0.0, lam_sparse=0.01)
```

### dynamic_assign.py

```python
def build_gt_affinity(targets, img_w, tau_px=20, sigma_px=30) -> A_gt
def rel_loss(att_logits_pred, A_gt, lam_sparse=0.01) -> loss
def relation_cost_matrix(att_logits_pred, A_gt, topk=8) -> cost
def assign(predictions, targets, img_w, img_h, ...,
           rel_cost_weight=0.0, pred_affinity_logits=None) -> (matched_row_inds, matched_col_inds)
```

## 注意事项

1. **维度匹配**：N = num_priors, M = num_targets，在计算rel_loss时只使用matched的子图
2. **初始化**：`_affinity_lists` 在forward开始时初始化
3. **向后兼容**：如果affinity_lists不存在，代码会自动处理（返回None）
4. **训练稳定性**：建议按阶段逐步启用，观察训练曲线

## 预期效果

- **模块A**：让每个prior通过图交互后再回归，提升特征表示
- **模块C**：让学到的affinity对齐几何关系，提升可解释性
- **模块B**：让matching考虑结构一致性，提升匹配质量

三个模块协同工作，预期能带来稳定的性能提升。







