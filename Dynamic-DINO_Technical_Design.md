# Dynamic-DINO 技术设计文档
*基于 Grounding DINO 1.5 Edge 的 MoE 改造方案*

## 1. 项目背景与目标

### 1.1 背景
- **原始模型**: Grounding DINO 1.5 Edge - 开放词汇目标检测模型
- **目标**: 在保持整体 pipeline 不变的前提下，将 decoder 的 FFN 替换为 MoE-FFN
- **参考论文**: Dynamic-DINO Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection

### 1.2 核心思路
**Dynamic-DINO 的三大核心策略：**
1. **复制 (Replication)**: 将预训练 FFN 扩展成多个专家副本 (supernet)
2. **分解 (Decomposition)**: 每个 FFN hidden dimension 拆分成多个小专家
3. **路由 (Router)**: 为每个 token 动态选择 top-k 个专家

## 2. Grounding DINO 1.5 Edge 架构分析

### 2.1 完整检测流程

```
输入图像 → Backbone → Encoder → Query Selection → Decoder → Detection Head → 输出
    ↓           ↓          ↓           ↓            ↓           ↓
  patch     多尺度特征   融合特征    900 queries   query     分类+回归
 embedding    tokens      +text                  embedding
```

#### 详细流程：
1. **Backbone (EfficientViT-L1)**
   - 输入图像切成 patch → patch embedding 
   - 输出多尺度特征 (8×8, 16×16, 32×32)
   - 代码位置: `self.backbone(samples)`

2. **Text Encoding**
   - 输入文本 → BERT 编码 → text tokens
   - 代码位置: `self.bert(**tokenized_for_encoder)`

3. **Encoder (TransformerEncoder)**
   - 多层 transformer block 融合图像和文本特征
   - 输出: image tokens (融合后的区域特征)
   - 代码位置: `self.encoder(...)`

4. **Language-guided Query Selection**
   - 结合图像 tokens 和 text tokens
   - 生成 900 个 query tokens (候选检测 query)
   - 代码位置: `tgt_embed.weight` + `refpoint_embed.weight`

5. **Decoder (TransformerDecoder)**
   - 多个 transformer block，每层包含：
     - Cross-attention: query ↔ image tokens
     - **FFN**: 每个 query token 独立过 FFN ⭐️ **改造目标**
   - 输出: 900 个 query embedding
   - 代码位置: `self.decoder(...)`

6. **Detection Head**
   - 分类: `self.class_embed`
   - 回归: `self.bbox_embed`
   - 匹配: Hungarian matcher

### 2.2 Decoder FFN 结构分析

**当前 FFN 实现** (`DeformableTransformerDecoderLayer`):

```python
# 初始化
self.linear1 = nn.Linear(d_model, d_ffn)           # 256 → 2048
self.activation = _get_activation_fn(activation)    # ReLU/GELU
self.linear2 = nn.Linear(d_ffn, d_model)           # 2048 → 256
self.dropout3 = nn.Dropout(dropout)
self.dropout4 = nn.Dropout(dropout)
self.norm3 = nn.LayerNorm(d_model)

# 前向传播
def forward_ffn(self, tgt):
    tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout4(tgt2)
    tgt = self.norm3(tgt)
    return tgt
```

**关键参数**:
- `d_model`: 256 (隐藏维度)
- `d_ffn`: 2048 (FFN 内部维度)
- 输入形状: `[num_queries, batch_size, d_model]` = `[900, bs, 256]`

## 3. MoE-FFN 改造方案

### 3.1 整体架构设计

```
原始 FFN:              MoE-FFN:
                      
Linear1 (256→2048)     Router (256→num_experts)
     ↓                     ↓
Activation             Top-K Selection
     ↓                     ↓
Linear2 (2048→256)     Expert1  Expert2  ...  ExpertN
                           ↓       ↓            ↓
                       Weighted Combination
                           ↓
                      Output (256)
```

### 3.2 MoE-FFN 详细设计

#### 3.2.1 核心组件

```python
class MoEFFN(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, num_experts=8, k_active=2):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.k_active = k_active
        
        # 专家网络：分解策略
        expert_hidden_dim = d_ffn // num_experts  # 2048 // 8 = 256
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])
        
        # 路由网络
        self.router = nn.Linear(d_model, num_experts)
        
        # 负载均衡
        self.load_balancing_loss_coef = 0.01
```

#### 3.2.2 路由机制

```python
def forward(self, x):
    batch_size, seq_len, d_model = x.shape
    x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
    
    # 路由决策
    router_logits = self.router(x_flat)  # [batch*seq, num_experts]
    router_probs = F.softmax(router_logits, dim=-1)
    
    # Top-K 选择
    topk_probs, topk_indices = torch.topk(router_probs, self.k_active, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 重新归一化
    
    # 专家计算
    output = torch.zeros_like(x_flat)
    for i in range(self.k_active):
        expert_idx = topk_indices[:, i]
        expert_weight = topk_probs[:, i].unsqueeze(-1)
        
        # 批量计算选中的专家
        for expert_id in range(self.num_experts):
            mask = (expert_idx == expert_id)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                output[mask] += expert_weight[mask] * expert_output
    
    return output.view(batch_size, seq_len, d_model)
```

#### 3.2.3 负载均衡机制

```python
def compute_load_balancing_loss(self, router_probs):
    """计算负载均衡损失，避免专家使用不均"""
    # router_probs: [batch*seq, num_experts]
    expert_usage = router_probs.mean(dim=0)  # [num_experts]
    ideal_usage = 1.0 / self.num_experts
    
    # 计算使用率偏差
    load_balance_loss = torch.sum((expert_usage - ideal_usage) ** 2)
    return self.load_balancing_loss_coef * load_balance_loss
```

### 3.3 权重初始化策略

#### 3.3.1 复制+分解初始化

```python
def initialize_from_pretrained_ffn(self, original_linear1, original_linear2):
    """从预训练 FFN 权重初始化专家"""
    # 原始权重: linear1 [2048, 256], linear2 [256, 2048]
    
    expert_hidden_dim = self.d_ffn // self.num_experts  # 256
    
    for i, expert in enumerate(self.experts):
        # 分解权重
        start_idx = i * expert_hidden_dim
        end_idx = (i + 1) * expert_hidden_dim
        
        # 初始化 expert 的第一层 (256 → 256)
        expert[0].weight.data = original_linear1.weight[start_idx:end_idx, :].clone()
        expert[0].bias.data = original_linear1.bias[start_idx:end_idx].clone()
        
        # 初始化 expert 的第二层 (256 → 256)  
        expert[2].weight.data = original_linear2.weight[:, start_idx:end_idx].clone()
        expert[2].bias.data = original_linear2.bias.clone() / self.num_experts
```

#### 3.3.2 Router 初始化

```python
def initialize_router(self):
    """Router 初始化策略"""
    # 策略1: 均匀分布初始化，确保初期各专家使用概率相近
    nn.init.normal_(self.router.weight, mean=0, std=0.01)
    nn.init.constant_(self.router.bias, 0)
    
    # 策略2: 添加温度参数，控制初期选择的锐度
    self.temperature = nn.Parameter(torch.ones(1) * 1.0)
```

## 4. 集成方案

### 4.1 替换策略

**替换位置**: `groundingdino/models/GroundingDINO/transformer.py`
- 类: `DeformableTransformerDecoderLayer`
- 方法: `forward_ffn`

**替换步骤**:

1. **创建 MoE 模块**: `groundingdino/models/GroundingDINO/moe_ffn.py`

2. **修改 DeformableTransformerDecoderLayer**:
```python
# 原来
self.linear1 = nn.Linear(d_model, d_ffn)
self.linear2 = nn.Linear(d_ffn, d_model)

# 替换为
self.moe_ffn = MoEFFN(d_model=d_model, d_ffn=d_ffn, 
                      num_experts=8, k_active=2)
```

3. **修改 forward_ffn 方法**:
```python
def forward_ffn(self, tgt):
    # 原来
    # tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    
    # 替换为
    tgt2 = self.moe_ffn(tgt)
    
    tgt = tgt + self.dropout4(tgt2)
    tgt = self.norm3(tgt)
    return tgt
```

### 4.2 兼容性保证

1. **接口一致性**: MoE-FFN 输入输出形状与原 FFN 完全一致
2. **梯度流**: 保持反向传播路径正确
3. **设备兼容**: 支持 CPU/GPU/混合精度训练

## 5. 训练策略

### 5.1 训练流程

```python
# 伪代码
1. 加载预训练 Grounding DINO 模型
2. 替换 decoder FFN → MoE-FFN
3. 用预训练权重初始化专家网络
4. 冻结 backbone + encoder，只训练 decoder
5. 在大规模数据上微调 (Objects365 + GoldG + V3Det)
```

### 5.2 损失函数

```python
total_loss = detection_loss + load_balancing_loss

# detection_loss: 原始检测损失 (分类 + 回归)
# load_balancing_loss: MoE 负载均衡损失
```

### 5.3 超参数设置

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| num_experts | 8 | 专家数量 |
| k_active | 2 | 每次激活的专家数 |
| load_balancing_coef | 0.01 | 负载均衡权重 |
| learning_rate | 1e-5 | 学习率 |
| warmup_steps | 1000 | 预热步数 |

## 6. 性能预期

### 6.1 计算效率
- **原始 FFN**: 每个 token 激活所有参数 (256→2048→256)
- **MoE-FFN**: 每个 token 只激活 2/8 的参数 (256→256→256 × 2)
- **理论加速比**: ~75% 参数减少，实际推理加速取决于硬件优化

### 6.2 精度保持
- **初始化策略**确保训练初期性能不退化
- **负载均衡**避免专家退化，保持表达能力
- **渐进式训练**确保稳定收敛

## 7. 实验验证计划

### 7.1 基础验证
1. **功能测试**: 确保 MoE 模块前向/反向传播正确
2. **精度对比**: 初始化后与原模型精度对比
3. **速度测试**: 推理速度对比

### 7.2 端到端验证
1. **COCO 验证集**: mAP 指标对比
2. **推理速度**: FPS 测试
3. **专家使用分析**: 路由决策可视化

## 8. 风险与应对

### 8.1 潜在风险
1. **精度下降**: MoE 可能导致精度损失
2. **训练不稳定**: 路由网络训练可能不稳定
3. **内存开销**: 专家网络增加内存使用

### 8.2 应对策略
1. **渐进式替换**: 先在少数层尝试，逐步扩展
2. **路由正则化**: 添加 entropy 正则化稳定训练
3. **动态专家数**: 根据性能动态调整专家数量

---

*本文档将随着实现进展持续更新*
