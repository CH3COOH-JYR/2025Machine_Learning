import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class EnhancedCardinalityModel(nn.Module):
    """
    增强的基数估计模型，融合了：
    1. SQL查询特征（表、谓词、连接）
    2. 查询计划特征（节点类型、成本、行数）
    3. 多层注意力机制
    """
    
    def __init__(self, 
                 table_vocab_size: int,
                 predicate_vocab_size: int, 
                 join_vocab_size: int,
                 plan_vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(EnhancedCardinalityModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 表特征编码器
        self.table_encoder = nn.Sequential(
            nn.Linear(table_vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 谓词特征编码器
        self.predicate_encoder = nn.Sequential(
            nn.Linear(predicate_vocab_size, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 连接特征编码器
        self.join_encoder = nn.Sequential(
            nn.Linear(join_vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 查询计划特征编码器
        self.plan_encoder = nn.Sequential(
            nn.Linear(plan_vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 4 if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            batch_data: 包含table_features, predicate_features, join_features, plan_features的字典
        Returns:
            预测的cardinality (log scale)
        """
        # 编码各类特征
        table_emb = self.table_encoder(batch_data['table_features'])  # [B, H]
        predicate_emb = self.predicate_encoder(batch_data['predicate_features'])  # [B, H]
        join_emb = self.join_encoder(batch_data['join_features'])  # [B, H]
        plan_emb = self.plan_encoder(batch_data['plan_features'])  # [B, H]
        
        # 准备注意力输入 [seq_len, batch, hidden_dim]
        features = torch.stack([table_emb, predicate_emb, join_emb, plan_emb], dim=0)
        
        # 应用多头注意力
        attn_output, _ = self.attention(features, features, features)
        
        # 转回 [batch, seq_len, hidden_dim] 并展平
        attn_output = attn_output.transpose(0, 1)  # [B, 4, H]
        fused_features = attn_output.flatten(1)  # [B, 4*H]
        
        # 通过融合层
        for fusion_layer in self.fusion_layers:
            residual = fused_features
            fused_features = fusion_layer(fused_features)
            # 残差连接（如果维度匹配）
            if residual.size(-1) == fused_features.size(-1):
                fused_features = fused_features + residual
        
        # 输出预测
        output = self.output_layers(fused_features)
        
        return output

class QueryCardinalityDataset(torch.utils.data.Dataset):
    """查询基数估计数据集"""
    
    def __init__(self, data: List[Dict], cardinalities: List[int], processor):
        self.data = data
        self.cardinalities = cardinalities
        self.processor = processor
        
        # 预处理所有样本
        self.encoded_samples = []
        for i, sample in enumerate(data):
            encoded = processor.encode_sample(sample['query'], sample['explain_result'])
            self.encoded_samples.append(encoded)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoded = self.encoded_samples[idx]
        
        # 转换为张量
        features = {
            'table_features': torch.FloatTensor(encoded['table_features']),
            'predicate_features': torch.FloatTensor(encoded['predicate_features']),
            'join_features': torch.FloatTensor(encoded['join_features']),
            'plan_features': torch.FloatTensor(encoded['plan_features'])
        }
        
        # 目标值（log scale）
        if self.cardinalities is not None:
            # 确保cardinality至少为1，并限制log值的范围
            cardinality = max(1, self.cardinalities[idx])
            log_cardinality = np.log(cardinality)
            # 限制log值范围，避免极值
            log_cardinality = np.clip(log_cardinality, 0, 20)  # e^20 ≈ 4.8亿
            target = torch.FloatTensor([log_cardinality])
        else:
            target = torch.FloatTensor([0])  # 测试数据没有目标值
        
        return features, target

def print_metrics(y_true, y_pred, prefix=""):
    """打印评估指标"""
    # 限制输入范围，避免exp溢出
    y_true = np.clip(y_true, -10, 20)
    y_pred = np.clip(y_pred, -10, 20)
    
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    
    # 避免除零和inf
    y_true_exp = np.clip(y_true_exp, 1e-6, 1e9)
    y_pred_exp = np.clip(y_pred_exp, 1e-6, 1e9)
    
    qerror = np.maximum(y_pred_exp / y_true_exp, y_true_exp / y_pred_exp)
    
    # 检查是否有inf或nan
    if np.any(np.isinf(qerror)) or np.any(np.isnan(qerror)):
        print(f"{prefix}Q-Error Statistics: 数值不稳定，跳过计算")
        return
    
    print(f"{prefix}Q-Error Statistics:")
    print(f"  Mean: {np.mean(qerror):.4f}")
    print(f"  Median: {np.median(qerror):.4f}")
    print(f"  90th percentile: {np.percentile(qerror, 90):.4f}")
    print(f"  95th percentile: {np.percentile(qerror, 95):.4f}")
    print(f"  99th percentile: {np.percentile(qerror, 99):.4f}")
    print(f"  Max: {np.max(qerror):.4f}")

class CardinalityLoss(nn.Module):
    """自定义的基数估计损失函数"""
    
    def __init__(self, alpha=0.9):
        super(CardinalityLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # 数值稳定性检查
        pred = torch.clamp(pred, min=-10, max=10)  # 限制预测值范围
        target = torch.clamp(target, min=-10, max=10)  # 限制目标值范围
        
        # MSE损失（在log空间）
        mse = self.mse_loss(pred, target)
        
        # 简化损失，主要使用MSE
        return mse 