import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class QueryTokenizer:
    """SQL查询分词器"""
    
    def __init__(self):
        # SQL关键词
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE', 'BETWEEN', 'IS',
            'NULL', 'GROUP', 'BY', 'ORDER', 'HAVING', 'DISTINCT', 'COUNT', 'SUM',
            'AVG', 'MIN', 'MAX', 'AS', 'UNION', 'ALL', 'LIMIT', 'OFFSET'
        }
        
        # 操作符
        self.operators = {'=', '>', '<', '>=', '<=', '!=', '<>'}
        
        # 特殊token
        self.special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            '[NUM]': 4, '[STR]': 5, '[COL]': 6, '[TAB]': 7
        }
        
        self.vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, queries: List[str]):
        """构建词汇表"""
        token_counts = {}
        
        for query in queries:
            tokens = self.tokenize(query)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # 添加特殊token
        self.vocab.update(self.special_tokens)
        
        # 添加高频token
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens:
            if token not in self.vocab and count >= 2:  # 至少出现2次
                self.vocab[token] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        print(f"词汇表大小: {self.vocab_size}")
    
    def tokenize(self, query: str) -> List[str]:
        """分词"""
        import re
        
        # 预处理：统一大小写，处理特殊字符
        query = query.upper().strip()
        
        # 分离操作符和标点符号
        for op in self.operators:
            query = query.replace(op, f' {op} ')
        
        for punct in '(),;':
            query = query.replace(punct, f' {punct} ')
        
        # 分词
        tokens = query.split()
        
        processed_tokens = []
        for token in tokens:
            if token in self.sql_keywords:
                processed_tokens.append(token)
            elif token in self.operators:
                processed_tokens.append(token)
            elif token.isdigit():
                processed_tokens.append('[NUM]')
            elif "'" in token or '"' in token:
                processed_tokens.append('[STR]')
            elif '.' in token:  # 可能是表.列格式
                parts = token.split('.')
                if len(parts) == 2:
                    processed_tokens.extend(['[TAB]', '.', '[COL]'])
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def encode(self, query: str, max_length: int = 512) -> List[int]:
        """编码查询为token ID序列"""
        tokens = self.tokenize(query)
        
        # 添加特殊token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 转换为ID
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['[UNK]'])
        
        # 截断或填充
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['[PAD]']] * (max_length - len(token_ids)))
        
        return token_ids

class PlanEncoder(nn.Module):
    """查询计划编码器"""
    
    def __init__(self, d_model: int = 256):
        super(PlanEncoder, self).__init__()
        self.d_model = d_model
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(50, d_model // 4)  # 假设最多50种节点类型
        
        # 数值特征投影
        self.cost_projection = nn.Linear(1, d_model // 4)
        self.rows_projection = nn.Linear(1, d_model // 4)
        self.selectivity_projection = nn.Linear(1, d_model // 4)
        
        # 融合层
        self.fusion = nn.Linear(d_model, d_model)
        
    def forward(self, plan_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            plan_features: 包含node_types, costs, rows, selectivities的字典
        Returns:
            编码后的计划特征 [batch_size, d_model]
        """
        batch_size = plan_features['node_types'].size(0)
        
        # 节点类型嵌入
        node_emb = self.node_type_embedding(plan_features['node_types'])  # [B, seq_len, d_model//4]
        node_emb = torch.mean(node_emb, dim=1)  # [B, d_model//4]
        
        # 数值特征 - 确保维度正确
        costs = plan_features['costs']
        rows = plan_features['rows'] 
        selectivities = plan_features['selectivities']
        
        # 如果是1维，添加维度；如果已经是2维，保持不变
        if costs.dim() == 1:
            costs = costs.unsqueeze(-1)
        if rows.dim() == 1:
            rows = rows.unsqueeze(-1)
        if selectivities.dim() == 1:
            selectivities = selectivities.unsqueeze(-1)
            
        cost_emb = self.cost_projection(costs)  # [B, d_model//4]
        rows_emb = self.rows_projection(rows)  # [B, d_model//4]
        sel_emb = self.selectivity_projection(selectivities)  # [B, d_model//4]
        
        # 拼接所有特征
        combined = torch.cat([node_emb, cost_emb, rows_emb, sel_emb], dim=-1)  # [B, d_model]
        
        # 融合
        output = self.fusion(combined)
        return output

class QueryFormerModel(nn.Module):
    """基于Transformer的查询基数估计模型"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 512):
        super(QueryFormerModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 查询计划编码器
        self.plan_encoder = PlanEncoder(d_model)
        
        # 特征融合
        self.feature_fusion = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_padding_mask(self, token_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """创建padding mask"""
        return (token_ids == pad_token_id)
    
    def forward(self, 
                token_ids: torch.Tensor,
                plan_features: Dict[str, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len] 查询token序列
            plan_features: 查询计划特征字典
            attention_mask: [batch_size, seq_len] 注意力mask
        Returns:
            预测的log cardinality [batch_size, 1]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token嵌入
        token_emb = self.token_embedding(token_ids) * math.sqrt(self.d_model)  # [B, L, D]
        
        # 位置编码
        token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)  # [B, L, D]
        
        # 创建attention mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(token_ids)
        
        # Transformer编码
        query_repr = self.transformer(token_emb, src_key_padding_mask=attention_mask)  # [B, L, D]
        
        # 查询表示：使用[CLS] token或平均池化
        query_vector = query_repr[:, 0, :]  # 使用[CLS] token [B, D]
        
        # 查询计划编码
        plan_vector = self.plan_encoder(plan_features)  # [B, D]
        
        # 特征融合：使用注意力机制
        query_vector = query_vector.unsqueeze(1)  # [B, 1, D]
        plan_vector = plan_vector.unsqueeze(1)    # [B, 1, D]
        
        # 交叉注意力
        fused_features, _ = self.feature_fusion(
            query_vector, plan_vector, plan_vector
        )  # [B, 1, D]
        
        fused_features = fused_features.squeeze(1)  # [B, D]
        
        # 输出预测
        output = self.output_projection(fused_features)  # [B, 1]
        
        return output

class QueryCardinalityLoss(nn.Module):
    """查询基数估计损失函数"""
    
    def __init__(self, alpha: float = 0.5):
        super(QueryCardinalityLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [batch_size, 1]
            target: 真实值 [batch_size, 1]
        """
        # 数值稳定性
        pred = torch.clamp(pred, min=-10, max=20)
        target = torch.clamp(target, min=-10, max=20)
        
        # 组合损失
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        
        return self.alpha * mse + (1 - self.alpha) * mae

def q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算Q-Error"""
    # 转换回原始空间
    y_true_exp = np.exp(np.clip(y_true, -10, 20))
    y_pred_exp = np.exp(np.clip(y_pred, -10, 20))
    
    # 避免除零
    y_true_exp = np.maximum(y_true_exp, 1e-6)
    y_pred_exp = np.maximum(y_pred_exp, 1e-6)
    
    # 计算Q-Error
    qerror = np.maximum(y_pred_exp / y_true_exp, y_true_exp / y_pred_exp)
    
    return qerror

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""):
    """打印评估指标"""
    qerror = q_error(y_true, y_pred)
    
    print(f"{prefix}Q-Error Statistics:")
    print(f"  Mean: {np.mean(qerror):.4f}")
    print(f"  Median: {np.median(qerror):.4f}")
    print(f"  90th percentile: {np.percentile(qerror, 90):.4f}")
    print(f"  95th percentile: {np.percentile(qerror, 95):.4f}")
    print(f"  99th percentile: {np.percentile(qerror, 99):.4f}")
    print(f"  Max: {np.max(qerror):.4f}") 