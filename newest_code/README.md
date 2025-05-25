# QueryFormer: 基于Transformer的SQL查询基数估计

## 项目概述

本项目实现了基于QueryFormer思想的Transformer模型，用于SQL查询基数（cardinality）估计。相比传统方法，QueryFormer能够更好地理解SQL查询的语义结构和查询计划信息，从而提供更准确的基数预测。

## 核心特性

### 🚀 先进的模型架构
- **Transformer编码器**: 使用多层Transformer编码器处理SQL查询序列
- **位置编码**: 捕捉SQL token的位置信息
- **多头注意力**: 自动学习查询不同部分的重要性
- **查询计划融合**: 结合查询计划信息提升预测精度

### 🔧 智能特征工程
- **SQL分词器**: 专门为SQL查询设计的分词器
- **查询计划编码**: 深度提取查询计划的结构和统计信息
- **多模态融合**: 融合SQL文本和查询计划特征

### 📊 优化的训练策略
- **组合损失函数**: MSE + MAE损失的加权组合
- **学习率调度**: 自适应学习率调整
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: 防止过拟合

## 项目结构

```
newest_code/
├── queryformer_model.py      # QueryFormer模型定义
├── advanced_data_processor.py # 高级数据处理器
├── train_queryformer.py      # 训练脚本
├── predict_queryformer.py    # 预测脚本
├── requirements.txt          # 依赖包
├── README.md                # 项目说明
└── models/                  # 模型保存目录
    ├── queryformer_*.pth    # 训练好的模型
    ├── tokenizer.pkl        # 分词器
    ├── processor.pkl        # 数据处理器
    └── training_history.json # 训练历史
```

## 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r newest_code/requirements.txt
```

### 2. 训练模型

```bash
# 基础训练（推荐设置）
python newest_code/train_queryformer.py \
    --train_samples 30000 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --d_model 256 \
    --num_layers 6 \
    --nhead 8

# 高性能训练（如果有GPU）
python newest_code/train_queryformer.py \
    --train_samples 50000 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --d_model 512 \
    --num_layers 8 \
    --nhead 16

# 快速测试（小规模）
python newest_code/train_queryformer.py \
    --train_samples 5000 \
    --epochs 10 \
    --batch_size 16 \
    --d_model 128 \
    --num_layers 4
```

### 3. 生成预测

```bash
# 使用训练好的模型进行预测
python newest_code/predict_queryformer.py \
    --model_dir newest_code/models \
    --test_file data/test_data.json \
    --output_file newest_code/queryformer_predictions.csv
```

## 模型架构详解

### QueryFormer核心组件

1. **SQL分词器 (QueryTokenizer)**
   - 专门处理SQL关键词、操作符、标识符
   - 支持特殊token：[PAD], [UNK], [CLS], [SEP], [NUM], [STR], [COL], [TAB]
   - 智能识别表名、列名、数值、字符串

2. **位置编码 (PositionalEncoding)**
   - 正弦/余弦位置编码
   - 支持任意长度序列
   - 帮助模型理解token顺序

3. **Transformer编码器**
   - 多层自注意力机制
   - 前馈神经网络
   - 残差连接和层归一化
   - GELU激活函数

4. **查询计划编码器 (PlanEncoder)**
   - 节点类型嵌入
   - 成本、行数、选择性特征
   - 多特征融合

5. **特征融合层**
   - 交叉注意力机制
   - SQL查询与计划特征融合
   - 自适应权重学习

### 训练优化策略

- **数值稳定性**: log空间训练，避免数值溢出
- **正则化**: Dropout + 权重衰减
- **批归一化**: 稳定训练过程
- **梯度裁剪**: 防止梯度爆炸

## 性能优势

### 相比现有方法的改进

1. **更好的语义理解**
   - Transformer能够捕捉SQL查询的长距离依赖
   - 注意力机制自动学习重要特征

2. **多模态信息融合**
   - 同时利用SQL文本和查询计划信息
   - 交叉注意力机制优化特征融合

3. **端到端学习**
   - 无需手工特征工程
   - 自动学习最优特征表示

4. **可扩展性强**
   - 支持复杂查询结构
   - 易于扩展到新的数据库和查询类型

## 训练参数说明

### 模型参数
- `--d_model`: 模型维度 (默认: 256)
- `--num_layers`: Transformer层数 (默认: 6)
- `--nhead`: 注意力头数 (默认: 8)
- `--dropout`: Dropout率 (默认: 0.1)
- `--max_seq_length`: 最大序列长度 (默认: 512)

### 训练参数
- `--train_samples`: 训练样本数量 (默认: 30000)
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 1e-4)

### 推荐配置

| 场景 | d_model | num_layers | nhead | batch_size | learning_rate |
|------|---------|------------|-------|------------|---------------|
| 快速测试 | 128 | 4 | 4 | 16 | 1e-4 |
| 标准训练 | 256 | 6 | 8 | 32 | 1e-4 |
| 高精度 | 512 | 8 | 16 | 64 | 5e-5 |

## 评估指标

模型使用Q-Error作为主要评估指标：

```
Q-Error = max(predicted/actual, actual/predicted)
```

关键统计量：
- Mean Q-Error: 平均Q-Error
- Median Q-Error: 中位数Q-Error  
- 90th/95th/99th percentile: 高分位数Q-Error
- Max Q-Error: 最大Q-Error

## 输出文件说明

### 训练输出
- `queryformer_epoch_X_loss_Y.pth`: 训练好的模型
- `tokenizer.pkl`: 分词器
- `processor.pkl`: 数据处理器
- `training_history.json`: 训练历史
- `training_history.png`: 训练曲线图

### 预测输出
- `queryformer_predictions.csv`: 预测结果
- `queryformer_predictions_stats.json`: 预测统计信息

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 减小d_model或num_layers
   - 减少train_samples

2. **训练速度慢**
   - 使用GPU训练
   - 增大batch_size
   - 减少num_workers

3. **模型不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 性能调优

1. **提升精度**
   - 增加模型容量 (d_model, num_layers)
   - 使用更多训练数据
   - 调整损失函数权重

2. **加速训练**
   - 使用混合精度训练
   - 增大batch_size
   - 使用多GPU训练

## 技术细节

### 数据预处理
1. SQL查询分词和编码
2. 查询计划特征提取
3. 数值特征归一化
4. 序列填充和截断

### 模型训练
1. 随机种子设置确保可重现性
2. 学习率调度器自适应调整
3. 早停机制防止过拟合
4. 模型检查点保存

### 预测流程
1. 加载训练好的模型和组件
2. 预处理测试数据
3. 批量预测
4. 后处理和格式化输出

## 扩展方向

1. **模型改进**
   - 引入图神经网络处理查询计划图
   - 使用预训练语言模型
   - 多任务学习

2. **特征增强**
   - 历史查询统计
   - 数据分布特征
   - 索引使用情况

3. **工程优化**
   - 模型量化和压缩
   - 在线学习和增量更新
   - 分布式训练

## 作者信息

- **项目**: QueryFormer查询基数估计
- **基于**: VLDB 2022 QueryFormer论文思想
- **实现**: 基于PyTorch的端到端解决方案

## 参考文献

1. QueryFormer: A Tree Transformer Model for Query Plan Representation (VLDB 2022)
2. Attention Is All You Need (NIPS 2017)
3. Learned Cardinalities: Estimating Correlated Joins with Deep Learning (CIDR 2019) 