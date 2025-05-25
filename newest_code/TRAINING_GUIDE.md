# QueryFormer训练指南

## 🎯 项目概述

我已经为您在`newest_code`文件夹下创建了一个全新的基于**QueryFormer (VLDB 2022) Transformer**的查询基数估计模型。这个实现相比现有的`enhanced_model`有以下显著优势：

### 🚀 核心优势

1. **先进的Transformer架构**
   - 多层自注意力机制，能够捕捉SQL查询的长距离依赖关系
   - 位置编码帮助理解SQL token的顺序信息
   - 多头注意力自动学习查询不同部分的重要性

2. **智能的SQL理解**
   - 专门设计的SQL分词器，能够正确处理SQL关键词、操作符、表名、列名
   - 支持特殊token：[PAD], [UNK], [CLS], [SEP], [NUM], [STR], [COL], [TAB]
   - 自动识别数值、字符串、表列引用等不同类型的token

3. **多模态特征融合**
   - 同时利用SQL查询文本和查询计划信息
   - 交叉注意力机制优化两种特征的融合
   - 查询计划编码器深度提取节点类型、成本、行数、选择性等信息

4. **优化的训练策略**
   - 组合损失函数（MSE + MAE）
   - 自适应学习率调度
   - 梯度裁剪防止梯度爆炸
   - 早停机制防止过拟合

## 📁 文件结构

```
newest_code/
├── queryformer_model.py      # QueryFormer模型核心实现
├── advanced_data_processor.py # 高级数据处理器
├── train_queryformer.py      # 训练脚本
├── predict_queryformer.py    # 预测脚本
├── quick_start.py            # 快速启动脚本
├── requirements.txt          # 依赖包列表
├── README.md                # 详细项目说明
├── TRAINING_GUIDE.md        # 本训练指南
└── models/                  # 模型保存目录（训练后生成）
```

## 🛠️ 环境准备

### 1. 安装依赖

```bash
pip install -r newest_code/requirements.txt
```

主要依赖包：
- `torch>=1.12.0` - PyTorch深度学习框架
- `transformers>=4.20.0` - Transformer模型支持
- `numpy`, `pandas` - 数据处理
- `matplotlib`, `seaborn` - 可视化
- `tqdm` - 进度条
- `sqlparse` - SQL解析

### 2. 数据准备

确保以下数据文件存在：
- `data/train_data.json` - 训练数据
- `data/test_data.json` - 测试数据  
- `data/column_min_max_vals.csv` - 列统计信息

## 🚀 快速开始

### 方法1：使用快速启动脚本（推荐）

```bash
# 完整流程（训练+预测）
python newest_code/quick_start.py --mode full

# 仅训练
python newest_code/quick_start.py --mode train

# 仅预测（需要先有训练好的模型）
python newest_code/quick_start.py --mode predict

# 快速测试模式（小规模数据）
python newest_code/quick_start.py --mode full --quick
```

### 方法2：手动执行

#### 步骤1：训练模型

```bash
# 标准训练（推荐）
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

# 快速测试
python newest_code/train_queryformer.py \
    --train_samples 5000 \
    --epochs 10 \
    --batch_size 16 \
    --d_model 128 \
    --num_layers 4
```

#### 步骤2：生成预测

```bash
python newest_code/predict_queryformer.py \
    --model_dir newest_code/models \
    --test_file data/test_data.json \
    --output_file newest_code/queryformer_predictions.csv
```

## 📊 训练参数详解

### 核心参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `--train_samples` | 30000 | 训练样本数量 | 5000-60000 |
| `--epochs` | 50 | 训练轮数 | 10-100 |
| `--batch_size` | 32 | 批大小 | 16-128 |
| `--learning_rate` | 1e-4 | 学习率 | 1e-5 to 1e-3 |
| `--early_stopping_patience` | 10 | 早停耐心值 | 5-20 |
| `--min_delta` | 1e-4 | 早停最小改进阈值 | 1e-5-1e-3 |

### 模型架构参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `--d_model` | 256 | 模型维度 | 128-512 |
| `--num_layers` | 6 | Transformer层数 | 4-12 |
| `--nhead` | 8 | 注意力头数 | 4-16 |
| `--dropout` | 0.1 | Dropout率 | 0.05-0.3 |
| `--max_seq_length` | 512 | 最大序列长度 | 256-1024 |

### 推荐配置

#### 快速测试配置
```bash
--train_samples 5000 --epochs 10 --batch_size 16 --d_model 128 --num_layers 4 --nhead 4
```

#### 标准配置（推荐）
```bash
--train_samples 30000 --epochs 50 --batch_size 32 --d_model 256 --num_layers 6 --nhead 8
```

#### 高精度配置
```bash
--train_samples 50000 --epochs 100 --batch_size 64 --d_model 512 --num_layers 8 --nhead 16
```

## 📈 训练过程监控

### 训练输出

训练过程中会显示：
- 每个epoch的训练损失和验证损失
- 验证集的Q-Error统计（Mean, Median, 90th/95th/99th percentile）
- 学习率调整信息
- 最佳模型保存信息

### 输出文件

训练完成后会生成：
- `newest_code/models/queryformer_epoch_X_loss_Y.pth` - 训练好的模型
- `newest_code/models/tokenizer.pkl` - SQL分词器
- `newest_code/models/processor.pkl` - 数据处理器
- `newest_code/models/training_history.json` - 训练历史数据
- `newest_code/models/training_history.png` - 训练曲线图

## 🔮 预测和评估

### 预测输出

- `newest_code/queryformer_predictions.csv` - 预测结果文件
- `newest_code/queryformer_predictions_stats.json` - 预测统计信息

### 评估指标

模型使用Q-Error作为主要评估指标：
```
Q-Error = max(predicted/actual, actual/predicted)
```

关键统计量：
- **Mean Q-Error**: 平均Q-Error，越小越好
- **Median Q-Error**: 中位数Q-Error，更稳健的指标
- **90th/95th/99th percentile**: 高分位数，反映极端情况的处理能力

## 🎯 为什么选择QueryFormer？

### 相比现有enhanced_model的优势

1. **更强的表达能力**
   - Transformer架构能够建模复杂的查询结构
   - 自注意力机制自动发现查询中的重要模式
   - 位置编码保留SQL语法的顺序信息

2. **更好的特征融合**
   - 专门的SQL分词器，比简单的特征工程更智能
   - 查询计划编码器深度提取计划信息
   - 交叉注意力优化多模态特征融合

3. **更现代的训练方法**
   - 基于最新的Transformer技术
   - 优化的损失函数和训练策略
   - 更好的数值稳定性和收敛性

4. **更强的泛化能力**
   - 端到端学习，减少人工特征工程的偏差
   - 注意力机制能够适应不同类型的查询
   - 更好地处理未见过的查询模式

## 🔧 性能调优建议

### 提升精度

1. **增加模型容量**
   ```bash
   --d_model 512 --num_layers 8 --nhead 16
   ```

2. **使用更多训练数据**
   ```bash
   --train_samples 50000
   ```

3. **调整损失函数权重**
   - 修改`QueryCardinalityLoss`中的`alpha`参数

### 加速训练

1. **使用GPU**
   - 确保安装了CUDA版本的PyTorch
   - 模型会自动检测并使用GPU

2. **增大批大小**
   ```bash
   --batch_size 64  # 如果内存允许
   ```

3. **减少序列长度**
   ```bash
   --max_seq_length 256  # 如果查询较短
   ```

### 内存优化

1. **减小模型大小**
   ```bash
   --d_model 128 --num_layers 4
   ```

2. **减小批大小**
   ```bash
   --batch_size 16
   ```

3. **减少训练样本**
   ```bash
   --train_samples 10000
   ```

## 🐛 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减小`batch_size`
   - 减小`d_model`或`num_layers`
   - 使用CPU训练：`--device cpu`

2. **训练速度慢**
   - 检查是否使用了GPU
   - 增大`batch_size`