#!/usr/bin/env python3
"""
增强的SQL基数估计模型训练脚本
结合SQL查询特征和查询计划特征进行训练
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from enhanced_model import EnhancedCardinalityModel, QueryCardinalityDataset, CardinalityLoss, print_metrics

def collate_fn(batch):
    """自定义collate函数，处理不同特征的批处理"""
    features_batch = {
        'table_features': [],
        'predicate_features': [],
        'join_features': [],
        'plan_features': []
    }
    targets = []
    
    for features, target in batch:
        for key in features_batch.keys():
            features_batch[key].append(features[key])
        targets.append(target)
    
    # 转换为张量
    for key in features_batch.keys():
        features_batch[key] = torch.stack(features_batch[key])
    
    targets = torch.stack(targets)
    
    return features_batch, targets

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch_features, batch_targets in pbar:
            # 移动到设备
            for key in batch_features:
                batch_features[key] = batch_features[key].to(device)
            batch_targets = batch_targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation') as pbar:
            for batch_features, batch_targets in pbar:
                # 移动到设备
                for key in batch_features:
                    batch_features[key] = batch_features[key].to(device)
                batch_targets = batch_targets.to(device)
                
                # 前向传播
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    return avg_loss, predictions, targets

def predict_test_data(model, test_dataloader, device):
    """预测测试数据"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(test_dataloader, desc='Predicting') as pbar:
            for batch_features, _ in pbar:
                # 移动到设备
                for key in batch_features:
                    batch_features[key] = batch_features[key].to(device)
                
                # 前向传播
                outputs = model(batch_features)
                all_predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(all_predictions).flatten()
    # 限制预测值范围，避免exp溢出
    predictions = np.clip(predictions, 0, 20)  # 限制在合理范围内
    # 转换回原始空间
    predictions = np.exp(predictions)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='训练增强的SQL基数估计模型')
    parser.add_argument('--train_samples', type=int, default=20000, help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='融合层数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心')
    parser.add_argument('--output_name', type=str, default='任宣宇_2023202303', help='输出文件名')
    
    args = parser.parse_args()
    
    print("=== 增强SQL基数估计模型训练 ===")
    print(f"训练样本: {args.train_samples}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_data, train_cardinalities = processor.load_data('data/train_data.json', args.train_samples)
    print(f"训练数据数量: {len(train_data)}")
    
    # 构建词汇表
    print("\n构建词汇表...")
    processor.build_vocabularies(train_data)
    
    # 创建数据集
    print("\n创建数据集...")
    full_dataset = QueryCardinalityDataset(train_data, train_cardinalities, processor)
    
    # 划分训练和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # 创建模型
    print("\n创建模型...")
    model = EnhancedCardinalityModel(
        table_vocab_size=len(processor.table2vec),
        predicate_vocab_size=len(processor.column2vec) + len(processor.op2vec) + 1,
        join_vocab_size=len(processor.join2vec),
        plan_vocab_size=len(processor.node_type2vec) + 3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数和优化器
    criterion = CardinalityLoss(alpha=0.7)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_predictions, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        
        # 计算验证集Q-Error
        print_metrics(val_targets, val_predictions, "验证集 ")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'processor': processor,
                'args': args
            }, 'best_model.pth')
            print("保存最佳模型!")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"验证损失连续{args.patience}轮未改善，提前停止训练")
            break
    
    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_data, _ = processor.load_data('data/test_data.json')
    print(f"测试数据数量: {len(test_data)}")
    
    # 创建测试数据集
    test_dataset = QueryCardinalityDataset(test_data, None, processor)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # 预测测试数据
    print("\n预测测试数据...")
    test_predictions = predict_test_data(model, test_loader, device)
    
    # 保存预测结果
    output_filename = f"预测结果_{args.output_name}.csv"
    results_df = pd.DataFrame({
        'query_id': range(len(test_predictions)),
        'cardinality': test_predictions.astype(int)
    })
    results_df.to_csv(output_filename, index=False)
    print(f"\n预测结果已保存到: {output_filename}")
    
    # 打印预测统计信息
    print(f"\n预测统计:")
    print(f"  最小值: {test_predictions.min():.0f}")
    print(f"  最大值: {test_predictions.max():.0f}")
    print(f"  中位数: {np.median(test_predictions):.0f}")
    print(f"  平均值: {np.mean(test_predictions):.0f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(test_predictions, bins=50, alpha=0.7)
    plt.xlabel('预测基数')
    plt.ylabel('频次')
    plt.title('测试集预测分布')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("\n训练结果图表已保存到: training_results.png")
    
    print("\n=== 训练完成 ===")

if __name__ == '__main__':
    main() 