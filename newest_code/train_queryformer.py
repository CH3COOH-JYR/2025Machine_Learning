import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from queryformer_model import (
    QueryFormerModel, QueryTokenizer, QueryCardinalityLoss, 
    print_metrics, q_error
)
from advanced_data_processor import AdvancedDataProcessor, QueryFormerDataset

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    """自定义批处理函数"""
    token_ids = torch.stack([item['token_ids'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    # 处理计划特征
    plan_features = {
        'node_types': torch.stack([item['plan_features']['node_types'] for item in batch]),
        'costs': torch.stack([item['plan_features']['costs'] for item in batch]),
        'rows': torch.stack([item['plan_features']['rows'] for item in batch]),
        'selectivities': torch.stack([item['plan_features']['selectivities'] for item in batch])
    }
    
    return {
        'token_ids': token_ids,
        'plan_features': plan_features,
        'targets': targets
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动数据到设备
        token_ids = batch['token_ids'].to(device)
        plan_features = {k: v.to(device) for k, v in batch['plan_features'].items()}
        targets = batch['targets'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(token_ids, plan_features)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    return total_loss / num_batches

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 移动数据到设备
            token_ids = batch['token_ids'].to(device)
            plan_features = {k: v.to(device) for k, v in batch['plan_features'].items()}
            targets = batch['targets'].to(device)
            
            # 前向传播
            outputs = model(token_ids, plan_features)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 收集预测和目标
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算评估指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    return avg_loss, predictions, targets

def save_model(model, tokenizer, processor, save_dir, epoch, loss):
    """保存模型和相关组件"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, f'queryformer_epoch_{epoch}_loss_{loss:.4f}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'd_model': model.d_model,
            'max_seq_length': model.max_seq_length
        },
        'epoch': epoch,
        'loss': loss
    }, model_path)
    
    # 保存分词器
    tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # 保存处理器
    processor_path = os.path.join(save_dir, 'processor.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"模型已保存到: {model_path}")
    return model_path

def plot_training_history(train_losses, val_losses, save_path):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='训练QueryFormer模型')
    parser.add_argument('--train_samples', type=int, default=30000, help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--max_seq_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--save_dir', type=str, default='newest_code/models', help='模型保存目录')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='早停最小改进阈值')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建数据处理器
    print("初始化数据处理器...")
    processor = AdvancedDataProcessor()
    
    # 加载训练数据
    print("加载训练数据...")
    train_data, train_cardinalities = processor.load_data(
        'data/train_data.json', 
        max_samples=args.train_samples
    )
    
    if not train_data:
        print("加载训练数据失败！")
        return
    
    # 构建词汇表
    print("构建词汇表...")
    processor.build_vocabularies(train_data)
    
    # 创建分词器
    print("创建分词器...")
    tokenizer = QueryTokenizer()
    queries = [item['query'] for item in train_data]
    tokenizer.build_vocab(queries)
    
    # 创建数据集
    print("创建数据集...")
    full_dataset = QueryFormerDataset(
        train_data, 
        train_cardinalities, 
        processor, 
        tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 改为0避免多进程问题
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # 改为0避免多进程问题
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = QueryFormerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数和优化器
    criterion = QueryCardinalityLoss(alpha=0.5)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    # 早停机制变量
    best_val_loss = float('inf')
    best_model_path = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("开始训练...")
    print(f"早停设置: 耐心值={args.early_stopping_patience}, 最小改进阈值={args.min_delta}")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_predictions, val_targets = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        
        # 计算验证集指标
        print_metrics(val_targets, val_predictions, "验证集 ")
        
        # 早停检查
        improvement = best_val_loss - val_loss
        if improvement > args.min_delta:
            # 有显著改进
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_model(
                model, tokenizer, processor, args.save_dir, epoch, val_loss
            )
            print(f"🎉 新的最佳模型！验证损失: {val_loss:.4f} (改进: {improvement:.4f})")
        else:
            # 没有显著改进
            patience_counter += 1
            print(f"⏳ 验证损失未改进 ({patience_counter}/{args.early_stopping_patience})")
        
        # 检查是否需要早停
        if patience_counter >= args.early_stopping_patience:
            print(f"\n🛑 早停触发！连续 {args.early_stopping_patience} 个epoch验证损失未改进")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            break
        
        print("-" * 50)
    
    # 训练结束总结
    if patience_counter < args.early_stopping_patience:
        print(f"\n✅ 训练完成！完成了所有 {args.epochs} 个epoch")
    else:
        print(f"\n🛑 训练因早停而结束，实际训练了 {epoch} 个epoch")
    
    # 绘制训练历史
    plot_path = os.path.join(args.save_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, plot_path)
    print(f"训练历史图已保存到: {plot_path}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_model_path': best_model_path,
        'early_stopped': patience_counter >= args.early_stopping_patience,
        'total_epochs': epoch,
        'args': vars(args)
    }
    
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"训练完成！最佳模型: {best_model_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    return best_model_path, best_val_loss

if __name__ == '__main__':
    main() 