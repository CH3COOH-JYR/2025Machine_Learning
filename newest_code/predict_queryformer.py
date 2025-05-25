import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from queryformer_model import QueryFormerModel, QueryTokenizer
from advanced_data_processor import AdvancedDataProcessor, QueryFormerDataset

def load_model_and_components(model_dir):
    """加载模型和相关组件"""
    # 加载分词器
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # 加载处理器
    processor_path = os.path.join(model_dir, 'processor.pkl')
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # 查找最佳模型文件
    model_files = [f for f in os.listdir(model_dir) if f.startswith('queryformer_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("未找到模型文件")
    
    # 选择损失最小的模型
    best_model_file = min(model_files, key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
    model_path = os.path.join(model_dir, best_model_file)
    
    print(f"加载模型: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    model = QueryFormerModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        max_seq_length=model_config['max_seq_length']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, processor, checkpoint

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

def predict(model, dataloader, device):
    """进行预测"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            # 移动数据到设备
            token_ids = batch['token_ids'].to(device)
            plan_features = {k: v.to(device) for k, v in batch['plan_features'].items()}
            
            # 前向传播
            outputs = model(token_ids, plan_features)
            
            # 收集预测结果
            predictions = outputs.cpu().numpy().flatten()
            all_predictions.extend(predictions)
    
    return np.array(all_predictions)

def main():
    parser = argparse.ArgumentParser(description='使用QueryFormer模型进行预测')
    parser.add_argument('--model_dir', type=str, default='newest_code/models', help='模型目录')
    parser.add_argument('--test_file', type=str, default='data/test_data.json', help='测试数据文件')
    parser.add_argument('--output_file', type=str, default='newest_code/queryformer_predictions.csv', help='输出文件')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--max_seq_length', type=int, default=512, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载模型和组件
    print("加载模型和组件...")
    model, tokenizer, processor, checkpoint = load_model_and_components(args.model_dir)
    model = model.to(device)
    
    print(f"模型训练轮数: {checkpoint['epoch']}")
    print(f"模型验证损失: {checkpoint['loss']:.4f}")
    
    # 加载测试数据
    print("加载测试数据...")
    test_data, _ = processor.load_data(args.test_file)
    
    if not test_data:
        print("加载测试数据失败！")
        return
    
    print(f"测试样本数量: {len(test_data)}")
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dataset = QueryFormerDataset(
        test_data, 
        None,  # 测试数据没有标签
        processor, 
        tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # 改为0避免多进程问题
    )
    
    # 进行预测
    print("开始预测...")
    log_predictions = predict(model, test_loader, device)
    
    # 转换回原始空间
    predictions = np.exp(np.clip(log_predictions, 0, 20))
    predictions = np.round(predictions).astype(int)
    
    # 确保预测值至少为1
    predictions = np.maximum(predictions, 1)
    
    print(f"预测完成！")
    print(f"预测值统计:")
    print(f"  最小值: {np.min(predictions)}")
    print(f"  最大值: {np.max(predictions)}")
    print(f"  中位数: {np.median(predictions)}")
    print(f"  平均值: {np.mean(predictions):.0f}")
    
    # 创建结果DataFrame
    results = []
    for i, prediction in enumerate(predictions):
        results.append({
            'query_id': i,
            'cardinality': prediction
        })
    
    df = pd.DataFrame(results)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"预测结果已保存到: {args.output_file}")
    
    # 保存详细统计信息
    stats = {
        'total_samples': len(predictions),
        'min_prediction': int(np.min(predictions)),
        'max_prediction': int(np.max(predictions)),
        'median_prediction': int(np.median(predictions)),
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'model_info': {
            'epoch': checkpoint['epoch'],
            'validation_loss': checkpoint['loss'],
            'model_config': checkpoint['model_config']
        }
    }
    
    stats_file = args.output_file.replace('.csv', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息已保存到: {stats_file}")

if __name__ == '__main__':
    main() 