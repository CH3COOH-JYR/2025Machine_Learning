#!/usr/bin/env python3
"""
QueryFormer快速启动脚本
用于演示训练和预测的完整流程
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} 完成")
    else:
        print(f"❌ {description} 失败")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='QueryFormer快速启动')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='运行模式: train(仅训练), predict(仅预测), full(完整流程)')
    parser.add_argument('--quick', action='store_true', 
                       help='快速模式（小规模数据，适合测试）')
    
    args = parser.parse_args()
    
    print("🎯 QueryFormer: 基于Transformer的SQL查询基数估计")
    print("=" * 60)
    
    # 检查数据文件
    if not os.path.exists('data/train_data.json'):
        print("❌ 未找到训练数据文件: data/train_data.json")
        return
    
    if not os.path.exists('data/test_data.json'):
        print("❌ 未找到测试数据文件: data/test_data.json")
        return
    
    print("✅ 数据文件检查通过")
    
    # 设置参数
    if args.quick:
        train_params = {
            'train_samples': 5000,
            'epochs': 10,
            'batch_size': 16,
            'd_model': 128,
            'num_layers': 4,
            'nhead': 4
        }
        print("🔧 使用快速模式参数")
    else:
        train_params = {
            'train_samples': 30000,
            'epochs': 50,
            'batch_size': 32,
            'd_model': 256,
            'num_layers': 6,
            'nhead': 8
        }
        print("🔧 使用标准模式参数")
    
    # 训练阶段
    if args.mode in ['train', 'full']:
        train_cmd = f"""python newest_code/train_queryformer.py \
            --train_samples {train_params['train_samples']} \
            --epochs {train_params['epochs']} \
            --batch_size {train_params['batch_size']} \
            --d_model {train_params['d_model']} \
            --num_layers {train_params['num_layers']} \
            --nhead {train_params['nhead']} \
            --learning_rate 1e-4 \
            --early_stopping_patience 8 \
            --min_delta 1e-4 \
            --save_dir newest_code/models"""
        
        if not run_command(train_cmd, "训练QueryFormer模型"):
            return
    
    # 预测阶段
    if args.mode in ['predict', 'full']:
        # 检查模型是否存在
        if not os.path.exists('newest_code/models'):
            print("❌ 未找到训练好的模型，请先运行训练")
            return
        
        predict_cmd = """python newest_code/predict_queryformer.py \
            --model_dir newest_code/models \
            --test_file data/test_data.json \
            --output_file newest_code/queryformer_predictions.csv \
            --batch_size 64"""
        
        if not run_command(predict_cmd, "生成预测结果"):
            return
    
    # 显示结果
    print(f"\n{'='*60}")
    print("🎉 QueryFormer流程完成！")
    print(f"{'='*60}")
    
    if args.mode in ['train', 'full']:
        print("📊 训练结果:")
        if os.path.exists('newest_code/models/training_history.json'):
            print("  - 训练历史: newest_code/models/training_history.json")
        if os.path.exists('newest_code/models/training_history.png'):
            print("  - 训练曲线: newest_code/models/training_history.png")
        
        # 查找最佳模型
        models_dir = 'newest_code/models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) 
                          if f.startswith('queryformer_') and f.endswith('.pth')]
            if model_files:
                best_model = min(model_files, 
                               key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
                print(f"  - 最佳模型: {models_dir}/{best_model}")
    
    if args.mode in ['predict', 'full']:
        print("🔮 预测结果:")
        if os.path.exists('newest_code/queryformer_predictions.csv'):
            print("  - 预测文件: newest_code/queryformer_predictions.csv")
        if os.path.exists('newest_code/queryformer_predictions_stats.json'):
            print("  - 统计信息: newest_code/queryformer_predictions_stats.json")
    
    print("\n📖 更多信息请查看: newest_code/README.md")
    print("🔧 参数调优请参考README中的推荐配置")

if __name__ == '__main__':
    main() 