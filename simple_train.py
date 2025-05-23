#!/usr/bin/env python3
"""
简化的SQL基数估计模型训练脚本
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
from tqdm import tqdm

class SimpleFeatureExtractor:
    """简化的特征提取器"""
    
    def __init__(self):
        self.table_features = {}
        self.column_features = {}
        
    def extract_features(self, query, explain_result):
        """提取简单的查询特征"""
        features = []
        
        # 1. 查询长度
        features.append(len(query))
        
        # 2. 表数量
        table_count = len(re.findall(r'FROM\s+([^,\s]+(?:\s*,\s*[^,\s]+)*)', query, re.IGNORECASE))
        features.append(table_count)
        
        # 3. WHERE条件数量
        where_count = len(re.findall(r'WHERE|AND|OR', query, re.IGNORECASE))
        features.append(where_count)
        
        # 4. 连接数量
        join_count = len(re.findall(r'=.*\.', query))
        features.append(join_count)
        
        # 5. 从查询计划中提取特征
        try:
            plan = json.loads(explain_result)
            if 'QUERY PLAN' in plan:
                root_plan = plan['QUERY PLAN'][0]['Plan']
                features.append(root_plan.get('Total Cost', 0))
                features.append(root_plan.get('Plan Rows', 1))
                features.append(len(str(root_plan.get('Node Type', ''))))
            else:
                features.extend([0, 1, 0])
        except:
            features.extend([0, 1, 0])
        
        # 6. 数值范围特征
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            nums = [int(n) for n in numbers]
            features.append(max(nums))
            features.append(min(nums))
            features.append(len(nums))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)

def extract_cardinality_from_plan(explain_result_str):
    """从查询计划中提取cardinality"""
    try:
        explain_result = json.loads(explain_result_str)
        if 'QUERY PLAN' in explain_result:
            plan = explain_result['QUERY PLAN'][0]['Plan']
            return plan.get('Plan Rows', 1)
    except:
        pass
    return 1

def load_data(file_path, max_samples=None):
    """加载数据"""
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    if content.startswith('['):
        data = json.loads(content)
        if max_samples:
            data = data[:max_samples]
    else:
        data = []
        for line in content.split('\n'):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
                if max_samples and len(data) >= max_samples:
                    break
    
    return data

def main():
    print("=== 简化SQL基数估计模型训练 ===")
    
    # 加载训练数据
    print("加载训练数据...")
    train_data = load_data('data/train_data.json', max_samples=10000)
    print(f"训练数据数量: {len(train_data)}")
    
    # 特征提取
    print("提取特征...")
    extractor = SimpleFeatureExtractor()
    
    X_train = []
    y_train = []
    
    for sample in tqdm(train_data, desc="提取训练特征"):
        features = extractor.extract_features(sample['query'], sample['explain_result'])
        cardinality = extract_cardinality_from_plan(sample['explain_result'])
        
        X_train.append(features)
        y_train.append(np.log(max(1, cardinality)))  # log scale
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"特征维度: {X_train.shape}")
    print(f"目标值范围: {y_train.min():.2f} - {y_train.max():.2f}")
    
    # 训练模型
    print("训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 验证
    train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, train_pred)
    print(f"训练MSE: {mse:.4f}")
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = load_data('data/test_data.json')
    print(f"测试数据数量: {len(test_data)}")
    
    # 提取测试特征
    print("提取测试特征...")
    X_test = []
    for sample in tqdm(test_data, desc="提取测试特征"):
        features = extractor.extract_features(sample['query'], sample['explain_result'])
        X_test.append(features)
    
    X_test = np.array(X_test)
    
    # 预测
    print("预测测试数据...")
    test_pred_log = model.predict(X_test)
    test_pred = np.exp(test_pred_log)  # 转换回原始空间
    
    # 确保预测值合理
    test_pred = np.clip(test_pred, 1, 1e9).astype(int)
    
    # 保存结果
    results_df = pd.DataFrame({
        'query_id': range(len(test_pred)),
        'cardinality': test_pred
    })
    
    output_filename = "预测结果_任宣宇_2023202303.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"预测结果已保存到: {output_filename}")
    
    # 统计信息
    print(f"\n预测统计:")
    print(f"  最小值: {test_pred.min()}")
    print(f"  最大值: {test_pred.max()}")
    print(f"  中位数: {np.median(test_pred)}")
    print(f"  平均值: {np.mean(test_pred):.0f}")
    
    # 显示前10个预测结果
    print(f"\n前10个预测结果:")
    for i in range(min(10, len(test_pred))):
        print(f"  Query {i}: {test_pred[i]}")
    
    print("\n=== 训练完成 ===")

if __name__ == '__main__':
    main() 