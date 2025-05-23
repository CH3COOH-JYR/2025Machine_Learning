#!/usr/bin/env python3
"""
高级SQL基数估计模型训练脚本
结合深度学习和丰富的特征工程
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
from collections import Counter

class AdvancedFeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self):
        self.column_stats = self._load_column_stats()
        self.table_vocab = set()
        self.column_vocab = set()
        self.operator_vocab = set()
        self.node_type_vocab = set()
        self.fitted = False
        
    def _load_column_stats(self):
        """加载列统计信息"""
        try:
            df = pd.read_csv('data/column_min_max_vals.csv')
            return df.set_index('name').to_dict('index')
        except:
            return {}
    
    def fit_vocabularies(self, data):
        """构建词汇表"""
        print("构建词汇表...")
        
        for sample in tqdm(data, desc="扫描词汇"):
            query = sample['query']
            explain_result = sample['explain_result']
            
            # 提取表名
            tables = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
            self.table_vocab.update(tables)
            
            # 提取列名
            columns = re.findall(r'(\w+\.\w+)', query)
            self.column_vocab.update(columns)
            
            # 提取操作符
            if '>' in query:
                self.operator_vocab.add('>')
            if '<' in query:
                self.operator_vocab.add('<')
            if '=' in query:
                self.operator_vocab.add('=')
            if 'LIKE' in query.upper():
                self.operator_vocab.add('LIKE')
            
            # 提取查询计划节点类型
            try:
                plan = json.loads(explain_result)
                self._extract_node_types_recursive(plan.get('QUERY PLAN', [{}])[0].get('Plan', {}))
            except:
                pass
        
        self.table_vocab = sorted(list(self.table_vocab))
        self.column_vocab = sorted(list(self.column_vocab))
        self.operator_vocab = sorted(list(self.operator_vocab))
        self.node_type_vocab = sorted(list(self.node_type_vocab))
        
        print(f"词汇表大小: 表({len(self.table_vocab)}), 列({len(self.column_vocab)}), 操作符({len(self.operator_vocab)}), 节点类型({len(self.node_type_vocab)})")
        self.fitted = True
    
    def _extract_node_types_recursive(self, plan):
        """递归提取节点类型"""
        if not plan:
            return
        
        node_type = plan.get('Node Type', '')
        if node_type:
            self.node_type_vocab.add(node_type)
        
        for subplan in plan.get('Plans', []):
            self._extract_node_types_recursive(subplan)
    
    def extract_features(self, query, explain_result):
        """提取高级特征"""
        features = []
        
        # 1. 基本查询特征
        features.extend(self._extract_basic_features(query))
        
        # 2. 表和列特征
        features.extend(self._extract_schema_features(query))
        
        # 3. 谓词特征
        features.extend(self._extract_predicate_features(query))
        
        # 4. 查询计划特征
        features.extend(self._extract_plan_features(explain_result))
        
        # 5. 统计特征
        features.extend(self._extract_statistical_features(query))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_basic_features(self, query):
        """基本查询特征"""
        features = []
        
        # 查询长度
        features.append(len(query))
        features.append(len(query.split()))
        
        # 表数量
        tables = re.findall(r'FROM\s+([^,\s]+(?:\s*,\s*[^,\s]+)*)', query, re.IGNORECASE)
        table_count = len(tables[0].split(',')) if tables else 0
        features.append(table_count)
        
        # 关键字统计
        features.append(query.upper().count('SELECT'))
        features.append(query.upper().count('WHERE'))
        features.append(query.upper().count('AND'))
        features.append(query.upper().count('OR'))
        features.append(query.upper().count('JOIN'))
        features.append(query.upper().count('INNER'))
        features.append(query.upper().count('LEFT'))
        features.append(query.upper().count('GROUP BY'))
        features.append(query.upper().count('ORDER BY'))
        
        return features
    
    def _extract_schema_features(self, query):
        """表和列特征"""
        features = []
        
        # 表特征向量
        table_vector = np.zeros(len(self.table_vocab))
        tables = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
        for table in tables:
            if table in self.table_vocab:
                idx = self.table_vocab.index(table)
                table_vector[idx] = 1
        
        # 列特征向量
        column_vector = np.zeros(len(self.column_vocab))
        columns = re.findall(r'(\w+\.\w+)', query)
        for column in columns:
            if column in self.column_vocab:
                idx = self.column_vocab.index(column)
                column_vector[idx] = 1
        
        # 统计特征
        features.append(np.sum(table_vector))  # 涉及的表数量
        features.append(np.sum(column_vector))  # 涉及的列数量
        
        # 如果词汇表不太大，可以包含one-hot编码
        if len(self.table_vocab) <= 50:
            features.extend(table_vector)
        if len(self.column_vocab) <= 100:
            features.extend(column_vector)
        
        return features
    
    def _extract_predicate_features(self, query):
        """谓词特征"""
        features = []
        
        # 操作符统计
        for op in self.operator_vocab:
            features.append(query.count(op))
        
        # 数值范围特征
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            nums = [int(n) for n in numbers]
            features.append(len(nums))
            features.append(max(nums))
            features.append(min(nums))
            features.append(np.mean(nums))
            features.append(np.std(nums) if len(nums) > 1 else 0)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 字符串常量数量
        string_literals = len(re.findall(r"'[^']*'", query))
        features.append(string_literals)
        
        return features
    
    def _extract_plan_features(self, explain_result):
        """查询计划特征"""
        features = []
        
        try:
            plan = json.loads(explain_result)
            root_plan = plan['QUERY PLAN'][0]['Plan']
            
            # 基本成本和行数信息
            features.append(root_plan.get('Total Cost', 0))
            features.append(root_plan.get('Startup Cost', 0))
            features.append(root_plan.get('Plan Rows', 1))
            features.append(root_plan.get('Plan Width', 0))
            
            # 计划结构特征
            plan_stats = self._analyze_plan_structure(root_plan)
            features.extend([
                plan_stats['depth'],
                plan_stats['node_count'],
                plan_stats['scan_count'],
                plan_stats['join_count'],
                plan_stats['total_cost'],
                plan_stats['total_rows']
            ])
            
            # 节点类型特征
            node_types = Counter()
            self._count_node_types(root_plan, node_types)
            for node_type in self.node_type_vocab:
                features.append(node_types.get(node_type, 0))
            
        except Exception as e:
            # 如果解析失败，填充默认值
            default_count = 4 + 6 + len(self.node_type_vocab)
            features.extend([0] * default_count)
        
        return features
    
    def _analyze_plan_structure(self, plan, depth=0):
        """分析计划结构"""
        stats = {
            'depth': depth,
            'node_count': 1,
            'scan_count': 1 if 'Scan' in plan.get('Node Type', '') else 0,
            'join_count': 1 if 'Join' in plan.get('Node Type', '') else 0,
            'total_cost': plan.get('Total Cost', 0),
            'total_rows': plan.get('Plan Rows', 0)
        }
        
        for subplan in plan.get('Plans', []):
            substats = self._analyze_plan_structure(subplan, depth + 1)
            stats['depth'] = max(stats['depth'], substats['depth'])
            stats['node_count'] += substats['node_count']
            stats['scan_count'] += substats['scan_count']
            stats['join_count'] += substats['join_count']
            stats['total_cost'] += substats['total_cost']
            stats['total_rows'] += substats['total_rows']
        
        return stats
    
    def _count_node_types(self, plan, counter):
        """统计节点类型"""
        node_type = plan.get('Node Type', '')
        if node_type:
            counter[node_type] += 1
        
        for subplan in plan.get('Plans', []):
            self._count_node_types(subplan, counter)
    
    def _extract_statistical_features(self, query):
        """统计特征（基于列统计信息）"""
        features = []
        
        # 从列统计信息中提取特征
        columns = re.findall(r'(\w+\.\w+)', query)
        if columns and self.column_stats:
            cardinalities = []
            max_values = []
            min_values = []
            
            for col in columns:
                if col in self.column_stats:
                    stats = self.column_stats[col]
                    cardinalities.append(stats.get('cardinality', 0))
                    max_values.append(stats.get('max', 0))
                    min_values.append(stats.get('min', 0))
            
            if cardinalities:
                features.append(np.mean(cardinalities))
                features.append(np.max(cardinalities))
                features.append(np.mean(max_values))
                features.append(np.mean(min_values))
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        return features

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
    print("=== 高级SQL基数估计模型训练 ===")
    
    # 加载训练数据
    print("加载训练数据...")
    train_data = load_data('data/train_data.json', max_samples=30000)  # 使用更多数据
    print(f"训练数据数量: {len(train_data)}")
    
    # 初始化特征提取器
    extractor = AdvancedFeatureExtractor()
    extractor.fit_vocabularies(train_data)
    
    # 特征提取
    print("提取特征...")
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
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 训练模型
    print("训练梯度提升模型...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 验证
    train_pred = model.predict(X_train_scaled)
    mse = mean_squared_error(y_train, train_pred)
    print(f"训练MSE: {mse:.4f}")
    
    # 特征重要性
    feature_importance = model.feature_importances_
    print(f"平均特征重要性: {np.mean(feature_importance):.4f}")
    print(f"最大特征重要性: {np.max(feature_importance):.4f}")
    
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
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    print("预测测试数据...")
    test_pred_log = model.predict(X_test_scaled)
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