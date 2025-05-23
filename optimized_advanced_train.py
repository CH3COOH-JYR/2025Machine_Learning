#!/usr/bin/env python3
"""
优化的SQL基数估计模型训练脚本
- 加强正则化防止过拟合
- 交叉验证和早停机制
- 更精细的超参数调优
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import re
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureExtractor:
    """增强特征提取器 - 防过拟合版本"""
    
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
        print("构建优化词汇表...")
        
        for sample in tqdm(data, desc="扫描词汇"):
            query = sample['query']
            explain_result = sample['explain_result']
            
            # 提取表名
            tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
            self.table_vocab.update(tables)
            
            # 提取列名
            columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
            self.column_vocab.update(columns)
            
            # 提取操作符
            operators = ['>', '<', '=', '>=', '<=', 'LIKE', 'IN', 'BETWEEN']
            for op in operators:
                if op in query.upper():
                    self.operator_vocab.add(op)
            
            # 提取查询计划节点类型
            try:
                plan = json.loads(explain_result)
                self._extract_node_types_recursive(plan.get('QUERY PLAN', [{}])[0].get('Plan', {}))
            except:
                pass
        
        # 限制词汇表大小以防过拟合
        self.table_vocab = sorted(list(self.table_vocab))[:20]  # 限制表数量
        self.column_vocab = sorted(list(self.column_vocab))[:50]  # 限制列数量
        self.operator_vocab = sorted(list(self.operator_vocab))
        self.node_type_vocab = sorted(list(self.node_type_vocab))
        
        print(f"优化词汇表大小: 表({len(self.table_vocab)}), 列({len(self.column_vocab)}), 操作符({len(self.operator_vocab)}), 节点类型({len(self.node_type_vocab)})")
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
        """提取优化特征"""
        features = []
        
        # 1. 核心基本特征
        features.extend(self._extract_core_basic_features(query))
        
        # 2. 查询计划核心特征
        features.extend(self._extract_core_plan_features(explain_result))
        
        # 3. 统计特征
        features.extend(self._extract_statistical_features(query))
        
        # 4. 简化的复杂度特征
        features.extend(self._extract_simple_complexity_features(query))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_core_basic_features(self, query):
        """核心基本查询特征"""
        features = []
        
        # 基本长度特征
        features.append(len(query))
        features.append(len(query.split()))
        
        # 表和列数量
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        features.append(len(set(tables)))
        
        columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
        features.append(len(set(columns)))
        
        # 关键字统计
        keywords = ['WHERE', 'AND', 'OR', 'JOIN', 'GROUP BY', 'ORDER BY']
        for keyword in keywords:
            features.append(query.upper().count(keyword))
        
        # 操作符统计
        for op in self.operator_vocab:
            features.append(query.upper().count(op))
        
        # 数值特征
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        if numbers:
            nums = [float(n) for n in numbers]
            features.extend([
                len(nums),
                np.log1p(max(nums)),  # 使用log1p避免极值
                np.log1p(min(nums)),
                np.log1p(np.mean(nums))
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_core_plan_features(self, explain_result):
        """核心查询计划特征"""
        features = []
        
        try:
            plan = json.loads(explain_result)
            root_plan = plan['QUERY PLAN'][0]['Plan']
            
            # 核心成本特征
            total_cost = root_plan.get('Total Cost', 0)
            startup_cost = root_plan.get('Startup Cost', 0)
            plan_rows = root_plan.get('Plan Rows', 1)
            plan_width = root_plan.get('Plan Width', 0)
            
            # 使用log变换避免极值
            features.extend([
                np.log1p(total_cost),
                np.log1p(startup_cost),
                np.log1p(plan_rows),
                np.log1p(plan_width)
            ])
            
            # 计划结构特征
            plan_stats = self._analyze_plan_structure(root_plan)
            features.extend([
                plan_stats['depth'],
                plan_stats['node_count'],
                plan_stats['scan_count'],
                plan_stats['join_count']
            ])
            
            # 主要节点类型特征
            node_type_counts = Counter()
            self._count_node_types(root_plan, node_type_counts)
            
            # 简化的节点类型特征
            scan_count = sum(node_type_counts.get(t, 0) for t in ['Seq Scan', 'Index Scan', 'Index Only Scan'])
            join_count = sum(node_type_counts.get(t, 0) for t in ['Nested Loop', 'Hash Join', 'Merge Join'])
            
            features.extend([scan_count, join_count])
            
        except Exception as e:
            # 填充默认值
            features.extend([0] * 10)
        
        return features
    
    def _analyze_plan_structure(self, plan, depth=0):
        """分析计划结构"""
        stats = {
            'depth': depth,
            'node_count': 1,
            'scan_count': 1 if 'Scan' in plan.get('Node Type', '') else 0,
            'join_count': 1 if 'Join' in plan.get('Node Type', '') else 0
        }
        
        for subplan in plan.get('Plans', []):
            substats = self._analyze_plan_structure(subplan, depth + 1)
            stats['depth'] = max(stats['depth'], substats['depth'])
            stats['node_count'] += substats['node_count']
            stats['scan_count'] += substats['scan_count']
            stats['join_count'] += substats['join_count']
        
        return stats
    
    def _count_node_types(self, plan, counter):
        """统计节点类型"""
        node_type = plan.get('Node Type', '')
        if node_type:
            counter[node_type] += 1
        
        for subplan in plan.get('Plans', []):
            self._count_node_types(subplan, counter)
    
    def _extract_statistical_features(self, query):
        """统计特征"""
        features = []
        
        columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
        if columns and self.column_stats:
            cardinalities = []
            for col in columns:
                if col in self.column_stats:
                    stats = self.column_stats[col]
                    cardinalities.append(stats.get('cardinality', 1))
            
            if cardinalities:
                features.extend([
                    np.log1p(np.mean(cardinalities)),
                    np.log1p(np.max(cardinalities)),
                    len(cardinalities)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_simple_complexity_features(self, query):
        """简化的复杂度特征"""
        features = []
        
        # 括号嵌套
        features.append(query.count('('))
        
        # 子查询
        subquery_count = query.upper().count('SELECT') - 1
        features.append(max(0, subquery_count))
        
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

class RegularizedEnsembleModel:
    """正则化集成模型"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        """训练集成模型"""
        print("训练正则化集成模型...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用交叉验证评估模型
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 定义模型 - 增强正则化
        models_config = {
            'xgb': xgb.XGBRegressor(
                n_estimators=150,  # 减少树数量
                max_depth=6,       # 减少深度
                learning_rate=0.05, # 降低学习率
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,       # L1正则化
                reg_lambda=2,      # L2正则化
                random_state=42,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=100,  # 减少树数量
                max_depth=5,       # 减少深度
                learning_rate=0.05,
                subsample=0.7,
                max_features=0.7,  # 特征采样
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,       # 限制深度
                min_samples_split=10,  # 增加最小分割样本
                min_samples_leaf=5,    # 增加最小叶子样本
                max_features=0.7,      # 特征采样
                random_state=42,
                n_jobs=-1
            )
        }
        
        # 交叉验证评估
        cv_scores = {}
        for name, model in models_config.items():
            print(f"交叉验证 {name}...")
            scores = cross_val_score(model, X_scaled, y, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            cv_score = -scores.mean()
            cv_scores[name] = cv_score
            print(f"{name} CV MSE: {cv_score:.4f} (+/- {scores.std() * 2:.4f})")
        
        # 训练最终模型
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        val_scores = {}
        for name, model in models_config.items():
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, val_pred)
            val_scores[name] = val_score
            self.models[name] = model
        
        # 计算权重 - 结合CV和验证分数
        combined_scores = {}
        for name in models_config.keys():
            combined_scores[name] = (cv_scores[name] + val_scores[name]) / 2
        
        # 基于性能计算权重
        total_inv_score = sum(1.0 / score for score in combined_scores.values())
        for name, score in combined_scores.items():
            self.weights[name] = (1.0 / score) / total_inv_score
        
        print("模型权重:", {name: f"{weight:.3f}" for name, weight in self.weights.items()})
        
        return self
    
    def predict(self, X):
        """集成预测"""
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)

def main():
    print("=== 优化的SQL基数估计模型训练 ===")
    
    # 加载训练数据 - 使用适中的数据量
    print("加载训练数据...")
    train_data = load_data('data/train_data.json', max_samples=35000)
    print(f"训练数据数量: {len(train_data)}")
    
    # 初始化优化特征提取器
    extractor = EnhancedFeatureExtractor()
    extractor.fit_vocabularies(train_data)
    
    # 特征提取
    print("提取优化特征...")
    X_train = []
    y_train = []
    
    for sample in tqdm(train_data, desc="提取训练特征"):
        features = extractor.extract_features(sample['query'], sample['explain_result'])
        cardinality = extract_cardinality_from_plan(sample['explain_result'])
        
        X_train.append(features)
        y_train.append(np.log(max(1, cardinality)))  # log scale
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"优化特征维度: {X_train.shape}")
    print(f"目标值范围: {y_train.min():.2f} - {y_train.max():.2f}")
    
    # 检查数据质量
    print(f"特征缺失值: {np.isnan(X_train).sum()}")
    print(f"特征无穷值: {np.isinf(X_train).sum()}")
    
    # 处理异常值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 添加特征选择
    from sklearn.feature_selection import SelectKBest, f_regression
    selector = SelectKBest(score_func=f_regression, k=min(50, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    print(f"特征选择后维度: {X_train_selected.shape}")
    
    # 训练正则化集成模型
    ensemble = RegularizedEnsembleModel()
    ensemble.fit(X_train_selected, y_train)
    
    # 训练集评估
    train_pred = ensemble.predict(X_train_selected)
    train_mse = mean_squared_error(y_train, train_pred)
    print(f"训练正则化MSE: {train_mse:.4f}")
    
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
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 应用特征选择
    X_test_selected = selector.transform(X_test)
    
    # 预测
    print("预测测试数据...")
    test_pred_log = ensemble.predict(X_test_selected)
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
    
    print("\n=== 优化训练完成 ===")

if __name__ == '__main__':
    main() 