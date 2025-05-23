#!/usr/bin/env python3
"""
最终优化的SQL基数估计模型训练脚本
- Stacking集成学习
- 精细特征工程和选择
- 高级正则化技术
- 最优超参数调优
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
import re
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class OptimalFeatureExtractor:
    """最优特征提取器"""
    
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
        print("构建最优词汇表...")
        
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
        
        # 保持适中的词汇表大小
        self.table_vocab = sorted(list(self.table_vocab))[:15]
        self.column_vocab = sorted(list(self.column_vocab))[:30]
        self.operator_vocab = sorted(list(self.operator_vocab))
        self.node_type_vocab = sorted(list(self.node_type_vocab))
        
        print(f"最优词汇表大小: 表({len(self.table_vocab)}), 列({len(self.column_vocab)}), 操作符({len(self.operator_vocab)}), 节点类型({len(self.node_type_vocab)})")
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
        """提取最优特征"""
        features = []
        
        # 1. 核心查询特征
        features.extend(self._extract_core_query_features(query))
        
        # 2. 查询计划特征
        features.extend(self._extract_plan_features(explain_result))
        
        # 3. 统计特征
        features.extend(self._extract_statistical_features(query))
        
        # 4. 交互特征
        features.extend(self._extract_interaction_features(query, explain_result))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_core_query_features(self, query):
        """核心查询特征"""
        features = []
        
        # 基本长度特征
        query_len = len(query)
        word_count = len(query.split())
        features.extend([
            np.log1p(query_len),
            np.log1p(word_count),
            query_len / max(1, word_count)  # 平均单词长度
        ])
        
        # 表和列计数
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
        
        unique_tables = len(set(tables))
        unique_columns = len(set(columns))
        
        features.extend([
            unique_tables,
            unique_columns,
            unique_columns / max(1, unique_tables)  # 列表比率
        ])
        
        # 关键字密度
        total_words = max(1, word_count)
        keywords = ['WHERE', 'AND', 'OR', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING']
        keyword_counts = []
        for keyword in keywords:
            count = query.upper().count(keyword)
            keyword_counts.append(count)
            features.append(count / total_words)  # 关键字密度
        
        # 操作符特征
        for op in self.operator_vocab:
            features.append(query.upper().count(op))
        
        # 数值特征 (改进版)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        if numbers:
            nums = [float(n) for n in numbers]
            features.extend([
                len(nums),
                np.log1p(max(nums)),
                np.log1p(min(nums)),
                np.log1p(np.mean(nums)),
                np.log1p(np.std(nums)) if len(nums) > 1 else 0
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 字符串复杂度
        string_literals = re.findall(r"'[^']*'", query)
        features.append(len(string_literals))
        
        return features
    
    def _extract_plan_features(self, explain_result):
        """查询计划特征"""
        features = []
        
        try:
            plan = json.loads(explain_result)
            root_plan = plan['QUERY PLAN'][0]['Plan']
            
            # 成本特征 (log变换)
            total_cost = root_plan.get('Total Cost', 0)
            startup_cost = root_plan.get('Startup Cost', 0)
            plan_rows = root_plan.get('Plan Rows', 1)
            plan_width = root_plan.get('Plan Width', 0)
            
            features.extend([
                np.log1p(total_cost),
                np.log1p(startup_cost),
                np.log1p(plan_rows),
                np.log1p(plan_width),
                np.log1p(total_cost - startup_cost),  # 执行成本
                np.log1p(total_cost / max(1, plan_rows)),  # 单行成本
            ])
            
            # 计划结构
            plan_stats = self._analyze_plan_structure(root_plan)
            features.extend([
                plan_stats['depth'],
                plan_stats['node_count'],
                plan_stats['scan_count'],
                plan_stats['join_count'],
                plan_stats['scan_count'] / max(1, plan_stats['node_count']),  # 扫描比率
                plan_stats['join_count'] / max(1, plan_stats['node_count'])   # 连接比率
            ])
            
            # 节点类型特征
            node_type_counts = Counter()
            self._count_node_types(root_plan, node_type_counts)
            
            scan_types = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']
            join_types = ['Nested Loop', 'Hash Join', 'Merge Join']
            
            scan_count = sum(node_type_counts.get(t, 0) for t in scan_types)
            join_count = sum(node_type_counts.get(t, 0) for t in join_types)
            
            features.extend([
                scan_count,
                join_count,
                node_type_counts.get('Sort', 0),
                node_type_counts.get('Aggregate', 0)
            ])
            
        except Exception as e:
            features.extend([0] * 16)
        
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
            max_values = []
            
            for col in columns:
                if col in self.column_stats:
                    stats = self.column_stats[col]
                    cardinalities.append(stats.get('cardinality', 1))
                    max_values.append(stats.get('max', 1))
            
            if cardinalities:
                features.extend([
                    np.log1p(np.mean(cardinalities)),
                    np.log1p(np.max(cardinalities)),
                    np.log1p(np.min(cardinalities)),
                    np.log1p(np.std(cardinalities)) if len(cardinalities) > 1 else 0,
                    len(cardinalities)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def _extract_interaction_features(self, query, explain_result):
        """交互特征"""
        features = []
        
        # 查询复杂度评分
        complexity_score = 0
        complexity_score += query.upper().count('JOIN') * 2
        complexity_score += query.upper().count('WHERE') * 1
        complexity_score += query.upper().count('GROUP BY') * 3
        complexity_score += query.upper().count('ORDER BY') * 1
        complexity_score += query.count('(') * 1  # 嵌套复杂度
        
        features.append(np.log1p(complexity_score))
        
        # 子查询特征
        subquery_count = query.upper().count('SELECT') - 1
        features.append(subquery_count)
        
        # 谓词选择性估计
        predicates = query.upper().count('=') + query.upper().count('>')  + query.upper().count('<')
        features.append(predicates)
        
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

class StackingEnsembleModel:
    """Stacking集成模型"""
    
    def __init__(self):
        self.stacking_model = None
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        """训练Stacking集成模型"""
        print("训练Stacking集成模型...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 定义基学习器 - 最优超参数
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=120,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=120,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )),
            ('gbr', GradientBoostingRegressor(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                max_features=0.8,
                random_state=42
            ))
        ]
        
        # 元学习器
        meta_model = Ridge(alpha=10.0)
        
        # 创建Stacking模型
        self.stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # 交叉验证评估
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.stacking_model, X_scaled, y, 
                                  cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        
        print(f"Stacking CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 训练最终模型
        self.stacking_model.fit(X_scaled, y)
        
        # 评估基学习器
        print("\n基学习器性能:")
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        for name, model in base_models:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            print(f"{name}: {val_mse:.4f}")
        
        return self
    
    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        return self.stacking_model.predict(X_scaled)

def main():
    print("=== 最终优化的SQL基数估计模型训练 ===")
    
    # 加载训练数据
    print("加载训练数据...")
    train_data = load_data('data/train_data.json', max_samples=38000)
    print(f"训练数据数量: {len(train_data)}")
    
    # 初始化最优特征提取器
    extractor = OptimalFeatureExtractor()
    extractor.fit_vocabularies(train_data)
    
    # 特征提取
    print("提取最优特征...")
    X_train = []
    y_train = []
    
    for sample in tqdm(train_data, desc="提取训练特征"):
        features = extractor.extract_features(sample['query'], sample['explain_result'])
        cardinality = extract_cardinality_from_plan(sample['explain_result'])
        
        X_train.append(features)
        y_train.append(np.log(max(1, cardinality)))  # log scale
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"最优特征维度: {X_train.shape}")
    print(f"目标值范围: {y_train.min():.2f} - {y_train.max():.2f}")
    
    # 数据质量检查和处理
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 特征选择 (更保守)
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesRegressor
    
    # 使用ExtraTreesRegressor进行特征选择
    feature_selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=50, random_state=42),
        threshold='median'
    )
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    
    print(f"特征选择后维度: {X_train_selected.shape}")
    
    # 训练Stacking集成模型
    ensemble = StackingEnsembleModel()
    ensemble.fit(X_train_selected, y_train)
    
    # 训练集评估
    train_pred = ensemble.predict(X_train_selected)
    train_mse = mean_squared_error(y_train, train_pred)
    print(f"\nStacking训练MSE: {train_mse:.4f}")
    
    # 加载测试数据
    print("\n加载测试数据...")
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
    X_test_selected = feature_selector.transform(X_test)
    
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
    print(f"  标准差: {np.std(test_pred):.0f}")
    
    # 显示前10个预测结果
    print(f"\n前10个预测结果:")
    for i in range(min(10, len(test_pred))):
        print(f"  Query {i}: {test_pred[i]}")
    
    print("\n=== 最终优化训练完成 ===")

if __name__ == '__main__':
    main() 