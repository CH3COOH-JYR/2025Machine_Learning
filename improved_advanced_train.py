#!/usr/bin/env python3
"""
改进的SQL基数估计模型训练脚本
- 使用XGBoost/LightGBM等先进模型
- 增强特征工程
- 模型集成
- 交叉验证和超参数优化
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    """增强特征提取器"""
    
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
        print("构建增强词汇表...")
        
        for sample in tqdm(data, desc="扫描词汇"):
            query = sample['query']
            explain_result = sample['explain_result']
            
            # 提取表名 - 更全面的正则表达式
            tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
            self.table_vocab.update(tables)
            
            # 提取列名 - 包括别名
            columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
            columns.extend(re.findall(r'([a-zA-Z_]\w*)\s*[<>=]', query))
            self.column_vocab.update(columns)
            
            # 提取操作符 - 更完整
            operators = ['>', '<', '=', '>=', '<=', '<>', '!=', 'LIKE', 'ILIKE', 'IN', 'NOT IN', 'BETWEEN']
            for op in operators:
                if op in query.upper():
                    self.operator_vocab.add(op)
            
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
        
        print(f"增强词汇表大小: 表({len(self.table_vocab)}), 列({len(self.column_vocab)}), 操作符({len(self.operator_vocab)}), 节点类型({len(self.node_type_vocab)})")
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
        """提取增强特征"""
        features = []
        
        # 1. 基本查询特征 (增强版)
        features.extend(self._extract_enhanced_basic_features(query))
        
        # 2. 表和列特征
        features.extend(self._extract_schema_features(query))
        
        # 3. 谓词特征 (增强版)
        features.extend(self._extract_enhanced_predicate_features(query))
        
        # 4. 查询计划特征 (增强版)
        features.extend(self._extract_enhanced_plan_features(explain_result))
        
        # 5. 统计特征 (增强版)
        features.extend(self._extract_enhanced_statistical_features(query))
        
        # 6. 新增：复杂度特征
        features.extend(self._extract_complexity_features(query, explain_result))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_enhanced_basic_features(self, query):
        """增强基本查询特征"""
        features = []
        
        # 查询长度相关
        features.append(len(query))
        features.append(len(query.split()))
        features.append(query.count('\n'))  # 行数
        features.append(len(set(query.lower().split())))  # 唯一单词数
        
        # 表相关
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        features.append(len(set(tables)))  # 唯一表数量
        
        # 关键字统计 (增强版)
        keywords = ['SELECT', 'WHERE', 'AND', 'OR', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 
                   'GROUP BY', 'ORDER BY', 'HAVING', 'DISTINCT', 'UNION', 'LIMIT', 'OFFSET']
        for keyword in keywords:
            features.append(query.upper().count(keyword))
        
        # 嵌套查询特征
        features.append(query.count('('))  # 括号数量
        features.append(query.upper().count('EXISTS'))
        features.append(query.upper().count('NOT EXISTS'))
        
        return features
    
    def _extract_schema_features(self, query):
        """表和列特征"""
        features = []
        
        # 表特征
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        unique_tables = set(tables)
        features.append(len(unique_tables))
        
        # 列特征
        columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
        unique_columns = set(columns)
        features.append(len(unique_columns))
        
        # 表-列映射特征
        table_column_pairs = []
        for col in columns:
            if '.' in col:
                table, column = col.split('.', 1)
                table_column_pairs.append((table, column))
        
        features.append(len(set(table_column_pairs)))  # 唯一表列对数量
        
        # 计算选择性特征 (基于统计信息)
        selectivity_scores = []
        for col in unique_columns:
            if col in self.column_stats:
                stats = self.column_stats[col]
                card = stats.get('cardinality', 1)
                max_val = stats.get('max', 1)
                selectivity = min(1.0, card / max(1, max_val))
                selectivity_scores.append(selectivity)
        
        if selectivity_scores:
            features.extend([
                np.mean(selectivity_scores),
                np.min(selectivity_scores),
                np.max(selectivity_scores),
                np.std(selectivity_scores)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_enhanced_predicate_features(self, query):
        """增强谓词特征"""
        features = []
        
        # 操作符统计
        operator_counts = {}
        for op in self.operator_vocab:
            count = query.upper().count(op)
            operator_counts[op] = count
            features.append(count)
        
        # 谓词复杂度
        total_predicates = sum(operator_counts.values())
        features.append(total_predicates)
        
        # 数值特征 (增强版)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        if numbers:
            nums = [float(n) for n in numbers]
            features.extend([
                len(nums),
                max(nums),
                min(nums),
                np.mean(nums),
                np.std(nums) if len(nums) > 1 else 0,
                np.median(nums),
                len(set(nums))  # 唯一数值数量
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # 字符串常量
        string_literals = re.findall(r"'[^']*'", query)
        features.append(len(string_literals))
        if string_literals:
            avg_len = np.mean([len(s) for s in string_literals])
            features.append(avg_len)
        else:
            features.append(0)
        
        # LIKE模式复杂度
        like_patterns = re.findall(r"LIKE\s+'([^']*)'", query, re.IGNORECASE)
        like_complexity = sum(p.count('%') + p.count('_') for p in like_patterns)
        features.append(like_complexity)
        
        return features
    
    def _extract_enhanced_plan_features(self, explain_result):
        """增强查询计划特征"""
        features = []
        
        try:
            plan = json.loads(explain_result)
            root_plan = plan['QUERY PLAN'][0]['Plan']
            
            # 基本成本和行数信息
            total_cost = root_plan.get('Total Cost', 0)
            startup_cost = root_plan.get('Startup Cost', 0)
            plan_rows = root_plan.get('Plan Rows', 1)
            plan_width = root_plan.get('Plan Width', 0)
            
            features.extend([
                total_cost,
                startup_cost,
                plan_rows,
                plan_width,
                total_cost - startup_cost,  # 执行成本
                total_cost / max(1, plan_rows),  # 单行成本
                plan_rows * plan_width  # 估计数据量
            ])
            
            # 计划结构分析 (增强版)
            plan_stats = self._analyze_enhanced_plan_structure(root_plan)
            features.extend([
                plan_stats['depth'],
                plan_stats['node_count'],
                plan_stats['scan_count'],
                plan_stats['join_count'],
                plan_stats['total_cost'],
                plan_stats['total_rows'],
                plan_stats['avg_cost_per_node'],
                plan_stats['max_single_cost'],
                plan_stats['cost_variance']
            ])
            
            # 节点类型分布
            node_type_counts = Counter()
            self._count_node_types(root_plan, node_type_counts)
            
            # 主要节点类型特征
            scan_types = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']
            join_types = ['Nested Loop', 'Hash Join', 'Merge Join']
            
            features.append(sum(node_type_counts.get(t, 0) for t in scan_types))
            features.append(sum(node_type_counts.get(t, 0) for t in join_types))
            
            # 计算join selectivity
            if plan_stats['join_count'] > 0:
                join_selectivity = plan_stats['total_rows'] / max(1, plan_stats['scan_count'])
                features.append(join_selectivity)
            else:
                features.append(1.0)
            
        except Exception as e:
            # 如果解析失败，填充默认值
            features.extend([0] * 20)
        
        return features
    
    def _analyze_enhanced_plan_structure(self, plan, depth=0):
        """增强计划结构分析"""
        costs = []
        
        def collect_costs(p, d=0):
            cost = p.get('Total Cost', 0)
            costs.append(cost)
            for subplan in p.get('Plans', []):
                collect_costs(subplan, d + 1)
        
        collect_costs(plan)
        
        stats = {
            'depth': depth,
            'node_count': 1,
            'scan_count': 1 if 'Scan' in plan.get('Node Type', '') else 0,
            'join_count': 1 if 'Join' in plan.get('Node Type', '') else 0,
            'total_cost': plan.get('Total Cost', 0),
            'total_rows': plan.get('Plan Rows', 0),
            'avg_cost_per_node': np.mean(costs) if costs else 0,
            'max_single_cost': max(costs) if costs else 0,
            'cost_variance': np.var(costs) if len(costs) > 1 else 0
        }
        
        for subplan in plan.get('Plans', []):
            substats = self._analyze_enhanced_plan_structure(subplan, depth + 1)
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
    
    def _extract_enhanced_statistical_features(self, query):
        """增强统计特征"""
        features = []
        
        # 从列统计信息中提取特征
        columns = re.findall(r'([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query)
        if columns and self.column_stats:
            cardinalities = []
            max_values = []
            min_values = []
            unique_counts = []
            
            for col in columns:
                if col in self.column_stats:
                    stats = self.column_stats[col]
                    cardinalities.append(stats.get('cardinality', 0))
                    max_values.append(stats.get('max', 0))
                    min_values.append(stats.get('min', 0))
                    unique_counts.append(stats.get('unique', 0))
            
            if cardinalities:
                features.extend([
                    np.mean(cardinalities),
                    np.max(cardinalities),
                    np.min(cardinalities),
                    np.std(cardinalities),
                    np.mean(max_values),
                    np.mean(min_values),
                    np.mean(unique_counts),
                    len(cardinalities)  # 涉及的有统计信息的列数
                ])
            else:
                features.extend([0] * 8)
        else:
            features.extend([0] * 8)
        
        return features
    
    def _extract_complexity_features(self, query, explain_result):
        """提取查询复杂度特征"""
        features = []
        
        # 查询语法复杂度
        nesting_level = query.count('(') - query.count(')')
        features.append(abs(nesting_level))
        
        # 子查询数量
        subquery_count = query.upper().count('SELECT') - 1  # 减去主查询
        features.append(max(0, subquery_count))
        
        # 函数调用数量
        function_patterns = [r'\w+\s*\(', r'COUNT\s*\(', r'SUM\s*\(', r'AVG\s*\(', r'MAX\s*\(', r'MIN\s*\(']
        function_count = sum(len(re.findall(pattern, query, re.IGNORECASE)) for pattern in function_patterns)
        features.append(function_count)
        
        # 查询计划复杂度评分
        try:
            plan = json.loads(explain_result)
            root_plan = plan['QUERY PLAN'][0]['Plan']
            
            # 计算复杂度评分
            complexity_score = 0
            
            def calc_complexity(p, depth=0):
                nonlocal complexity_score
                node_type = p.get('Node Type', '')
                
                # 不同节点类型的权重
                weights = {
                    'Seq Scan': 1,
                    'Index Scan': 2,
                    'Nested Loop': 5,
                    'Hash Join': 3,
                    'Merge Join': 4,
                    'Sort': 2,
                    'Aggregate': 1
                }
                
                complexity_score += weights.get(node_type, 1) * (depth + 1)
                
                for subplan in p.get('Plans', []):
                    calc_complexity(subplan, depth + 1)
            
            calc_complexity(root_plan)
            features.append(complexity_score)
            
        except:
            features.append(0)
        
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

class EnsembleModel:
    """集成模型"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()  # 使用RobustScaler处理异常值
        
    def fit(self, X, y):
        """训练集成模型"""
        print("训练集成模型...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 定义模型
        models_config = {
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # 训练每个模型
        val_scores = {}
        for name, model in models_config.items():
            print(f"训练 {name}...")
            model.fit(X_train, y_train)
            
            # 验证
            val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, val_pred)
            val_scores[name] = val_score
            
            self.models[name] = model
            print(f"{name} 验证MSE: {val_score:.4f}")
        
        # 计算权重 (基于验证性能)
        total_inv_score = sum(1.0 / score for score in val_scores.values())
        for name, score in val_scores.items():
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
    print("=== 改进的SQL基数估计模型训练 ===")
    
    # 加载训练数据
    print("加载训练数据...")
    train_data = load_data('data/train_data.json', max_samples=40000)  # 使用更多数据
    print(f"训练数据数量: {len(train_data)}")
    
    # 初始化增强特征提取器
    extractor = EnhancedFeatureExtractor()
    extractor.fit_vocabularies(train_data)
    
    # 特征提取
    print("提取增强特征...")
    X_train = []
    y_train = []
    
    for sample in tqdm(train_data, desc="提取训练特征"):
        features = extractor.extract_features(sample['query'], sample['explain_result'])
        cardinality = extract_cardinality_from_plan(sample['explain_result'])
        
        X_train.append(features)
        y_train.append(np.log(max(1, cardinality)))  # log scale
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"增强特征维度: {X_train.shape}")
    print(f"目标值范围: {y_train.min():.2f} - {y_train.max():.2f}")
    
    # 检查数据质量
    print(f"特征缺失值: {np.isnan(X_train).sum()}")
    print(f"特征无穷值: {np.isinf(X_train).sum()}")
    
    # 处理异常值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 训练集成模型
    ensemble = EnsembleModel()
    ensemble.fit(X_train, y_train)
    
    # 训练集评估
    train_pred = ensemble.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    print(f"训练集成MSE: {train_mse:.4f}")
    
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
    
    # 预测
    print("预测测试数据...")
    test_pred_log = ensemble.predict(X_test)
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
    
    print("\n=== 改进训练完成 ===")

if __name__ == '__main__':
    main() 