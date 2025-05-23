import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML

class DataProcessor:
    def __init__(self):
        self.column_stats = None
        self.table2vec = {}
        self.column2vec = {}
        self.op2vec = {}
        self.join2vec = {}
        self.node_type2vec = {}
        
        # 加载列统计信息
        self.load_column_stats()
        
    def load_column_stats(self):
        """加载列统计信息"""
        self.column_stats = pd.read_csv('data/column_min_max_vals.csv')
        self.column_stats.set_index('name', inplace=True)
        
    def extract_cardinality_from_plan(self, explain_result_str: str) -> int:
        """从查询计划中提取cardinality"""
        try:
            explain_result = json.loads(explain_result_str)
            if 'QUERY PLAN' in explain_result:
                plan = explain_result['QUERY PLAN'][0]['Plan']
                return plan.get('Plan Rows', 1)
        except:
            pass
        return 1
    
    def parse_sql_query(self, query: str) -> Dict[str, Any]:
        """解析SQL查询，提取表、列、操作符等信息"""
        features = {
            'tables': [],
            'columns': [],
            'predicates': [],
            'joins': [],
            'operators': [],
            'values': []
        }
        
        # 提取表名
        tables = self._extract_tables(query)
        features['tables'] = tables
        
        # 直接从查询字符串中提取WHERE子句
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|;|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            predicates, joins = self._parse_where_clause(where_clause)
            features['predicates'] = predicates
            features['joins'] = joins
        
        return features
    
    def _extract_tables(self, query: str) -> List[str]:
        """提取查询中的表名"""
        tables = []
        # 使用正则表达式提取FROM子句中的表
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|$)', query, re.IGNORECASE)
        if from_match:
            from_clause = from_match.group(1)
            # 分割表名和别名
            table_parts = re.split(r',\s*', from_clause)
            for part in table_parts:
                # 提取表名（忽略别名）
                table_match = re.match(r'(\w+)(?:\s+(\w+))?', part.strip())
                if table_match:
                    table_name = table_match.group(1)
                    alias = table_match.group(2) if table_match.group(2) else table_name
                    tables.append((table_name, alias))
        return tables
    
    def _extract_where_clause(self, parsed):
        """提取WHERE子句"""
        # 简化方法：直接从原始查询中提取WHERE子句
        return None
    
    def _parse_where_clause(self, where_str: str) -> Tuple[List, List]:
        """解析WHERE子句，提取谓词和连接条件"""
        predicates = []
        joins = []
        
        # 简单的条件解析
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_str, flags=re.IGNORECASE)
        
        for condition in conditions:
            condition = condition.strip()
            
            # 检查是否是连接条件（包含等号且两边都有表前缀）
            if '=' in condition and '.' in condition:
                parts = condition.split('=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    if '.' in left and '.' in right:
                        joins.append((left, right))
                        continue
            
            # 其他条件作为谓词
            predicates.append(condition)
        
        return predicates, joins
    
    def extract_plan_features(self, explain_result_str: str) -> Dict[str, Any]:
        """从查询计划中提取特征"""
        features = {
            'node_types': [],
            'costs': [],
            'rows': [],
            'operators': [],
            'scan_types': []
        }
        
        try:
            explain_result = json.loads(explain_result_str)
            if 'QUERY PLAN' in explain_result:
                plan = explain_result['QUERY PLAN'][0]['Plan']
                self._extract_plan_recursive(plan, features)
        except:
            pass
        
        return features
    
    def _extract_plan_recursive(self, plan: Dict, features: Dict):
        """递归提取查询计划特征"""
        # 提取节点类型
        node_type = plan.get('Node Type', '')
        features['node_types'].append(node_type)
        
        # 提取成本信息
        features['costs'].append(plan.get('Total Cost', 0))
        features['rows'].append(plan.get('Plan Rows', 0))
        
        # 提取扫描类型
        if 'Scan Direction' in plan:
            features['scan_types'].append(plan['Scan Direction'])
        
        # 递归处理子计划
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self._extract_plan_recursive(subplan, features)
    
    def build_vocabularies(self, data: List[Dict]) -> None:
        """构建词汇表"""
        tables = set()
        columns = set()
        operators = set()
        joins = set()
        node_types = set()
        
        for item in data:
            query_features = self.parse_sql_query(item['query'])
            plan_features = self.extract_plan_features(item['explain_result'])
            
            # 收集表名
            for table, alias in query_features['tables']:
                tables.add(table)
                tables.add(alias)
            
            # 收集列名（从谓词中提取）
            for pred in query_features['predicates']:
                # 提取列名
                col_matches = re.findall(r'(\w+\.\w+)', pred)
                for col in col_matches:
                    columns.add(col)
            
            # 收集操作符
            for pred in query_features['predicates']:
                if '>' in pred:
                    operators.add('>')
                elif '<' in pred:
                    operators.add('<')
                elif '=' in pred:
                    operators.add('=')
                elif 'LIKE' in pred.upper():
                    operators.add('LIKE')
            
            # 收集连接信息
            for left, right in query_features['joins']:
                joins.add(f"{left}={right}")
            
            # 收集节点类型
            for node_type in plan_features['node_types']:
                node_types.add(node_type)
        
        # 构建向量映射
        self.table2vec = {table: i for i, table in enumerate(sorted(tables))}
        self.column2vec = {col: i for i, col in enumerate(sorted(columns))}
        self.op2vec = {op: i for i, op in enumerate(sorted(operators))}
        self.join2vec = {join: i for i, join in enumerate(sorted(joins))}
        self.node_type2vec = {nt: i for i, nt in enumerate(sorted(node_types))}
        
        print(f"词汇表构建完成:")
        print(f"  表: {len(self.table2vec)}")
        print(f"  列: {len(self.column2vec)}")
        print(f"  操作符: {len(self.op2vec)}")
        print(f"  连接: {len(self.join2vec)}")
        print(f"  节点类型: {len(self.node_type2vec)}")
    
    def encode_sample(self, query: str, explain_result: str) -> Dict[str, np.ndarray]:
        """编码单个样本"""
        query_features = self.parse_sql_query(query)
        plan_features = self.extract_plan_features(explain_result)
        
        # 编码表特征
        table_features = np.zeros(len(self.table2vec))
        for table, alias in query_features['tables']:
            if table in self.table2vec:
                table_features[self.table2vec[table]] = 1
            if alias in self.table2vec:
                table_features[self.table2vec[alias]] = 1
        
        # 编码谓词特征
        predicate_features = np.zeros(len(self.column2vec) + len(self.op2vec) + 1)  # +1 for value
        for pred in query_features['predicates']:
            # 编码列
            col_matches = re.findall(r'(\w+\.\w+)', pred)
            for col in col_matches:
                if col in self.column2vec:
                    predicate_features[self.column2vec[col]] = 1
            
            # 编码操作符
            if '>' in pred and '>' in self.op2vec:
                predicate_features[len(self.column2vec) + self.op2vec['>']] = 1
            elif '<' in pred and '<' in self.op2vec:
                predicate_features[len(self.column2vec) + self.op2vec['<']] = 1
            elif '=' in pred and '=' in self.op2vec:
                predicate_features[len(self.column2vec) + self.op2vec['=']] = 1
        
        # 编码连接特征
        join_features = np.zeros(len(self.join2vec))
        for left, right in query_features['joins']:
            join_key = f"{left}={right}"
            if join_key in self.join2vec:
                join_features[self.join2vec[join_key]] = 1
        
        # 编码计划特征
        plan_feature_vec = np.zeros(len(self.node_type2vec) + 3)  # +3 for cost, rows, depth
        for node_type in plan_features['node_types']:
            if node_type in self.node_type2vec:
                plan_feature_vec[self.node_type2vec[node_type]] = 1
        
        # 添加成本和行数统计
        if plan_features['costs']:
            plan_feature_vec[-3] = np.mean(plan_features['costs'])
            plan_feature_vec[-2] = np.mean(plan_features['rows'])
            plan_feature_vec[-1] = len(plan_features['node_types'])  # 计划深度
        
        return {
            'table_features': table_features,
            'predicate_features': predicate_features,
            'join_features': join_features,
            'plan_features': plan_feature_vec
        }
    
    def load_data(self, file_path: str, max_samples: int = None) -> Tuple[List[Dict], List[int]]:
        """加载数据"""
        data = []
        cardinalities = []
        
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
        # 尝试解析为JSON数组
        try:
            if content.startswith('['):
                # JSON数组格式
                all_data = json.loads(content)
                if max_samples:
                    all_data = all_data[:max_samples]
                
                for sample in all_data:
                    data.append(sample)
                    
                    # 提取cardinality（仅训练数据有）
                    if 'train' in file_path:
                        cardinality = self.extract_cardinality_from_plan(sample['explain_result'])
                        cardinalities.append(cardinality)
            else:
                # JSON Lines格式
                lines = content.split('\n')
                if max_samples:
                    lines = lines[:max_samples]
                    
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            data.append(sample)
                            
                            # 提取cardinality（仅训练数据有）
                            if 'train' in file_path:
                                cardinality = self.extract_cardinality_from_plan(sample['explain_result'])
                                cardinalities.append(cardinality)
                        except json.JSONDecodeError:
                            continue
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return [], []
        
        print(f"成功加载 {len(data)} 个样本")
        return data, cardinalities if cardinalities else None 