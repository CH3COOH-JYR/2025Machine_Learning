import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset

class AdvancedDataProcessor:
    """高级数据处理器，专门为QueryFormer模型设计"""
    
    def __init__(self):
        self.column_stats = None
        self.node_type_vocab = {}
        self.table_vocab = {}
        self.column_vocab = {}
        
        # 加载列统计信息
        self.load_column_stats()
        
    def load_column_stats(self):
        """加载列统计信息"""
        try:
            self.column_stats = pd.read_csv('data/column_min_max_vals.csv')
            self.column_stats.set_index('name', inplace=True)
            print(f"加载列统计信息: {len(self.column_stats)} 列")
        except Exception as e:
            print(f"加载列统计信息失败: {e}")
            self.column_stats = pd.DataFrame()
    
    def extract_cardinality_from_plan(self, explain_result_str: str) -> int:
        """从查询计划中提取真实cardinality"""
        try:
            explain_result = json.loads(explain_result_str)
            if 'QUERY PLAN' in explain_result:
                plan = explain_result['QUERY PLAN'][0]['Plan']
                # 使用实际执行行数
                actual_rows = plan.get('Actual Rows', plan.get('Plan Rows', 1))
                return max(1, actual_rows)
        except Exception as e:
            print(f"解析查询计划失败: {e}")
        return 1
    
    def extract_plan_features(self, explain_result_str: str) -> Dict[str, Any]:
        """从查询计划中提取详细特征"""
        features = {
            'node_types': [],
            'costs': [],
            'rows': [],
            'selectivities': [],
            'scan_directions': [],
            'join_types': [],
            'index_names': [],
            'relation_names': []
        }
        
        try:
            explain_result = json.loads(explain_result_str)
            if 'QUERY PLAN' in explain_result:
                plan = explain_result['QUERY PLAN'][0]['Plan']
                self._extract_plan_recursive(plan, features)
        except Exception as e:
            print(f"提取计划特征失败: {e}")
        
        return features
    
    def _extract_plan_recursive(self, plan: Dict, features: Dict):
        """递归提取查询计划特征"""
        # 节点类型
        node_type = plan.get('Node Type', 'Unknown')
        features['node_types'].append(node_type)
        
        # 成本信息
        startup_cost = plan.get('Startup Cost', 0)
        total_cost = plan.get('Total Cost', 0)
        features['costs'].append(total_cost)
        
        # 行数信息
        plan_rows = plan.get('Plan Rows', 0)
        actual_rows = plan.get('Actual Rows', plan_rows)
        features['rows'].append(actual_rows)
        
        # 选择性（实际行数/计划行数）
        if plan_rows > 0:
            selectivity = actual_rows / plan_rows
        else:
            selectivity = 1.0
        features['selectivities'].append(selectivity)
        
        # 扫描方向
        scan_direction = plan.get('Scan Direction', 'Unknown')
        features['scan_directions'].append(scan_direction)
        
        # 连接类型
        join_type = plan.get('Join Type', 'Unknown')
        features['join_types'].append(join_type)
        
        # 索引名
        index_name = plan.get('Index Name', 'Unknown')
        features['index_names'].append(index_name)
        
        # 关系名
        relation_name = plan.get('Relation Name', 'Unknown')
        features['relation_names'].append(relation_name)
        
        # 递归处理子计划
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self._extract_plan_recursive(subplan, features)
    
    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """提取查询的结构化特征"""
        features = {
            'tables': [],
            'columns': [],
            'predicates': [],
            'joins': [],
            'aggregations': [],
            'orderings': [],
            'groupings': [],
            'subqueries': 0
        }
        
        # 转换为大写便于处理
        query_upper = query.upper()
        
        # 提取表名
        features['tables'] = self._extract_tables(query)
        
        # 提取列名
        features['columns'] = self._extract_columns(query)
        
        # 提取谓词
        features['predicates'] = self._extract_predicates(query)
        
        # 提取连接
        features['joins'] = self._extract_joins(query)
        
        # 提取聚合函数
        features['aggregations'] = self._extract_aggregations(query_upper)
        
        # 提取排序
        features['orderings'] = self._extract_orderings(query_upper)
        
        # 提取分组
        features['groupings'] = self._extract_groupings(query_upper)
        
        # 计算子查询数量
        features['subqueries'] = query_upper.count('SELECT') - 1
        
        return features
    
    def _extract_tables(self, query: str) -> List[str]:
        """提取表名"""
        tables = []
        
        # 匹配FROM子句
        from_pattern = r'FROM\s+([^WHERE^GROUP^ORDER^HAVING^LIMIT^;]+)'
        from_match = re.search(from_pattern, query, re.IGNORECASE)
        
        if from_match:
            from_clause = from_match.group(1).strip()
            
            # 处理JOIN
            join_pattern = r'(\w+)\s+(?:AS\s+)?(\w+)?'
            matches = re.findall(join_pattern, from_clause)
            
            for match in matches:
                table_name = match[0]
                alias = match[1] if match[1] else table_name
                tables.append((table_name, alias))
        
        return tables
    
    def _extract_columns(self, query: str) -> List[str]:
        """提取列名"""
        columns = []
        
        # 从SELECT子句提取
        select_pattern = r'SELECT\s+(.+?)\s+FROM'
        select_match = re.search(select_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if select_match:
            select_clause = select_match.group(1)
            # 简单的列名提取
            col_pattern = r'(\w+\.\w+|\w+)'
            col_matches = re.findall(col_pattern, select_clause)
            columns.extend(col_matches)
        
        # 从WHERE子句提取
        where_pattern = r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+HAVING|;|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            col_pattern = r'(\w+\.\w+)'
            col_matches = re.findall(col_pattern, where_clause)
            columns.extend(col_matches)
        
        return list(set(columns))
    
    def _extract_predicates(self, query: str) -> List[Dict[str, Any]]:
        """提取谓词信息"""
        predicates = []
        
        where_pattern = r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+HAVING|;|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # 分割条件
            conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
            
            for condition in conditions:
                condition = condition.strip()
                predicate = self._parse_predicate(condition)
                if predicate:
                    predicates.append(predicate)
        
        return predicates
    
    def _parse_predicate(self, condition: str) -> Optional[Dict[str, Any]]:
        """解析单个谓词"""
        # 操作符模式
        operators = ['>=', '<=', '!=', '<>', '>', '<', '=', 'LIKE', 'IN', 'BETWEEN']
        
        for op in operators:
            if op in condition.upper():
                parts = re.split(f'\\s*{re.escape(op)}\\s*', condition, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    return {
                        'column': left,
                        'operator': op,
                        'value': right,
                        'value_type': self._infer_value_type(right)
                    }
        
        return None
    
    def _infer_value_type(self, value: str) -> str:
        """推断值类型"""
        value = value.strip()
        
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return 'integer'
        elif re.match(r'^-?\d+\.\d+$', value):
            return 'float'
        elif value.startswith("'") and value.endswith("'"):
            return 'string'
        elif value.upper() in ['TRUE', 'FALSE']:
            return 'boolean'
        elif value.upper() == 'NULL':
            return 'null'
        else:
            return 'unknown'
    
    def _extract_joins(self, query: str) -> List[Dict[str, str]]:
        """提取连接信息"""
        joins = []
        
        # JOIN模式
        join_pattern = r'(\w+\s+)?JOIN\s+(\w+)(?:\s+AS\s+(\w+))?\s+ON\s+(.+?)(?:\s+(?:INNER|LEFT|RIGHT|FULL|WHERE|GROUP|ORDER|HAVING)|;|$)'
        join_matches = re.findall(join_pattern, query, re.IGNORECASE)
        
        for match in join_matches:
            join_type = match[0].strip() if match[0] else 'INNER'
            table = match[1]
            alias = match[2] if match[2] else table
            condition = match[3]
            
            joins.append({
                'type': join_type,
                'table': table,
                'alias': alias,
                'condition': condition
            })
        
        return joins
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """提取聚合函数"""
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT']
        found_aggs = []
        
        for agg in agg_functions:
            if agg in query:
                found_aggs.append(agg)
        
        return found_aggs
    
    def _extract_orderings(self, query: str) -> List[str]:
        """提取排序信息"""
        orderings = []
        
        order_pattern = r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|;|$)'
        order_match = re.search(order_pattern, query, re.IGNORECASE)
        
        if order_match:
            order_clause = order_match.group(1)
            orderings = [col.strip() for col in order_clause.split(',')]
        
        return orderings
    
    def _extract_groupings(self, query: str) -> List[str]:
        """提取分组信息"""
        groupings = []
        
        group_pattern = r'GROUP\s+BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|;|$)'
        group_match = re.search(group_pattern, query, re.IGNORECASE)
        
        if group_match:
            group_clause = group_match.group(1)
            groupings = [col.strip() for col in group_clause.split(',')]
        
        return groupings
    
    def build_vocabularies(self, data: List[Dict]) -> None:
        """构建词汇表"""
        node_types = set()
        tables = set()
        columns = set()
        
        print("构建词汇表...")
        
        for i, item in enumerate(data):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(data)}")
            
            # 提取查询特征
            query_features = self.extract_query_features(item['query'])
            
            # 提取计划特征
            plan_features = self.extract_plan_features(item['explain_result'])
            
            # 收集节点类型
            node_types.update(plan_features['node_types'])
            
            # 收集表名
            for table, alias in query_features['tables']:
                tables.add(table)
                if alias != table:
                    tables.add(alias)
            
            # 收集列名
            columns.update(query_features['columns'])
        
        # 构建词汇表映射
        self.node_type_vocab = {nt: i for i, nt in enumerate(sorted(node_types))}
        self.table_vocab = {t: i for i, t in enumerate(sorted(tables))}
        self.column_vocab = {c: i for i, c in enumerate(sorted(columns))}
        
        print(f"词汇表构建完成:")
        print(f"  节点类型: {len(self.node_type_vocab)}")
        print(f"  表: {len(self.table_vocab)}")
        print(f"  列: {len(self.column_vocab)}")
    
    def encode_plan_features(self, plan_features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """编码查询计划特征"""
        # 节点类型编码
        node_type_ids = []
        for nt in plan_features['node_types']:
            if nt in self.node_type_vocab:
                node_type_ids.append(self.node_type_vocab[nt])
            else:
                node_type_ids.append(0)  # Unknown
        
        # 填充到固定长度
        max_nodes = 20
        if len(node_type_ids) > max_nodes:
            node_type_ids = node_type_ids[:max_nodes]
        else:
            node_type_ids.extend([0] * (max_nodes - len(node_type_ids)))
        
        # 数值特征
        costs = plan_features['costs']
        rows = plan_features['rows']
        selectivities = plan_features['selectivities']
        
        # 统计特征
        avg_cost = np.mean(costs) if costs else 0.0
        avg_rows = np.mean(rows) if rows else 0.0
        avg_selectivity = np.mean(selectivities) if selectivities else 1.0
        
        # 对数变换以稳定数值
        avg_cost = np.log1p(avg_cost)
        avg_rows = np.log1p(avg_rows)
        
        return {
            'node_types': np.array(node_type_ids, dtype=np.int64),
            'costs': np.array(avg_cost, dtype=np.float32),  # 标量值
            'rows': np.array(avg_rows, dtype=np.float32),   # 标量值
            'selectivities': np.array(avg_selectivity, dtype=np.float32)  # 标量值
        }
    
    def load_data(self, file_path: str, max_samples: Optional[int] = None) -> Tuple[List[Dict], Optional[List[int]]]:
        """加载数据"""
        print(f"加载数据: {file_path}")
        
        data = []
        cardinalities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析JSON
            if content.startswith('['):
                all_data = json.loads(content)
            else:
                # JSON Lines格式
                lines = content.split('\n')
                all_data = []
                for line in lines:
                    if line.strip():
                        all_data.append(json.loads(line))
            
            # 限制样本数量
            if max_samples:
                all_data = all_data[:max_samples]
            
            for sample in all_data:
                data.append(sample)
                
                # 提取cardinality（仅训练数据）
                if 'train' in file_path:
                    cardinality = self.extract_cardinality_from_plan(sample['explain_result'])
                    cardinalities.append(cardinality)
            
            print(f"成功加载 {len(data)} 个样本")
            return data, cardinalities if cardinalities else None
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return [], None

class QueryFormerDataset(Dataset):
    """QueryFormer数据集"""
    
    def __init__(self, 
                 data: List[Dict], 
                 cardinalities: Optional[List[int]], 
                 processor: AdvancedDataProcessor,
                 tokenizer,
                 max_seq_length: int = 512):
        self.data = data
        self.cardinalities = cardinalities
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 预处理所有样本
        print("预处理数据集...")
        self.processed_samples = []
        
        for i, sample in enumerate(data):
            if i % 1000 == 0:
                print(f"预处理进度: {i}/{len(data)}")
            
            # 编码查询
            token_ids = self.tokenizer.encode(sample['query'], self.max_seq_length)
            
            # 提取并编码计划特征
            plan_features = self.processor.extract_plan_features(sample['explain_result'])
            encoded_plan = self.processor.encode_plan_features(plan_features)
            
            self.processed_samples.append({
                'token_ids': token_ids,
                'plan_features': encoded_plan
            })
        
        print("数据集预处理完成")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.processed_samples[idx]
        
        # 转换为张量
        token_ids = torch.LongTensor(sample['token_ids'])
        
        # 处理计划特征 - 确保正确处理标量值
        costs_val = sample['plan_features']['costs']
        rows_val = sample['plan_features']['rows']
        selectivities_val = sample['plan_features']['selectivities']
        
        # 如果是numpy标量，转换为Python标量
        if hasattr(costs_val, 'item'):
            costs_val = costs_val.item()
        if hasattr(rows_val, 'item'):
            rows_val = rows_val.item()
        if hasattr(selectivities_val, 'item'):
            selectivities_val = selectivities_val.item()
        
        plan_features = {
            'node_types': torch.LongTensor(sample['plan_features']['node_types']),
            'costs': torch.FloatTensor([costs_val]),  # 包装为1维张量
            'rows': torch.FloatTensor([rows_val]),    # 包装为1维张量
            'selectivities': torch.FloatTensor([selectivities_val])  # 包装为1维张量
        }
        
        # 目标值
        if self.cardinalities is not None:
            cardinality = max(1, self.cardinalities[idx])
            log_cardinality = np.log(cardinality)
            log_cardinality = np.clip(log_cardinality, 0, 20)  # 限制范围
            target = torch.FloatTensor([log_cardinality])
        else:
            target = torch.FloatTensor([0])
        
        return {
            'token_ids': token_ids,
            'plan_features': plan_features,
            'target': target
        } 