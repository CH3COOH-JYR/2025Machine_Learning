import json
import pandas as pd
import numpy as np
from collections import Counter
import re

def explore_data():
    print("=== 数据探索开始 ===")
    
    # 读取列统计信息
    print("\n1. 列统计信息:")
    column_stats = pd.read_csv('data/column_min_max_vals.csv')
    print(column_stats.head(10))
    
    # 探索训练数据
    print("\n2. 训练数据结构探索:")
    try:
        with open('data/train_data.json', 'r') as f:
            # 尝试读取为一个完整的JSON数组
            content = f.read().strip()
            if content.startswith('['):
                # 标准JSON数组格式
                train_data = json.loads(content)
                train_samples = train_data[:10]  # 取前10个样本
            else:
                # JSON Lines格式
                train_samples = []
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            train_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
        
        if train_samples:
            print(f"训练数据样本数（前10个）: {len(train_samples)}")
            sample = train_samples[0]
            print("样本结构:")
            print(f"- 键: {list(sample.keys())}")
            print(f"- query_id: {sample.get('query_id')}")
            print(f"- query: {sample.get('query', '')[:100]}...")
            
            # 检查是否有cardinality标签
            if 'cardinality' in sample:
                print(f"- cardinality: {sample.get('cardinality')}")
            
            # 解析explain_result
            explain_result_str = sample.get('explain_result', '{}')
            if isinstance(explain_result_str, str):
                try:
                    explain_result = json.loads(explain_result_str)
                    print("- explain_result结构:")
                    print(f"  - QUERY PLAN存在: {'QUERY PLAN' in explain_result}")
                    if 'QUERY PLAN' in explain_result:
                        plan = explain_result['QUERY PLAN'][0]['Plan']
                        print(f"  - Plan Rows: {plan.get('Plan Rows')}")
                        print(f"  - Node Type: {plan.get('Node Type')}")
                        print(f"  - Total Cost: {plan.get('Total Cost')}")
                except json.JSONDecodeError:
                    print("  - explain_result解析失败")
        else:
            print("未找到训练数据样本")
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
    
    # 探索测试数据
    print("\n3. 测试数据结构探索:")
    try:
        with open('data/test_data.json', 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                test_data = json.loads(content)
                test_samples = test_data[:10]
            else:
                test_samples = []
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            test_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
        
        if test_samples:
            print(f"测试数据样本数（前10个）: {len(test_samples)}")
            sample = test_samples[0]
            print("测试样本结构:")
            print(f"- 键: {list(sample.keys())}")
            # 检查是否有cardinality标签
            if 'cardinality' in sample:
                print(f"- cardinality: {sample.get('cardinality')}")
            else:
                print("- 测试数据没有cardinality标签（这是正常的）")
        else:
            print("未找到测试数据样本")
    except Exception as e:
        print(f"读取测试数据时出错: {e}")
    
    # 统计查询模式
    print("\n4. 查询模式分析:")
    all_samples = []
    if 'train_samples' in locals():
        all_samples.extend(train_samples)
    if 'test_samples' in locals():
        all_samples.extend(test_samples)
    
    if all_samples:
        table_counts = []
        for sample in all_samples:
            query = sample.get('query', '')
            # 提取FROM子句中的表
            from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|$)', query, re.IGNORECASE)
            if from_match:
                from_clause = from_match.group(1)
                # 统计逗号分隔的表数量
                table_count = len([t.strip() for t in from_clause.split(',') if t.strip()])
                table_counts.append(table_count)
        
        if table_counts:
            print(f"表数量分布: {Counter(table_counts)}")
    
    print("\n=== 数据探索完成 ===")

if __name__ == "__main__":
    explore_data() 