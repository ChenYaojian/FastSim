#!/usr/bin/env python3
"""
测试StateVector的末态采样功能
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import json
import numpy as np
from fastsim.state import StateVector, AbstractState
from fastsim.circuit import Circuit, load_gates_from_config

def test_basic_sampling():
    """测试基本采样功能"""
    print("=== 测试基本采样功能 ===\n")
    
    # 创建简单的量子态
    state = StateVector(2)
    
    # 创建Bell态 |00⟩ + |11⟩
    state.state_vector = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex64) / np.sqrt(2)
    
    print(f"量子态: {state.state_vector}")
    print(f"概率分布: {torch.abs(state.state_vector) ** 2}")
    
    # 进行采样
    num_shots = 10000
    print(f"\n进行 {num_shots} 次采样...")
    
    # 采样所有量子比特
    result_json = state.sample_final_state(num_shots=num_shots, return_json=True)
    result_dict = state.sample_final_state(num_shots=num_shots, return_json=False)
    
    print("JSON格式结果:")
    print(result_json)
    
    print("\n字典格式结果:")
    print(result_dict)
    
    # 验证结果
    print("\n验证结果:")
    expected_states = {0, 3}  # |00⟩ 和 |11⟩
    sampled_states = set(result_dict.keys())
    print(f"期望状态: {expected_states}")
    print(f"采样状态: {sampled_states}")
    print(f"状态匹配: {sampled_states == expected_states}")
    
    # 检查概率分布
    total_shots = sum(result_dict.values())
    for state_idx, count in result_dict.items():
        prob = count / total_shots
        print(f"状态 {state_idx} (|{format(state_idx, '02b')}⟩): {count} 次, 概率 {prob:.4f}")

def test_partial_sampling():
    """测试部分量子比特采样"""
    print("\n" + "="*50)
    print("=== 测试部分量子比特采样 ===\n")
    
    # 创建3比特量子态
    state = StateVector(3)
    
    # 创建GHZ态 |000⟩ + |111⟩
    state.state_vector = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                     dtype=torch.complex64) / np.sqrt(2)
    
    print(f"3比特GHZ态: {state.state_vector}")
    
    # 采样前两个量子比特
    num_shots = 5000
    print(f"\n采样前两个量子比特，{num_shots} 次...")
    
    result_dict = state.sample_final_state(num_shots=num_shots, 
                                          qubit_indices=[0, 1], 
                                          return_json=False)
    
    print("采样结果:")
    for state_idx, count in result_dict.items():
        bitstring = format(state_idx, '02b')
        prob = count / num_shots
        print(f"状态 {state_idx} (|{bitstring}⟩): {count} 次, 概率 {prob:.4f}")
    
    # 验证：GHZ态测量前两个比特应该得到|00⟩和|11⟩，各50%概率
    expected_states = {0, 3}  # |00⟩ 和 |11⟩
    sampled_states = set(result_dict.keys())
    print(f"\n期望状态: {expected_states}")
    print(f"采样状态: {sampled_states}")
    print(f"状态匹配: {sampled_states == expected_states}")

def test_circuit_sampling():
    """测试电路执行后的采样"""
    print("\n" + "="*50)
    print("=== 测试电路执行后的采样 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建电路
    circuit = Circuit(2)
    circuit.add_gate("H", [0])  # Hadamard门
    circuit.add_gate("CNOT", [0, 1])  # CNOT门
    
    print("电路:")
    print(circuit.draw())
    
    # 执行电路
    initial_state = torch.zeros(4, dtype=torch.complex64)
    initial_state[0] = 1.0  # |00⟩态
    initial_state = initial_state.unsqueeze(0)  # 添加batch维度
    
    final_state = circuit(initial_state)
    print(f"\n最终态向量: {final_state[0]}")
    
    # 创建StateVector并设置最终态
    state = StateVector(2)
    state.state_vector = final_state[0]
    
    print(f"最终态概率分布: {torch.abs(state.state_vector) ** 2}")
    
    # 进行采样
    num_shots = 8000
    print(f"\n进行 {num_shots} 次采样...")
    
    result_dict = state.sample_final_state(num_shots=num_shots, return_json=False)
    
    print("采样结果:")
    for state_idx, count in result_dict.items():
        bitstring = format(state_idx, '02b')
        prob = count / num_shots
        print(f"状态 {state_idx} (|{bitstring}⟩): {count} 次, 概率 {prob:.4f}")
    
    # 理论值：Bell态 |00⟩ + |11⟩，各50%概率
    print(f"\n理论概率: |00⟩: 0.5, |11⟩: 0.5")

def test_probability_distribution():
    """测试概率分布计算"""
    print("\n" + "="*50)
    print("=== 测试概率分布计算 ===\n")
    
    # 创建量子态
    state = StateVector(2)
    state.state_vector = torch.tensor([0.6, 0.0, 0.0, 0.8], dtype=torch.complex64)
    state.state_vector = state.state_vector / torch.norm(state.state_vector)  # 归一化
    
    print(f"量子态: {state.state_vector}")
    
    # 计算完整概率分布
    prob_dist = state.get_probability_distribution()
    print(f"\n完整概率分布: {prob_dist}")
    
    # 计算部分量子比特的概率分布
    prob_dist_partial = state.get_probability_distribution(qubit_indices=[0])
    print(f"第一个量子比特的概率分布: {prob_dist_partial}")
    
    # 验证概率和为1
    total_prob = sum(prob_dist.values())
    print(f"概率和: {total_prob:.6f} (应该接近1.0)")

def test_large_scale_sampling():
    """测试大规模采样"""
    print("\n" + "="*50)
    print("=== 测试大规模采样 ===\n")
    
    # 创建4比特量子态
    state = StateVector(4)
    
    # 创建均匀叠加态
    state.state_vector = torch.ones(16, dtype=torch.complex64) / np.sqrt(16)
    
    print(f"4比特均匀叠加态")
    print(f"状态向量长度: {len(state.state_vector)}")
    
    # 进行大规模采样
    num_shots = 20000
    print(f"\n进行 {num_shots} 次采样...")
    
    import time
    start_time = time.time()
    
    result_dict = state.sample_final_state(num_shots=num_shots, return_json=False)
    
    end_time = time.time()
    print(f"采样耗时: {end_time - start_time:.4f} 秒")
    
    # 统计结果
    print(f"采样到的不同状态数: {len(result_dict)}")
    print(f"理论不同状态数: 16")
    
    # 检查均匀性
    expected_count = num_shots / 16
    print(f"理论每个状态期望次数: {expected_count:.1f}")
    
    counts = list(result_dict.values())
    print(f"实际计数范围: {min(counts)} - {max(counts)}")
    print(f"计数标准差: {np.std(counts):.2f}")

def test_json_output():
    """测试JSON输出格式"""
    print("\n" + "="*50)
    print("=== 测试JSON输出格式 ===\n")
    
    # 创建简单量子态
    state = StateVector(2)
    state.state_vector = torch.tensor([0.7, 0.0, 0.0, 0.7], dtype=torch.complex64)
    state.state_vector = state.state_vector / torch.norm(state.state_vector)
    
    # 进行采样并输出JSON
    num_shots = 1000
    result_json = state.sample_final_state(num_shots=num_shots, return_json=True)
    
    print("JSON输出:")
    print(result_json)
    
    # 解析JSON验证格式
    parsed_result = json.loads(result_json)
    print(f"\n解析后的字典: {parsed_result}")
    print(f"数据类型检查:")
    for key, value in parsed_result.items():
        print(f"  键 {key} (类型: {type(key)}), 值 {value} (类型: {type(value)})")

if __name__ == "__main__":
    print("开始测试StateVector的末态采样功能...\n")
    
    # 1. 测试基本采样功能
    test_basic_sampling()
    
    # 2. 测试部分量子比特采样
    test_partial_sampling()
    
    # 3. 测试电路执行后的采样
    test_circuit_sampling()
    
    # 4. 测试概率分布计算
    test_probability_distribution()
    
    # 5. 测试大规模采样
    test_large_scale_sampling()
    
    # 6. 测试JSON输出格式
    test_json_output()
    
    print("\n" + "="*50)
    print("所有测试完成！") 