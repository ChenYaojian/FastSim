#!/usr/bin/env python3
"""
测试Circuit.from_json函数对两种格式的支持
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import torch
from fastsim.circuit import Circuit, load_gates_from_config

def create_test_circuits():
    """创建测试用的电路文件"""
    
    # 加载门配置
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 1. 分层格式的电路
    layered_circuit = {
        "0": [
            {"gate_name": "RX", "parameters": {"theta": 0.5}, "qubits": [0]},
            {"gate_name": "RZ", "parameters": {"theta": 0.3}, "qubits": [1]},
            {"gate_name": "CNOT", "parameters": {}, "qubits": [0, 1]}
        ],
        "1": [
            {"gate_name": "RX", "parameters": {"theta": 0.2}, "qubits": [0]},
            {"gate_name": "CZ", "parameters": {}, "qubits": [1, 2]}
        ],
        "2": [
            {"gate_name": "U", "parameters": {"alpha": 0.1, "beta": 0.4, "gamma": 0.7}, "qubits": [0]},
            {"gate_name": "CNOT", "parameters": {}, "qubits": [2, 3]}
        ]
    }
    
    # 2. 门序列格式的电路
    sequence_circuit = [
        {"gate_name": "RX", "parameters": {"theta": 0.5}, "qubits": [0]},
        {"gate_name": "RZ", "parameters": {"theta": 0.3}, "qubits": [1]},
        {"gate_name": "CNOT", "parameters": {}, "qubits": [0, 1]},
        {"gate_name": "RX", "parameters": {"theta": 0.2}, "qubits": [0]},
        {"gate_name": "CZ", "parameters": {}, "qubits": [1, 2]},
        {"gate_name": "U", "parameters": {"alpha": 0.1, "beta": 0.4, "gamma": 0.7}, "qubits": [0]},
        {"gate_name": "CNOT", "parameters": {}, "qubits": [2, 3]}
    ]
    
    # 保存到文件
    with open("test_layered_circuit.json", "w") as f:
        json.dump(layered_circuit, f, indent=2)
    
    with open("test_sequence_circuit.json", "w") as f:
        json.dump(sequence_circuit, f, indent=2)
    
    return layered_circuit, sequence_circuit

def test_circuit_loading():
    """测试电路加载功能"""
    print("=== 测试Circuit.from_json函数 ===\n")
    
    # 创建测试电路
    layered_circuit, sequence_circuit = create_test_circuits()
    
    # 测试1: 从文件加载分层格式
    print("1. 测试从文件加载分层格式:")
    try:
        circuit1 = Circuit.from_json("test_layered_circuit.json")
        print(f"   成功加载分层电路，比特数: {circuit1.num_qubits}, 门数: {len(circuit1.gates)}")
        print(f"   电路图:\n{circuit1.draw()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试2: 从文件加载门序列格式
    print("2. 测试从文件加载门序列格式:")
    try:
        circuit2 = Circuit.from_json("test_sequence_circuit.json")
        print(f"   成功加载门序列电路，比特数: {circuit2.num_qubits}, 门数: {len(circuit2.gates)}")
        print(f"   电路图:\n{circuit2.draw()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试3: 从dict加载分层格式
    print("3. 测试从dict加载分层格式:")
    try:
        circuit3 = Circuit.from_json(layered_circuit)
        print(f"   成功从dict加载分层电路，比特数: {circuit3.num_qubits}, 门数: {len(circuit3.gates)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试4: 从dict加载门序列格式
    print("4. 测试从dict加载门序列格式:")
    try:
        circuit4 = Circuit.from_json(sequence_circuit)
        print(f"   成功从dict加载门序列电路，比特数: {circuit4.num_qubits}, 门数: {len(circuit4.gates)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试5: 比较两种格式的结果
    print("5. 比较两种格式的结果:")
    if len(circuit1.gates) == len(circuit2.gates):
        print(f"   两种格式的门数量相同: {len(circuit1.gates)}")
        
        # 比较门序列
        gates_match = True
        for i, ((gate1, qubits1), (gate2, qubits2)) in enumerate(zip(circuit1.gates, circuit2.gates)):
            if gate1.name != gate2.name or qubits1 != qubits2:
                gates_match = False
                print(f"   第{i}个门不匹配: {gate1.name}{qubits1} vs {gate2.name}{qubits2}")
                break
        
        if gates_match:
            print("   门序列完全匹配！")
    else:
        print(f"   门数量不同: 分层格式 {len(circuit1.gates)}, 门序列格式 {len(circuit2.gates)}")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===\n")
    
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 测试1: 空电路
    print("1. 测试空电路:")
    empty_layered = {}
    empty_sequence = []
    
    try:
        circuit1 = Circuit.from_json(empty_layered)
        print(f"   空分层电路: 比特数 {circuit1.num_qubits}, 门数 {len(circuit1.gates)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    try:
        circuit2 = Circuit.from_json(empty_sequence)
        print(f"   空门序列电路: 比特数 {circuit2.num_qubits}, 门数 {len(circuit2.gates)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试2: 单门电路
    print("2. 测试单门电路:")
    single_gate = [
        {"gate_name": "H", "parameters": {}, "qubits": [0]}
    ]
    
    try:
        circuit = Circuit.from_json(single_gate)
        print(f"   单门电路: 比特数 {circuit.num_qubits}, 门数 {len(circuit.gates)}")
        print(f"   电路图:\n{circuit.draw()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 测试3: 包含测量门的电路
    print("3. 测试包含测量门的电路:")
    circuit_with_measurement = [
        {"gate_name": "RX", "parameters": {"theta": 0.5}, "qubits": [0]},
        {"gate_name": "MX", "parameters": {}, "qubits": [0]},  # 测量门
        {"gate_name": "CNOT", "parameters": {}, "qubits": [0, 1]}
    ]
    
    try:
        circuit = Circuit.from_json(circuit_with_measurement)
        print(f"   包含测量门的电路: 比特数 {circuit.num_qubits}, 门数 {len(circuit.gates)}")
        print(f"   电路图:\n{circuit.draw()}")
    except Exception as e:
        print(f"   错误: {e}")

def test_circuit_execution():
    """测试电路执行"""
    print("\n=== 测试电路执行 ===\n")
    
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 创建一个简单的电路
    simple_circuit = [
        {"gate_name": "H", "parameters": {}, "qubits": [0]},
        {"gate_name": "CNOT", "parameters": {}, "qubits": [0, 1]},
        {"gate_name": "RX", "parameters": {"theta": 0.5}, "qubits": [1]}
    ]
    
    try:
        circuit = Circuit.from_json(simple_circuit)
        print(f"电路信息: 比特数 {circuit.num_qubits}, 门数 {len(circuit.gates)}")
        print(f"电路图:\n{circuit.draw()}")
        
        # 执行电路
        initial_state = torch.zeros(4, dtype=torch.complex64)
        initial_state[0] = 1.0  # |00⟩态
        initial_state = initial_state.unsqueeze(0)  # 添加batch维度
        
        output_state = circuit(initial_state)
        print(f"\n输入态: |00⟩")
        print(f"输出态振幅: {output_state[0].abs()}")
        print(f"输出态相位: {output_state[0].angle()}")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("开始测试Circuit.from_json函数的格式支持...\n")
    
    # 1. 测试基本加载功能
    test_circuit_loading()
    
    # 2. 测试边界情况
    test_edge_cases()
    
    # 3. 测试电路执行
    test_circuit_execution()
    
    # 清理测试文件
    import os
    for filename in ["test_layered_circuit.json", "test_sequence_circuit.json"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\n测试完成！") 