#!/usr/bin/env python3
"""
VQE模块测试脚本
"""

import torch
import numpy as np
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.circuit import load_gates_from_config
from fastsim.vqe import VQE, PQC, create_random_hamiltonian


def test_pqc_basic():
    """测试PQC基本功能"""
    print("Testing PQC basic functionality...")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    # 创建PQC
    pqc = PQC(num_qubits=2)
    
    # 添加一些门
    pqc.add_parametric_gate("RX", [0], [0.5])
    pqc.add_parametric_gate("RY", [1], [0.3])
    pqc.add_gate("CNOT", [0, 1])
    pqc.add_parametric_gate("RX", [0], [0.2])
    
    print(f"PQC created with {pqc.parameter_count} parameters")
    print(f"Parameter names: {pqc.parameter_names}")
    
    # 测试前向传播
    state = pqc.forward()
    print(f"Output state shape: {state.shape}")
    print(f"State norm: {torch.norm(state).item():.6f}")
    
    # 测试参数获取和设置
    params = pqc.get_parameters()
    print(f"Parameters shape: {params.shape}")
    print(f"Parameters: {params}")
    
    # 修改参数
    new_params = params + 0.1
    pqc.set_parameters(new_params)
    updated_params = pqc.get_parameters()
    print(f"Updated parameters: {updated_params}")
    
    return pqc


def test_vqe_basic():
    """测试VQE基本功能"""
    print("\nTesting VQE basic functionality...")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    # 创建PQC
    pqc = PQC(num_qubits=2)
    pqc.add_parametric_gate("RX", [0], [0.5])
    pqc.add_parametric_gate("RY", [1], [0.3])
    pqc.add_gate("CNOT", [0, 1])
    
    # 创建随机哈密顿量
    hamiltonian = create_random_hamiltonian(2)
    print(f"Hamiltonian shape: {hamiltonian.shape}")
    
    # 创建VQE
    vqe = VQE(pqc, hamiltonian)
    
    # 测试前向传播
    energy = vqe.forward()
    print(f"Initial energy: {energy.item():.6f}")
    
    # 测试一步优化
    energy_after_step = vqe.optimize_step()
    print(f"Energy after one optimization step: {energy_after_step:.6f}")
    
    return vqe


def test_vqe_optimization():
    """测试VQE优化过程"""
    print("\nTesting VQE optimization...")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    # 创建简单的PQC
    pqc = PQC(num_qubits=2)
    pqc.add_parametric_gate("RX", [0], [0.1])
    pqc.add_parametric_gate("RY", [1], [0.1])
    pqc.add_gate("CNOT", [0, 1])
    pqc.add_parametric_gate("RX", [0], [0.1])
    
    # 创建哈密顿量
    hamiltonian = create_random_hamiltonian(2)
    
    # 创建VQE
    vqe = VQE(pqc, hamiltonian, optimizer_kwargs={'lr': 0.01})
    
    # 执行短时间优化
    print("Starting optimization...")
    results = vqe.optimize(num_iterations=100, convergence_threshold=1e-4, patience=50)
    
    print(f"Optimization completed:")
    print(f"  Final energy: {results['final_energy']:.6f}")
    print(f"  Best energy: {results['best_energy']:.6f}")
    print(f"  Iterations: {results['iterations']}")
    
    # 检查能量是否下降
    if len(results['energy_history']) > 1:
        initial_energy = results['energy_history'][0]
        final_energy = results['energy_history'][-1]
        print(f"  Energy improvement: {initial_energy - final_energy:.6f}")
    
    return results


def main():
    """主测试函数"""
    print("="*50)
    print("VQE MODULE TEST")
    print("="*50)
    
    try:
        # 测试PQC
        pqc = test_pqc_basic()
        
        # 测试VQE
        vqe = test_vqe_basic()
        
        # 测试优化
        results = test_vqe_optimization()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 