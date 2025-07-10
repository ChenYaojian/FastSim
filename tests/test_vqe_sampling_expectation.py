#!/usr/bin/env python3
"""
测试VQE末态采样计算期望值与直接计算的对比
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
import json
from typing import List, Union
from fastsim.vqe import VQE, build_pqc_u_cz
from fastsim.hamiltonian import create_heisenberg_hamiltonian
from fastsim.circuit import load_gates_from_config
from fastsim.state import StateVector
from fastsim.sampling import Sampler, calculate_expectation_from_sampling

def create_test_hamiltonian(num_qubits: int, J: float = 1.0, h: float = 0.0):
    """创建测试用的海森堡哈密顿量"""
    return create_heisenberg_hamiltonian(num_qubits, J=J, h=h)

def get_hamiltonian_matrix(hamiltonian) -> torch.Tensor:
    """获取哈密顿量的矩阵表示，兼容小系统和大系统"""
    if isinstance(hamiltonian, torch.Tensor):
        # 小系统：直接返回矩阵
        return hamiltonian
    else:
        # 大系统：通过矩阵-向量乘法构建矩阵
        dim = hamiltonian.dim
        matrix = torch.zeros(dim, dim, dtype=torch.complex64, device=hamiltonian.device)
        
        # 逐列构建矩阵
        for i in range(dim):
            basis_vector = torch.zeros(dim, dtype=torch.complex64, device=hamiltonian.device)
            basis_vector[i] = 1.0
            matrix[:, i] = hamiltonian @ basis_vector
        
        return matrix

def test_2qubit_vqe_sampling():
    """测试2比特VQE的采样期望值计算"""
    print("=== 测试2比特VQE采样期望值计算 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建2比特海森堡哈密顿量
    num_qubits = 2
    hamiltonian = create_test_hamiltonian(num_qubits, J=1.0, h=0.0)
    
    # 获取哈密顿量矩阵
    hamiltonian_matrix = get_hamiltonian_matrix(hamiltonian)
    print(f"哈密顿量矩阵:\n{hamiltonian_matrix}")
    
    # 创建PQC（参数化量子电路）
    pqc = build_pqc_u_cz(num_qubits, num_layers=2)
    
    # 创建VQE实例
    vqe = VQE(pqc, hamiltonian)
    
    # 运行VQE优化
    print("\n开始VQE优化...")
    result = vqe.optimize(num_iterations=100, convergence_threshold=1e-6, patience=50)
    
    print(f"优化结果:")
    print(f"  最终能量: {result.get('final_energy', 'N/A'):.6f}")
    print(f"  最佳能量: {result.get('best_energy', 'N/A'):.6f}")
    print(f"  迭代次数: {result.get('iterations', 'N/A')}")
    if 'converged' in result:
        print(f"  收敛: {result['converged']}")
    else:
        print(f"  收敛: 未知")
    
    # 获取优化后的电路和末态
    optimized_circuit = vqe.pqc
    print(f"\n优化后的电路:\n{optimized_circuit.draw()}")
    
    # 执行电路得到末态
    initial_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    initial_state[0] = 1.0  # |00⟩态
    initial_state = initial_state.unsqueeze(0)  # 添加batch维度
    
    final_state = optimized_circuit(initial_state)
    final_state_vector = final_state[0]
    
    print(f"\n末态向量: {final_state_vector}")
    print(f"末态概率分布: {torch.abs(final_state_vector) ** 2}")
    
    # 创建StateVector对象和Sampler
    state = StateVector(num_qubits)
    state.state_vector = final_state_vector
    sampler = Sampler(state, default_shots=10000)
    
    # 计算直接期望值
    direct_expectation = state.get_expectation(hamiltonian_matrix)
    print(f"\n直接计算的期望值: {direct_expectation:.6f}")
    
    # 通过采样计算期望值
    num_shots_list = [1000, 5000, 10000, 20000]
    
    print(f"\n通过采样计算期望值:")
    for num_shots in num_shots_list:
        sampling_expectation = sampler.calculate_expectation_from_sampling(
            hamiltonian_matrix, num_shots=num_shots
        )
        error = abs(sampling_expectation - direct_expectation)
        print(f"  采样{num_shots}次: {sampling_expectation:.6f}, 误差: {error:.6f}")
    
    # 使用Sampler的统计功能
    print(f"\n采样统计信息:")
    sampler.print_sampling_summary(num_shots=10000)

def test_4qubit_vqe_sampling():
    """测试4比特VQE的采样期望值计算"""
    print("\n" + "="*60)
    print("=== 测试4比特VQE采样期望值计算 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建4比特海森堡哈密顿量
    num_qubits = 4
    hamiltonian = create_test_hamiltonian(num_qubits, J=1.0, h=0.0)
    
    # 获取哈密顿量矩阵
    hamiltonian_matrix = get_hamiltonian_matrix(hamiltonian)
    print(f"4比特海森堡哈密顿量")
    print(f"矩阵维度: {hamiltonian_matrix.shape}")
    
    # 创建PQC（参数化量子电路）
    pqc = build_pqc_u_cz(num_qubits, num_layers=3)
    
    # 创建VQE实例
    vqe = VQE(pqc, hamiltonian)
    
    # 运行VQE优化
    print("\n开始VQE优化...")
    result = vqe.optimize(num_iterations=200, convergence_threshold=1e-6, patience=50)
    
    print(f"优化结果:")
    print(f"  最终能量: {result.get('final_energy', 'N/A'):.6f}")
    print(f"  最佳能量: {result.get('best_energy', 'N/A'):.6f}")
    print(f"  迭代次数: {result.get('iterations', 'N/A')}")
    if 'converged' in result:
        print(f"  收敛: {result['converged']}")
    else:
        print(f"  收敛: 未知")
    
    # 获取优化后的电路和末态
    optimized_circuit = vqe.pqc
    
    # 执行电路得到末态
    initial_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    initial_state[0] = 1.0  # |0000⟩态
    initial_state = initial_state.unsqueeze(0)  # 添加batch维度
    
    final_state = optimized_circuit(initial_state)
    final_state_vector = final_state[0]
    
    # 创建StateVector对象和Sampler
    state = StateVector(num_qubits)
    state.state_vector = final_state_vector
    sampler = Sampler(state, default_shots=10000)
    
    # 计算直接期望值
    direct_expectation = state.get_expectation(hamiltonian_matrix)
    print(f"\n直接计算的期望值: {direct_expectation:.6f}")
    
    # 通过采样计算期望值
    num_shots_list = [5000, 10000, 20000, 50000]
    
    print(f"\n通过采样计算期望值:")
    for num_shots in num_shots_list:
        sampling_expectation = sampler.calculate_expectation_from_sampling(
            hamiltonian_matrix, num_shots=num_shots
        )
        error = abs(sampling_expectation - direct_expectation)
        print(f"  采样{num_shots}次: {sampling_expectation:.6f}, 误差: {error:.6f}")

def test_partial_observable():
    """测试部分可观测量（只作用于部分量子比特）"""
    print("\n" + "="*60)
    print("=== 测试部分可观测量采样期望值计算 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建3比特海森堡哈密顿量
    num_qubits = 3
    hamiltonian = create_test_hamiltonian(num_qubits, J=1.0, h=0.0)
    
    # 创建PQC（参数化量子电路）
    pqc = build_pqc_u_cz(num_qubits, num_layers=2)
    
    # 创建VQE实例
    vqe = VQE(pqc, hamiltonian)
    
    # 运行VQE优化
    print("开始VQE优化...")
    result = vqe.optimize(num_iterations=150, convergence_threshold=1e-6, patience=50)
    
    print(f"优化结果: 最终能量 = {result.get('final_energy', 'N/A'):.6f}")
    
    # 获取末态
    optimized_circuit = vqe.pqc
    initial_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    initial_state[0] = 1.0
    initial_state = initial_state.unsqueeze(0)
    
    final_state = optimized_circuit(initial_state)
    final_state_vector = final_state[0]
    
    # 创建StateVector对象和Sampler
    state = StateVector(num_qubits)
    state.state_vector = final_state_vector
    sampler = Sampler(state, default_shots=10000)
    
    # 创建部分可观测量（只作用于前两个量子比特）
    # 例如：Z⊗Z算符
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    ZZ_observable = torch.kron(Z, Z)
    
    qubit_indices = [0, 1]  # 只作用于前两个量子比特
    
    print(f"\n部分可观测量 Z⊗Z (作用于量子比特 {qubit_indices}):")
    print(f"可观测量矩阵:\n{ZZ_observable}")
    
    # 计算直接期望值
    direct_expectation = state.get_expectation(ZZ_observable, qubit_indices=qubit_indices)
    print(f"\n直接计算的期望值: {direct_expectation:.6f}")
    
    # 通过采样计算期望值
    num_shots_list = [5000, 10000, 20000]
    
    print(f"\n通过采样计算期望值:")
    for num_shots in num_shots_list:
        sampling_expectation = sampler.calculate_expectation_from_sampling(
            ZZ_observable, num_shots=num_shots, qubit_indices=qubit_indices
        )
        error = abs(sampling_expectation - direct_expectation)
        print(f"  采样{num_shots}次: {sampling_expectation:.6f}, 误差: {error:.6f}")

def test_sampling_convergence():
    """测试采样收敛性"""
    print("\n" + "="*60)
    print("=== 测试采样收敛性 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建2比特海森堡哈密顿量
    num_qubits = 2
    hamiltonian = create_test_hamiltonian(num_qubits, J=1.0, h=0.0)
    
    # 获取哈密顿量矩阵
    hamiltonian_matrix = get_hamiltonian_matrix(hamiltonian)
    
    # 创建PQC（参数化量子电路）
    pqc = build_pqc_u_cz(num_qubits, num_layers=2)
    
    # 创建VQE实例并优化
    vqe = VQE(pqc, hamiltonian)
    result = vqe.optimize(num_iterations=100, convergence_threshold=1e-6, patience=50)
    
    # 获取末态
    optimized_circuit = vqe.pqc
    initial_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    initial_state[0] = 1.0
    initial_state = initial_state.unsqueeze(0)
    
    final_state = optimized_circuit(initial_state)
    final_state_vector = final_state[0]
    
    # 创建StateVector对象和Sampler
    state = StateVector(num_qubits)
    state.state_vector = final_state_vector
    sampler = Sampler(state, default_shots=10000)
    
    # 计算直接期望值
    direct_expectation = state.get_expectation(hamiltonian_matrix)
    print(f"直接计算的期望值: {direct_expectation:.6f}")
    
    # 使用Sampler的比较功能
    comparison_result = sampler.compare_sampling_vs_direct_expectation(
        hamiltonian_matrix, 
        num_shots_list=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    )
    
    print(f"\n采样收敛性测试:")
    print(f"{'采样次数':<10} {'采样期望值':<15} {'误差':<15} {'相对误差(%)':<15}")
    print("-" * 60)
    
    for result in comparison_result['comparison_results']:
        num_shots = result['num_shots']
        sampling_expectation = result['sampling_expectation']
        error = result['error']
        relative_error = result['relative_error']
        
        print(f"{num_shots:<10} {sampling_expectation:<15.6f} {error:<15.6f} {relative_error:<15.2f}")

def test_sampler_features():
    """测试Sampler类的其他功能"""
    print("\n" + "="*60)
    print("=== 测试Sampler类功能 ===\n")
    
    # 创建一个简单的量子态
    state = StateVector(2)
    state.state_vector = torch.tensor([0.6, 0.0, 0.0, 0.8], dtype=torch.complex64)
    state.state_vector = state.state_vector / torch.norm(state.state_vector)
    
    # 创建采样器
    sampler = Sampler(state, default_shots=5000)
    
    print("测试采样统计功能:")
    sampler.print_sampling_summary()
    
    print(f"\n测试概率分布:")
    prob_dist = sampler.get_probability_distribution()
    print(f"概率分布: {prob_dist}")
    
    print(f"\n测试保存功能:")
    filename = sampler.save_sampling_results("test_sampling_results.json")
    print(f"采样结果已保存到: {filename}")
    
    print(f"\n测试便捷函数:")
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    # 对第0比特测Z
    expectation = calculate_expectation_from_sampling(state, Z, num_shots=1000, qubit_indices=[0])
    print(f"使用便捷函数计算的期望值: {expectation:.6f}")

if __name__ == "__main__":
    print("开始测试VQE末态采样期望值计算...\n")
    
    # 1. 测试2比特VQE
    test_2qubit_vqe_sampling()
    
    # 2. 测试4比特VQE
    test_4qubit_vqe_sampling()
    
    # 3. 测试部分可观测量
    test_partial_observable()
    
    # 4. 测试采样收敛性
    test_sampling_convergence()
    
    # 5. 测试Sampler类功能
    test_sampler_features()
    
    print("\n" + "="*60)
    print("所有测试完成！") 