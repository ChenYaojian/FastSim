#!/usr/bin/env python3
"""
FastSim 使用示例

这个文件展示了如何使用FastSim包进行量子电路模拟和VQE计算。
"""

import torch
import numpy as np
from fastsim import circuit, state, vqe, hamiltonian

# 加载门配置
circuit.load_gates_from_config("configs/gates_config.json")

def basic_circuit_example():
    """基本量子电路示例"""
    print("=== 基本量子电路示例 ===")
    
    # 创建2量子比特电路
    circ = circuit.Circuit(2)
    
    # 添加量子门
    circ.add_gate("H", [0])  # Hadamard门
    circ.add_gate("CNOT", [0, 1])  # CNOT门
    
    # 创建初始态 |00⟩
    initial_state = state.StateVector(2)
    print(f"初始态: {initial_state}")
    
    # 执行电路
    final_state_tensor = circ(initial_state.get_state_vector().unsqueeze(0))[0]
    final_state = state.StateVector(initial_state.num_qubits)
    final_state.initialize(final_state_tensor)
    print(f"最终态: {final_state_tensor}")
    
    # 计算测量概率
    probabilities = final_state.get_probability_distribution()
    print(f"测量概率: {probabilities}")

def vqe_example():
    """VQE算法示例"""
    print("\n=== VQE算法示例 ===")
    
    # 创建简单的2量子比特哈密顿量
    H = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.complex64)
    
    # 创建参数化量子电路
    pqc = vqe.PQC(2)
    pqc.add_parametric_gate("RX", [0], [0.5])  # 旋转X门
    pqc.add_parametric_gate("RY", [1], [0.3])  # 旋转Y门
    pqc.add_parametric_gate("CNOT", [0, 1])    # CNOT门
    
    # 创建VQE实例
    vqe_solver = vqe.VQE(pqc, H)
    
    # 运行优化
    print("开始VQE优化...")
    result = vqe_solver.optimize(num_iterations=100, convergence_threshold=1e-6)
    
    print(f"优化完成！")
    print(f"最终能量: {result['final_energy']:.6f}")
    print(f"迭代次数: {result['iterations']}")
    print(f"最优能量: {result['best_energy']:.6f}")

def heisenberg_model_example():
    """海森堡模型示例"""
    print("\n=== 海森堡模型示例 ===")
    
    # 创建4量子比特海森堡模型哈密顿量
    H = hamiltonian.create_heisenberg_hamiltonian(4, J=1.0, h=0.5)
    
    # 创建参数化量子电路
    pqc = vqe.build_pqc_u_cz(4, num_layers=2)
    
    # 创建VQE实例
    vqe_solver = vqe.VQE(pqc, H)
    
    # 运行优化
    print("开始海森堡模型VQE优化...")
    result = vqe_solver.optimize(num_iterations=200, convergence_threshold=1e-5)
    
    print(f"优化完成！")
    print(f"最终能量: {result['final_energy']:.6f}")
    print(f"迭代次数: {result['iterations']}")
    print(f"最优能量: {result['best_energy']:.6f}")

if __name__ == "__main__":
    print("FastSim 量子模拟器使用示例")
    print("=" * 40)
    
    try:
        # 运行基本电路示例
        basic_circuit_example()
        
        # 运行VQE示例
        vqe_example()
        
        # 运行海森堡模型示例
        heisenberg_model_example()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc() 