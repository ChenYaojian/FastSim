#!/usr/bin/env python3
"""
演示不同的PQC结构构造函数
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from fastsim.vqe import (
    build_pqc_u_cz, build_pqc_rx_rz_cnot, build_pqc_alternating, 
    build_pqc_adaptive, VQE, create_heisenberg_hamiltonian
)
from fastsim.circuit import load_gates_from_config
from fastsim.tool import get_hf_init_state

def demo_pqc_structures():
    """演示不同的PQC结构"""
    print("=== PQC结构演示 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试不同比特数
    for num_qubits in [4, 6, 8]:
        print(f"--- {num_qubits} 比特系统 ---")
        
        # 创建哈密顿量
        H = create_heisenberg_hamiltonian(num_qubits)
        
        # 测试不同的PQC结构
        pqc_structures = [
            ("U+CZ (2层)", build_pqc_u_cz(num_qubits, num_layers=2)),
            ("U+CZ (3层)", build_pqc_u_cz(num_qubits, num_layers=3)),
            ("RX+RZ+CNOT (2层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=2)),
            ("RX+RZ+CNOT (3层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=3)),
            ("交替纠缠 (3层)", build_pqc_alternating(num_qubits, num_layers=3)),
            ("自适应", build_pqc_adaptive(num_qubits)),
        ]
        
        print(f"PQC结构对比 (参数数量):")
        for name, pqc in pqc_structures:
            print(f"  {name}: {pqc.parameter_count} 参数")
        
        # 选择几个结构进行VQE测试
        test_structures = [
            ("U+CZ (2层)", build_pqc_u_cz(num_qubits, num_layers=2)),
            ("RX+RZ+CNOT (2层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=2)),
            ("自适应", build_pqc_adaptive(num_qubits)),
        ]
        
        print(f"\nVQE性能测试:")
        for name, pqc in test_structures:
            print(f"\n  {name}:")
            
            # 创建VQE
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
            
            # 使用HF初态
            hf_state = get_hf_init_state(num_qubits)
            hf_energy = vqe.expectation_value(hf_state).item()
            
            # VQE优化
            result = vqe.optimize(
                num_iterations=500,
                input_state=hf_state,
                convergence_threshold=1e-5,
                patience=100
            )
            
            print(f"    HF能量: {hf_energy:.6f}")
            print(f"    VQE最终能量: {result['final_energy']:.6f}")
            print(f"    能量改善: {((hf_energy - result['final_energy']) / abs(hf_energy) * 100):.2f}%")
            print(f"    迭代次数: {result['iterations']}")
        
        print("\n" + "="*50 + "\n")

def demo_adaptive_pqc():
    """演示自适应PQC的智能选择"""
    print("=== 自适应PQC演示 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    for num_qubits in [2, 4, 6, 8, 10, 12]:
        pqc = build_pqc_adaptive(num_qubits)
        
        # 根据参数数量推断使用的结构
        if pqc.parameter_count == num_qubits * 6:  # 2层U+CZ
            structure = "U+CZ (2层)"
        elif pqc.parameter_count == num_qubits * 12:  # 3层RX+RZ+CNOT
            structure = "RX+RZ+CNOT (3层)"
        elif pqc.parameter_count == num_qubits * 24:  # 4层交替纠缠
            structure = "交替纠缠 (4层)"
        else:
            structure = "未知结构"
        
        print(f"{num_qubits:2d} 比特: {structure:15s} | {pqc.parameter_count:3d} 参数")

if __name__ == "__main__":
    demo_adaptive_pqc()
    print()
    demo_pqc_structures() 