#!/usr/bin/env python3
"""
使用新的PQC构造函数测试4*N AFM哈密顿量
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
from src.vqe import (
    build_pqc_adaptive, build_pqc_u_cz, build_pqc_rx_rz_cnot, 
    build_pqc_alternating, VQE
)
from src.hamiltonian import (
    create_paper_4N_heisenberg_hamiltonian_operator, 
    create_paper_4N_heisenberg_hamiltonian
)
from src.circuit import load_gates_from_config
from src.tool import get_hf_init_state

def test_4N_afm_with_different_pqc():
    """测试4*N AFM哈密顿量使用不同的PQC结构"""
    print("=== 4*N AFM哈密顿量 + 新PQC结构测试 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试不同的N值
    for N in [1, 2, 3]:
        num_qubits = 4 * N
        print(f"--- N={N} (4*{N}={num_qubits}比特) ---")
        
        # 创建哈密顿量
        H_operator = create_paper_4N_heisenberg_hamiltonian_operator(N)
        H_dense = create_paper_4N_heisenberg_hamiltonian(N)
        
        # 初始态
        init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        init_state[0] = 1.0
        init_state = init_state.unsqueeze(0)
        
        # 测试不同的PQC结构
        pqc_structures = [
            ("自适应PQC", build_pqc_adaptive(num_qubits)),
            ("U+CZ (2层)", build_pqc_u_cz(num_qubits, num_layers=2)),
            ("U+CZ (3层)", build_pqc_u_cz(num_qubits, num_layers=3)),
            ("RX+RZ+CNOT (2层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=2)),
            ("RX+RZ+CNOT (3层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=3)),
            ("交替纠缠 (3层)", build_pqc_alternating(num_qubits, num_layers=3)),
        ]
        
        print(f"PQC结构参数数量:")
        for name, pqc in pqc_structures:
            print(f"  {name}: {pqc.parameter_count} 参数")
        
        print(f"\nVQE性能测试:")
        
        # 选择几个代表性结构进行详细测试
        test_structures = [
            ("自适应PQC", build_pqc_adaptive(num_qubits)),
            ("U+CZ (3层)", build_pqc_u_cz(num_qubits, num_layers=3)),
            ("RX+RZ+CNOT (3层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=3)),
        ]
        
        results = {}
        
        for name, pqc in test_structures:
            print(f"\n  {name}:")
            
            # 根据系统大小调整优化参数
            if num_qubits <= 8:
                max_iterations = 1000
                lr = 0.01
                patience = 200
            else:
                max_iterations = 1500
                lr = 0.005
                patience = 300
            
            # 测试黑盒哈密顿量
            vqe_operator = VQE(pqc, H_operator, optimizer_kwargs={'lr': lr})
            result_operator = vqe_operator.optimize(
                num_iterations=max_iterations,
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=patience
            )
            
            # 测试密集哈密顿量
            vqe_dense = VQE(pqc, H_dense, optimizer_kwargs={'lr': lr})
            result_dense = vqe_dense.optimize(
                num_iterations=max_iterations,
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=patience
            )
            
            print(f"    黑盒哈密顿量: {result_operator['final_energy']:.6f}")
            print(f"    密集哈密顿量: {result_dense['final_energy']:.6f}")
            print(f"    能量差异: {abs(result_operator['final_energy'] - result_dense['final_energy']):.6f}")
            print(f"    迭代次数: {result_operator['iterations']} / {result_dense['iterations']}")
            
            # 记录最佳结果
            best_energy = min(result_operator['final_energy'], result_dense['final_energy'])
            results[name] = best_energy
        
        # 找出最佳PQC结构
        best_structure = min(results.items(), key=lambda x: x[1])
        print(f"\n  最佳PQC结构: {best_structure[0]} (能量: {best_structure[1]:.6f})")
        
        # 理论能量对比
        expected_energy = -7.0 * N
        print(f"  理论基态能量: {expected_energy:.6f}")
        print(f"  与理论值差异: {abs(best_structure[1] - expected_energy):.6f}")
        
        print("\n" + "="*60 + "\n")

def test_4N_afm_with_hf_init():
    """使用HF初态测试4*N AFM哈密顿量"""
    print("=== 4*N AFM哈密顿量 + HF初态测试 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    for N in [1, 2, 3]:
        num_qubits = 4 * N
        print(f"--- N={N} (4*{N}={num_qubits}比特) ---")
        
        # 创建哈密顿量
        H_operator = create_paper_4N_heisenberg_hamiltonian_operator(N)
        
        # 使用自适应PQC
        pqc = build_pqc_adaptive(num_qubits)
        print(f"使用自适应PQC: {pqc.parameter_count} 个参数")
        
        # 创建VQE
        vqe = VQE(pqc, H_operator, optimizer_kwargs={'lr': 0.01})
        
        # HF初态
        hf_state = get_hf_init_state(num_qubits)
        hf_energy = vqe.expectation_value(hf_state).item()
        print(f"HF能量: {hf_energy:.6f}")
        
        # VQE优化
        max_iterations = 1000 if num_qubits <= 8 else 1500
        patience = 200 if num_qubits <= 8 else 300
        
        result = vqe.optimize(
            num_iterations=max_iterations,
            input_state=hf_state,
            convergence_threshold=1e-6,
            patience=patience
        )
        
        print(f"VQE最终能量: {result['final_energy']:.6f}")
        print(f"VQE最优能量: {result['best_energy']:.6f}")
        print(f"能量改善: {((hf_energy - result['final_energy']) / abs(hf_energy) * 100):.2f}%")
        print(f"迭代次数: {result['iterations']}")
        
        # 与理论值对比
        expected_energy = -7.0 * N
        print(f"理论基态能量: {expected_energy:.6f}")
        print(f"与理论值差异: {abs(result['final_energy'] - expected_energy):.6f}")
        
        print("\n" + "="*50 + "\n")

def compare_pqc_performance():
    """比较不同PQC结构的性能"""
    print("=== PQC结构性能对比 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    N = 1  # 4比特系统，便于快速测试
    num_qubits = 4 * N
    
    # 密集哈密顿量
    H = create_paper_4N_heisenberg_hamiltonian(N)
    
    # 初始态
    init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    init_state[0] = 1.0
    init_state = init_state.unsqueeze(0)
    
    # 测试所有PQC结构
    pqc_structures = [
        ("自适应PQC", build_pqc_adaptive(num_qubits)),
        ("U+CZ (2层)", build_pqc_u_cz(num_qubits, num_layers=2)),
        ("U+CZ (3层)", build_pqc_u_cz(num_qubits, num_layers=3)),
        ("RX+RZ+CNOT (2层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=2)),
        ("RX+RZ+CNOT (3层)", build_pqc_rx_rz_cnot(num_qubits, num_layers=3)),
        ("交替纠缠 (2层)", build_pqc_alternating(num_qubits, num_layers=2)),
        ("交替纠缠 (3层)", build_pqc_alternating(num_qubits, num_layers=3)),
    ]
    
    results = {}
    
    for name, pqc in pqc_structures:
        print(f"测试 {name} ({pqc.parameter_count} 参数):")
        
        vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
        
        result = vqe.optimize(
            num_iterations=500,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=100
        )
        
        results[name] = {
            'energy': result['final_energy'],
            'best_energy': result['best_energy'],
            'iterations': result['iterations'],
            'params': pqc.parameter_count
        }
        
        print(f"  最终能量: {result['final_energy']:.6f}")
        print(f"  最优能量: {result['best_energy']:.6f}")
        print(f"  迭代次数: {result['iterations']}")
    
    # 性能排名
    print(f"\n性能排名 (按最终能量):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['energy'])
    
    for i, (name, data) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:20s} | 能量: {data['energy']:8.6f} | 参数: {data['params']:3d} | 迭代: {data['iterations']:3d}")
    
    # 找出最佳结构
    best = sorted_results[0]
    print(f"\n最佳PQC结构: {best[0]}")
    print(f"最终能量: {best[1]['energy']:.6f}")
    print(f"参数数量: {best[1]['params']}")
    print(f"迭代次数: {best[1]['iterations']}")

if __name__ == "__main__":
    print("开始测试4*N AFM哈密顿量使用新的PQC构造函数...\n")
    
    # 1. 比较PQC性能
    compare_pqc_performance()
    print("\n" + "="*80 + "\n")
    
    # 2. 测试不同PQC结构
    test_4N_afm_with_different_pqc()
    
    # 3. 测试HF初态
    test_4N_afm_with_hf_init()
    
    print("测试完成！") 