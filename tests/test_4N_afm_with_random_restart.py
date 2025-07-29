#!/usr/bin/env python3
"""
使用随机重启优化4*N AFM哈密顿量
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
from fastsim.vqe import (
    PQC, VQE, build_pqc_adaptive, build_pqc_u_cz, build_pqc_rx_rz_cnot
)
from fastsim.hamiltonian import create_hamiltonian
from fastsim.circuit import load_gates_from_config
from fastsim.tool import get_hf_init_state

def test_4N_afm_with_random_restart():
    """使用随机重启优化4*N AFM哈密顿量"""
    print("=== 4*N AFM哈密顿量 + 随机重启优化 ===\n")
    
    # 加载门配置
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 测试不同的N值
    for N in [1, 2, 3]:
        num_qubits = 4 * N
        print(f"--- N={N} (4*{N}={num_qubits}比特) ---")
        
        # 创建哈密顿量
        H_operator = create_hamiltonian('paper_4n_heisenberg', N=N)
        H_dense = create_hamiltonian('paper_4n_heisenberg', N=N)
        
        # 初始态
        init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        init_state[0] = 1.0
        init_state = init_state.unsqueeze(0)
        
        # 使用自适应PQC
        pqc = build_pqc_adaptive(num_qubits)
        print(f"使用自适应PQC: {pqc.parameter_count} 个参数")
        
        # 根据系统大小调整优化参数
        if num_qubits <= 8:
            num_epochs = 5
            iterations_per_epoch = 200
            lr = 0.01
            patience = 100
        else:
            num_epochs = 8
            iterations_per_epoch = 250
            lr = 0.005
            patience = 150
        
        # 测试黑盒哈密顿量
        print(f"\n黑盒哈密顿量随机重启优化:")
        vqe_operator = VQE(pqc, H_operator, optimizer_kwargs={'lr': lr})
        result_operator = vqe_operator.optimize_with_random_restarts(
            num_epochs=num_epochs,
            iterations_per_epoch=iterations_per_epoch,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=patience,
            random_scale=0.1,
            use_seed=True
        )
        
        # 测试密集哈密顿量
        print(f"\n密集哈密顿量随机重启优化:")
        vqe_dense = VQE(build_pqc_adaptive(num_qubits), H_dense, optimizer_kwargs={'lr': lr})
        result_dense = vqe_dense.optimize_with_random_restarts(
            num_epochs=num_epochs,
            iterations_per_epoch=iterations_per_epoch,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=patience,
            random_scale=0.1,
            use_seed=True
        )
        
        print(f"\nN={N} 随机重启结果对比:")
        print(f"  黑盒哈密顿量: {result_operator['best_energy']:.6f}")
        print(f"  密集哈密顿量: {result_dense['best_energy']:.6f}")
        print(f"  能量差异: {abs(result_operator['best_energy'] - result_dense['best_energy']):.6f}")
        print(f"  总迭代次数: {result_operator['total_iterations']} / {result_dense['total_iterations']}")
        
        # 理论能量对比
        expected_energy = -7.0 * N
        print(f"  理论基态能量: {expected_energy:.6f}")
        print(f"  与理论值差异: {abs(result_operator['best_energy'] - expected_energy):.6f}")
        
        # 分析各epoch结果
        print(f"\n黑盒哈密顿量各Epoch结果:")
        for epoch_result in result_operator['epoch_results']:
            print(f"    Epoch {epoch_result['epoch']}: 最优能量 = {epoch_result['best_energy']:.6f}")
        
        print("\n" + "="*60 + "\n")

def test_4N_afm_random_restart_vs_standard():
    """比较随机重启与标准优化的性能"""
    print("=== 随机重启 vs 标准优化对比 ===\n")
    
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    for N in [1, 2]:
        num_qubits = 4 * N
        print(f"--- N={N} (4*{N}={num_qubits}比特) ---")
        
        # 创建哈密顿量
        H = create_hamiltonian('paper_4n_heisenberg', N=N)
        
        # 初始态
        init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        init_state[0] = 1.0
        init_state = init_state.unsqueeze(0)
        
        # 标准优化
        print(f"标准VQE优化:")
        pqc_standard = build_pqc_adaptive(num_qubits)
        vqe_standard = VQE(pqc_standard, H, optimizer_kwargs={'lr': 0.01})
        result_standard = vqe_standard.optimize(
            num_iterations=1000,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=200
        )
        print(f"  最终能量: {result_standard['final_energy']:.6f}")
        print(f"  最优能量: {result_standard['best_energy']:.6f}")
        print(f"  迭代次数: {result_standard['iterations']}")
        
        # 随机重启优化
        print(f"\n随机重启VQE优化:")
        pqc_restart = build_pqc_adaptive(num_qubits)
        vqe_restart = VQE(pqc_restart, H, optimizer_kwargs={'lr': 0.01})
        result_restart = vqe_restart.optimize_with_random_restarts(
            num_epochs=5,
            iterations_per_epoch=200,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=100,
            random_scale=0.1,
            use_seed=True
        )
        print(f"  最终能量: {result_restart['final_energy']:.6f}")
        print(f"  最优能量: {result_restart['best_energy']:.6f}")
        print(f"  总迭代次数: {result_restart['total_iterations']}")
        
        # 性能对比
        improvement = result_standard['best_energy'] - result_restart['best_energy']
        print(f"\n性能对比:")
        print(f"  随机重启改善: {improvement:.6f}")
        print(f"  是否更好: {'是' if improvement > 0 else '否'}")
        
        print("\n" + "="*50 + "\n")

def test_4N_afm_with_hf_random_restart():
    """使用HF初态的随机重启优化"""
    print("=== 4*N AFM + HF初态 + 随机重启 ===\n")
    
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    for N in [1, 2, 3]:
        num_qubits = 4 * N
        print(f"--- N={N} (4*{N}={num_qubits}比特) ---")
        
        # 创建哈密顿量
        H_operator = create_hamiltonian('paper_4n_heisenberg', N=N)
        
        # 使用自适应PQC
        pqc = build_pqc_adaptive(num_qubits)
        print(f"使用自适应PQC: {pqc.parameter_count} 个参数")
        
        # 创建VQE
        vqe = VQE(pqc, H_operator, optimizer_kwargs={'lr': 0.01})
        
        # HF初态
        hf_state = get_hf_init_state(num_qubits)
        hf_energy = vqe.expectation_value(hf_state).item()
        print(f"HF能量: {hf_energy:.6f}")
        
        # 随机重启优化
        if num_qubits <= 8:
            num_epochs = 5
            iterations_per_epoch = 200
            patience = 100
        else:
            num_epochs = 8
            iterations_per_epoch = 250
            patience = 150
        
        result = vqe.optimize_with_random_restarts(
            num_epochs=num_epochs,
            iterations_per_epoch=iterations_per_epoch,
            input_state=hf_state,
            convergence_threshold=1e-6,
            patience=patience,
            random_scale=0.1,
            use_seed=True
        )
        
        print(f"随机重启VQE最终能量: {result['final_energy']:.6f}")
        print(f"随机重启VQE最优能量: {result['best_energy']:.6f}")
        print(f"能量改善: {((hf_energy - result['best_energy']) / abs(hf_energy) * 100):.2f}%")
        print(f"总迭代次数: {result['total_iterations']}")
        
        # 与理论值对比
        expected_energy = -7.0 * N
        print(f"理论基态能量: {expected_energy:.6f}")
        print(f"与理论值差异: {abs(result['best_energy'] - expected_energy):.6f}")
        
        # 分析各epoch结果
        print(f"各Epoch结果:")
        for epoch_result in result['epoch_results']:
            print(f"  Epoch {epoch_result['epoch']}: 最优能量 = {epoch_result['best_energy']:.6f}")
        
        print("\n" + "="*50 + "\n")

def test_different_random_scales():
    """测试不同随机化尺度对性能的影响"""
    print("=== 不同随机化尺度测试 ===\n")
    
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    N = 1  # 4比特系统
    num_qubits = 4 * N
    H = create_hamiltonian('paper_4n_heisenberg', N=N)
    
    # 初始态
    init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    init_state[0] = 1.0
    init_state = init_state.unsqueeze(0)
    
    # 测试不同随机化尺度
    scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = {}
    
    for scale in scales:
        print(f"测试随机化尺度: {scale}")
        
        pqc = build_pqc_adaptive(num_qubits)
        vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
        
        result = vqe.optimize_with_random_restarts(
            num_epochs=5,
            iterations_per_epoch=100,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=50,
            random_scale=scale,
            use_seed=True
        )
        
        results[scale] = result['best_energy']
        print(f"  最优能量: {result['best_energy']:.6f}")
    
    # 找出最佳尺度
    best_scale = min(results.items(), key=lambda x: x[1])
    print(f"\n最佳随机化尺度: {best_scale[0]} (能量: {best_scale[1]:.6f})")
    
    # 排名
    print(f"\n随机化尺度排名:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for i, (scale, energy) in enumerate(sorted_results, 1):
        print(f"{i}. 尺度 {scale}: {energy:.6f}")

if __name__ == "__main__":
    print("开始测试4*N AFM哈密顿量使用随机重启优化...\n")
    
    # 1. 测试不同随机化尺度
    test_different_random_scales()
    print("\n" + "="*80 + "\n")
    
    # 2. 测试随机重启 vs 标准优化
    test_4N_afm_random_restart_vs_standard()
    
    # 3. 测试随机重启优化
    test_4N_afm_with_random_restart()
    
    # 4. 测试HF初态随机重启
    test_4N_afm_with_hf_random_restart()
    
    print("测试完成！") 