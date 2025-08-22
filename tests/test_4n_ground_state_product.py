#!/usr/bin/env python3
"""
测试8、12、16比特系统使用4比特基态直积作为初态的效果
"""

import torch
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastsim.vqe import (
    VQE, build_pqc_hi_paper_4N, create_paper_4N_heisenberg_hamiltonian,
    compute_ground_state_4qubit, create_product_state_from_4qubit_ground,
    get_hf_init_state
)
from fastsim.circuit import load_gates_from_config


def test_4n_systems_with_ground_state_product():
    """
    测试8、12、16比特系统使用4比特基态直积作为初态的效果
    """
    # 加载门配置
    load_gates_from_config("../configs/gates_config.json")
    print("==== 4*N系统测试（使用4比特基态直积作为初态） ====")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 首先计算4比特基态
    print("\n--- 计算4比特基态 ---")
    H_4qubit = create_paper_4N_heisenberg_hamiltonian(1, device)
    ground_state_4qubit, ground_energy_4qubit = compute_ground_state_4qubit(H_4qubit, device)
    print(f"4比特基态能量: {ground_energy_4qubit:.6f}")
    
    # 测试不同的系统大小
    test_cases = [
        {'N': 2, 'num_qubits': 8, 'num_cycles': 4},
        {'N': 3, 'num_qubits': 12, 'num_cycles': 6},
        {'N': 4, 'num_qubits': 16, 'num_cycles': 8},
    ]
    
    results = []
    
    for case in test_cases:
        N = case['N']
        num_qubits = case['num_qubits']
        num_cycles = case['num_cycles']
        
        print(f"\n--- 测试 {num_qubits}比特系统 (N={N}, cycles={num_cycles}) ---")
        
        # 创建paper 4N哈密顿量
        start_time = time.time()
        H = create_paper_4N_heisenberg_hamiltonian(N, device)
        hamiltonian_time = time.time() - start_time
        print(f"哈密顿量创建时间: {hamiltonian_time:.4f}秒")
        print(f"哈密顿量维度: {H.shape}")
        
        # 创建4*N PQC
        start_time = time.time()
        pqc = build_pqc_hi_paper_4N(num_qubits, num_cycles, device)
        circuit_time = time.time() - start_time
        
        print(f"PQC创建时间: {circuit_time:.4f}秒")
        print(f"参数数量: {pqc.parameter_count}")
        
        # 创建直积态作为初始态
        start_time = time.time()
        init_state = create_product_state_from_4qubit_ground(ground_state_4qubit, N, device)
        product_time = time.time() - start_time
        print(f"直积态创建时间: {product_time:.4f}秒")
        
        # 创建VQE
        lr = 0.01 if num_qubits <= 8 else 0.005
        vqe = VQE(pqc, H, optimizer_kwargs={'lr': lr})
        
        # 计算初始态能量
        init_energy = vqe.expectation_value(init_state).item()
        print(f"直积态初始能量: {init_energy:.6f}")
        
        # 计算理论基态能量（4比特基态能量的N倍）
        theoretical_ground_energy = ground_energy_4qubit * N
        print(f"理论基态能量 (N × 4比特基态): {theoretical_ground_energy:.6f}")
        print(f"与理论基态能量差: {init_energy - theoretical_ground_energy:.6f}")
        
        # 对比Hartree-Fock初态
        hf_state = get_hf_init_state(num_qubits)
        hf_energy = vqe.expectation_value(hf_state).item()
        print(f"HF初态能量: {hf_energy:.6f}")
        print(f"直积态 vs HF改善: {hf_energy - init_energy:.6f}")
        
        # VQE优化
        max_iterations = 1500 if num_qubits <= 8 else 2000
        patience = 300 if num_qubits <= 8 else 400
        
        start_time = time.time()
        result = vqe.optimize(num_iterations=max_iterations, input_state=init_state, 
                            convergence_threshold=1e-6, patience=patience)
        opt_time = time.time() - start_time
        
        print(f"优化时间: {opt_time:.4f}秒")
        print(f"最终能量: {result['final_energy']:.6f}")
        print(f"最优能量: {result['best_energy']:.6f}")
        print(f"迭代次数: {result['iterations']}")
        
        # 计算能量改善
        improvement_from_init = (init_energy - result['best_energy']) / abs(init_energy) * 100
        improvement_from_hf = (hf_energy - result['best_energy']) / abs(hf_energy) * 100
        improvement_from_theoretical = (theoretical_ground_energy - result['best_energy']) / abs(theoretical_ground_energy) * 100
        
        results.append({
            'N': N,
            'num_qubits': num_qubits,
            'num_cycles': num_cycles,
            'parameter_count': pqc.parameter_count,
            'init_energy': init_energy,
            'hf_energy': hf_energy,
            'theoretical_ground_energy': theoretical_ground_energy,
            'final_energy': result['final_energy'],
            'best_energy': result['best_energy'],
            'iterations': result['iterations'],
            'improvement_from_init_percent': improvement_from_init,
            'improvement_from_hf_percent': improvement_from_hf,
            'improvement_from_theoretical_percent': improvement_from_theoretical,
            'optimization_time': opt_time,
            'hamiltonian_time': hamiltonian_time,
            'circuit_time': circuit_time,
            'product_time': product_time
        })
    
    # 打印总结
    print(f"\n{'='*140}")
    print("4*N系统测试结果总结（使用4比特基态直积作为初态）")
    print(f"{'='*140}")
    print(f"{'N':<4} {'比特数':<6} {'Cycles':<8} {'参数数':<8} {'直积初态':<12} {'HF初态':<12} {'理论基态':<12} {'最优能量':<12} {'改善%':<8} {'迭代数':<8}")
    print("-" * 140)
    
    for result in results:
        print(f"{result['N']:<4} {result['num_qubits']:<6} {result['num_cycles']:<8} "
              f"{result['parameter_count']:<8} {result['init_energy']:<12.6f} "
              f"{result['hf_energy']:<12.6f} {result['theoretical_ground_energy']:<12.6f} "
              f"{result['best_energy']:<12.6f} {result['improvement_from_init_percent']:<8.2f} "
              f"{result['iterations']:<8}")
    
    # 详细分析
    print(f"\n{'='*140}")
    print("详细分析")
    print(f"{'='*140}")
    for result in results:
        print(f"\n{result['num_qubits']}比特系统 (N={result['N']}):")
        print(f"  直积态 vs HF初态改善: {result['hf_energy'] - result['init_energy']:.6f}")
        print(f"  直积态 vs 理论基态差异: {result['init_energy'] - result['theoretical_ground_energy']:.6f}")
        print(f"  最终 vs 理论基态改善: {result['improvement_from_theoretical_percent']:.2f}%")
        print(f"  优化时间: {result['optimization_time']:.2f}秒")
    
    return results


def test_hi_circuit_performance():
    """
    测试HI电路在不同cycle数下的性能
    先计算4比特基态，然后用基态的直积态作为VQE的初始态
    """
    # 加载门配置
    load_gates_from_config("../configs/gates_config.json")
    print("==== HI电路性能测试（使用4比特基态直积作为初始态） ====")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建4比特paper哈密顿量
    N = 1  # 4比特系统
    H_4qubit = create_paper_4N_heisenberg_hamiltonian(N, device)
    print(f"4比特哈密顿量维度: {H_4qubit.shape}")
    
    # 计算4比特基态
    print("\n--- 计算4比特基态 ---")
    start_time = time.time()
    ground_state_4qubit, ground_energy_4qubit = compute_ground_state_4qubit(H_4qubit, device)
    ground_time = time.time() - start_time
    print(f"基态计算时间: {ground_time:.4f}秒")
    print(f"4比特基态能量: {ground_energy_4qubit:.6f}")
    
    # 测试不同的cycle数
    cycle_counts = [2, 3, 4, 5, 6]
    results = []
    
    for num_cycles in cycle_counts:
        print(f"\n--- 测试 {num_cycles} cycles ---")
        
        # 创建HI电路
        start_time = time.time()
        pqc = build_pqc_hi_paper_4N(4, num_cycles, device)
        circuit_time = time.time() - start_time
        
        print(f"电路创建时间: {circuit_time:.4f}秒")
        print(f"参数数量: {pqc.parameter_count}")
        
        # 创建VQE
        vqe = VQE(pqc, H_4qubit, optimizer_kwargs={'lr': 0.01})
        
        # 使用4比特基态作为初始态
        init_state = ground_state_4qubit
        init_energy = vqe.expectation_value(init_state).item()
        print(f"初始态能量: {init_energy:.6f}")
        print(f"与基态能量差: {init_energy - ground_energy_4qubit:.6f}")
        
        # VQE优化
        start_time = time.time()
        result = vqe.optimize(num_iterations=1000, input_state=init_state, 
                            convergence_threshold=1e-6, patience=200)
        opt_time = time.time() - start_time
        
        print(f"优化时间: {opt_time:.4f}秒")
        print(f"最终能量: {result['final_energy']:.6f}")
        print(f"最优能量: {result['best_energy']:.6f}")
        print(f"迭代次数: {result['iterations']}")
        
        # 计算能量改善
        improvement = (init_energy - result['best_energy']) / abs(init_energy) * 100
        improvement_from_ground = (ground_energy_4qubit - result['best_energy']) / abs(ground_energy_4qubit) * 100
        
        results.append({
            'num_cycles': num_cycles,
            'parameter_count': pqc.parameter_count,
            'init_energy': init_energy,
            'ground_energy': ground_energy_4qubit,
            'final_energy': result['final_energy'],
            'best_energy': result['best_energy'],
            'iterations': result['iterations'],
            'improvement_percent': improvement,
            'improvement_from_ground_percent': improvement_from_ground,
            'optimization_time': opt_time
        })
    
    # 打印总结
    print(f"\n{'='*100}")
    print("HI电路性能测试结果总结（使用4比特基态直积作为初始态）")
    print(f"{'='*100}")
    print(f"{'Cycles':<8} {'参数数':<8} {'初始能量':<12} {'基态能量':<12} {'最优能量':<12} {'改善%':<8} {'迭代数':<8}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['num_cycles']:<8} {result['parameter_count']:<8} "
              f"{result['init_energy']:<12.6f} {result['ground_energy']:<12.6f} "
              f"{result['best_energy']:<12.6f} {result['improvement_percent']:<8.2f} "
              f"{result['iterations']:<8}")
    
    return results


if __name__ == "__main__":
    print("开始测试4*N系统使用4比特基态直积作为初态的效果...")
    
    # 测试4*N系统
    print("\n" + "="*80)
    print("测试1: 4*N系统使用4比特基态直积作为初态")
    print("="*80)
    results_4n = test_4n_systems_with_ground_state_product()
    
    # 测试HI电路性能
    print("\n" + "="*80)
    print("测试2: HI电路性能测试")
    print("="*80)
    results_hi = test_hi_circuit_performance()
    
    print("\n所有测试完成！") 