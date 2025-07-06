#!/usr/bin/env python3
"""
测试随机重启VQE优化功能
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
from src.vqe import (
    build_pqc_adaptive, build_pqc_u_cz, VQE
)
from src.hamiltonian import create_heisenberg_hamiltonian
from src.circuit import load_gates_from_config
from src.tool import get_hf_init_state

def test_random_restart_vqe():
    """测试随机重启VQE优化"""
    print("=== 随机重启VQE优化测试 ===\n")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试不同系统大小
    for num_qubits in [4, 6, 8]:
        print(f"--- {num_qubits} 比特系统 ---")
        
        # 创建哈密顿量
        H = create_heisenberg_hamiltonian(num_qubits)
        
        # 创建PQC
        pqc = build_pqc_adaptive(num_qubits)
        print(f"PQC参数数量: {pqc.parameter_count}")
        
        # 创建VQE
        vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
        
        # 初始态
        init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        init_state[0] = 1.0
        init_state = init_state.unsqueeze(0)
        
        # 测试标准优化
        print(f"\n标准VQE优化:")
        result_standard = vqe.optimize(
            num_iterations=500,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=100
        )
        print(f"标准优化结果: 最终能量 = {result_standard['final_energy']:.6f}, 最优能量 = {result_standard['best_energy']:.6f}")
        
        # 测试随机重启优化
        print(f"\n随机重启VQE优化:")
        result_restart = vqe.optimize_with_random_restarts(
            num_epochs=5,
            iterations_per_epoch=100,
            input_state=init_state,
            convergence_threshold=1e-6,
            patience=50,
            random_scale=0.1,
            use_seed=True
        )
        
        print(f"随机重启结果: 最终能量 = {result_restart['final_energy']:.6f}, 最优能量 = {result_restart['best_energy']:.6f}")
        print(f"总迭代次数: {result_restart['total_iterations']}")
        
        # 分析每个epoch的结果
        print(f"\n各Epoch结果分析:")
        for epoch_result in result_restart['epoch_results']:
            print(f"  Epoch {epoch_result['epoch']}: 最优能量 = {epoch_result['best_energy']:.6f}, 迭代次数 = {epoch_result['iterations']}")
        
        # 计算改善
        improvement = result_standard['best_energy'] - result_restart['best_energy']
        print(f"\n随机重启改善: {improvement:.6f}")
        
        print("\n" + "="*60 + "\n")

def test_parameter_randomization():
    """测试参数随机化功能"""
    print("=== 参数随机化测试 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    num_qubits = 4
    pqc = build_pqc_adaptive(num_qubits)
    
    print(f"测试 {num_qubits} 比特系统，{pqc.parameter_count} 个参数")
    
    # 获取初始参数
    initial_params = pqc.get_parameters()
    print(f"初始参数范围: [{initial_params.min().item():.4f}, {initial_params.max().item():.4f}]")
    
    # 测试不同尺度的随机化
    scales = [0.01, 0.1, 0.5, 1.0]
    
    for scale in scales:
        print(f"\n测试随机化尺度: {scale}")
        
        # 随机化参数
        pqc.randomize_parameters(scale=scale, seed=42)
        params = pqc.get_parameters()
        
        print(f"  参数范围: [{params.min().item():.4f}, {params.max().item():.4f}]")
        print(f"  参数标准差: {params.std().item():.4f}")
        
        # 恢复初始参数
        pqc.set_parameters(initial_params)
    
    print(f"\n参数重置测试:")
    pqc.reset_parameters()  # 随机重置
    params = pqc.get_parameters()
    print(f"  重置后参数范围: [{params.min().item():.4f}, {params.max().item():.4f}]")

def test_random_restart_with_hf():
    """测试使用HF初态的随机重启优化"""
    print("\n=== 随机重启 + HF初态测试 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    num_qubits = 6
    H = create_heisenberg_hamiltonian(num_qubits)
    pqc = build_pqc_adaptive(num_qubits)
    vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
    
    # HF初态
    hf_state = get_hf_init_state(num_qubits)
    hf_energy = vqe.expectation_value(hf_state).item()
    print(f"HF能量: {hf_energy:.6f}")
    
    # 随机重启优化
    result = vqe.optimize_with_random_restarts(
        num_epochs=3,
        iterations_per_epoch=200,
        input_state=hf_state,
        convergence_threshold=1e-6,
        patience=100,
        random_scale=0.1,
        use_seed=True
    )
    
    print(f"随机重启VQE最终能量: {result['final_energy']:.6f}")
    print(f"随机重启VQE最优能量: {result['best_energy']:.6f}")
    print(f"能量改善: {((hf_energy - result['best_energy']) / abs(hf_energy) * 100):.2f}%")
    
    # 对比标准优化
    vqe_standard = VQE(build_pqc_adaptive(num_qubits), H, optimizer_kwargs={'lr': 0.01})
    result_standard = vqe_standard.optimize(
        num_iterations=600,  # 总迭代次数相同
        input_state=hf_state,
        convergence_threshold=1e-6,
        patience=100
    )
    
    print(f"标准VQE最优能量: {result_standard['best_energy']:.6f}")
    improvement = result_standard['best_energy'] - result['best_energy']
    print(f"随机重启相对于标准优化的改善: {improvement:.6f}")

def compare_optimization_strategies():
    """比较不同优化策略"""
    print("\n=== 优化策略对比 ===\n")
    
    load_gates_from_config("configs/gates_config.json")
    
    num_qubits = 4
    H = create_heisenberg_hamiltonian(num_qubits)
    init_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
    init_state[0] = 1.0
    init_state = init_state.unsqueeze(0)
    
    strategies = [
        ("标准优化", "standard"),
        ("随机重启 (3epoch)", "restart_3"),
        ("随机重启 (5epoch)", "restart_5"),
        ("随机重启 (10epoch)", "restart_10"),
    ]
    
    results = {}
    
    for name, strategy in strategies:
        print(f"测试 {name}:")
        
        if strategy == "standard":
            pqc = build_pqc_adaptive(num_qubits)
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
            result = vqe.optimize(
                num_iterations=500,
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=100
            )
            results[name] = result['best_energy']
            
        else:
            num_epochs = int(strategy.split('_')[1])
            pqc = build_pqc_adaptive(num_qubits)
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
            result = vqe.optimize_with_random_restarts(
                num_epochs=num_epochs,
                iterations_per_epoch=500 // num_epochs,
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=100 // num_epochs,
                random_scale=0.1,
                use_seed=True
            )
            results[name] = result['best_energy']
        
        print(f"  最优能量: {results[name]:.6f}")
    
    # 排名
    print(f"\n优化策略排名:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for i, (name, energy) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: {energy:.6f}")

if __name__ == "__main__":
    print("开始测试随机重启VQE优化功能...\n")
    
    # 1. 测试参数随机化
    test_parameter_randomization()
    
    # 2. 测试随机重启VQE
    test_random_restart_vqe()
    
    # 3. 测试HF初态随机重启
    test_random_restart_with_hf()
    
    # 4. 比较优化策略
    compare_optimization_strategies()
    
    print("\n测试完成！") 