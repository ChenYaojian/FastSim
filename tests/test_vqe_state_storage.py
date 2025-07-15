#!/usr/bin/env python3
"""
测试VQE状态存储功能
"""

import torch
import os
import sys
sys.path.append('.')

from fastsim.vqe import VQE, build_pqc_adaptive, create_heisenberg_hamiltonian
from fastsim.circuit import load_gates_from_config

def test_vqe_state_storage():
    """测试VQE状态存储功能"""
    print("=== 测试VQE状态存储功能 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 创建小系统进行测试
    num_qubits = 4
    print(f"\n创建 {num_qubits} 比特系统...")
    
    # 创建哈密顿量
    H = create_heisenberg_hamiltonian(num_qubits)
    print(f"哈密顿量维度: {H.shape}")
    
    # 创建PQC
    pqc = build_pqc_adaptive(num_qubits)
    print(f"PQC参数数量: {pqc.parameter_count}")
    
    # 创建VQE实例，启用状态存储
    vqe = VQE(pqc, H, 
              optimizer_kwargs={'lr': 0.01},
              store_best_state=True,  # 启用状态存储
              save_dir="test_vqe_results")  # 指定保存目录
    
    print(f"VQE已创建，状态存储: {'启用' if vqe.store_best_state else '禁用'}")
    
    # 执行优化
    print("\n开始VQE优化...")
    result = vqe.optimize(num_iterations=500, 
                         convergence_threshold=1e-6, 
                         patience=100)
    
    print(f"优化完成!")
    print(f"最终能量: {result['final_energy']:.6f}")
    print(f"最优能量: {result['best_energy']:.6f}")
    print(f"迭代次数: {result['iterations']}")
    
    # 获取最佳状态信息
    print("\n最佳状态信息:")
    best_info = vqe.get_best_state_info()
    for key, value in best_info.items():
        print(f"  {key}: {value}")
    
    # 保存最佳状态
    print("\n保存最佳状态...")
    vqe.save_best_state("test_heisenberg_4q")
    
    # 验证保存的文件
    print("\n验证保存的文件:")
    save_dir = "test_vqe_results"
    files = os.listdir(save_dir)
    for file in files:
        print(f"  - {file}")
    
    # 测试加载功能
    print("\n测试加载功能...")
    
    # 创建新的VQE实例
    pqc2 = build_pqc_adaptive(num_qubits)
    vqe2 = VQE(pqc2, H, 
               optimizer_kwargs={'lr': 0.01},
               store_best_state=True,
               save_dir="test_vqe_results")
    
    # 加载最佳状态
    success = vqe2.load_best_state("test_heisenberg_4q")
    if success:
        print("成功加载最佳状态!")
        
        # 验证加载的状态
        loaded_info = vqe2.get_best_state_info()
        print("加载的状态信息:")
        for key, value in loaded_info.items():
            print(f"  {key}: {value}")
        
        # 计算加载状态的能量
        with torch.no_grad():
            loaded_energy = vqe2.expectation_value(vqe2.get_ground_state()).item()
        print(f"加载状态的能量: {loaded_energy:.6f}")
        print(f"与保存能量的差异: {abs(loaded_energy - best_info['best_energy']):.2e}")
    
    print("\n=== 测试完成 ===")

def test_random_restarts_with_storage():
    """测试随机重启优化中的状态存储"""
    print("\n=== 测试随机重启优化中的状态存储 ===")
    
    # 创建系统
    num_qubits = 6
    H = create_heisenberg_hamiltonian(num_qubits)
    pqc = build_pqc_adaptive(num_qubits)
    
    # 创建VQE实例
    vqe = VQE(pqc, H, 
              optimizer_kwargs={'lr': 0.01},
              store_best_state=True,
              save_dir="test_restart_results")
    
    print(f"开始随机重启优化: {num_qubits} 比特系统")
    
    # 执行随机重启优化
    result = vqe.optimize_with_random_restarts(
        num_epochs=3,
        iterations_per_epoch=200,
        convergence_threshold=1e-6,
        patience=50,
        random_scale=0.1,
        use_seed=True
    )
    
    print(f"随机重启优化完成!")
    print(f"全局最优能量: {result['best_energy']:.6f}")
    
    # 保存最佳状态
    vqe.save_best_state("test_restart_6q")
    
    # 显示最佳状态信息
    best_info = vqe.get_best_state_info()
    print("\n最佳状态信息:")
    for key, value in best_info.items():
        print(f"  {key}: {value}")
    
    print("\n=== 随机重启测试完成 ===")

if __name__ == "__main__":
    # 运行测试
    test_vqe_state_storage()
    test_random_restarts_with_storage() 