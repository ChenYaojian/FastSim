#!/usr/bin/env python3
"""
快速运行准一维反铁磁模型的VQE计算（4和8比特）
"""

import torch
import numpy as np
import json
import os
import sys
from typing import Dict, List

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.append(project_root)
sys.path.append(src_path)

from src.vqe import VQE, PQC, create_quasi_1d_afm_hamiltonian, load_circuit_from_file, create_pqc_from_config
from src.circuit import load_gates_from_config


def run_quasi_1d_afm_vqe_quick(num_qubits_list: List[int], circuit_path: str, 
                              J_perp: float = 0.5, J_parallel: float = 1.0, h: float = 0.0,
                              num_iterations: int = 500, lr: float = 0.01):
    """
    快速运行准一维反铁磁模型的VQE计算
    
    Args:
        num_qubits_list: 量子比特数列表
        circuit_path: 电路配置文件路径
        J_perp: 横向相互作用强度
        J_parallel: 纵向相互作用强度
        h: 外场强度
        num_iterations: 优化迭代次数
        lr: 学习率
    """
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    # 加载电路配置
    circuit_config = load_circuit_from_file(circuit_path)
    
    results = {}
    
    for num_qubits in num_qubits_list:
        print(f"\n{'='*60}")
        print(f"运行 {num_qubits} 比特的准一维反铁磁模型VQE")
        print(f"参数: J⊥ = {J_perp}, J∥ = {J_parallel}, h = {h}")
        print(f"{'='*60}")
        
        try:
            # 创建哈密顿量
            hamiltonian = create_quasi_1d_afm_hamiltonian(
                num_qubits=num_qubits,
                J_perp=J_perp,
                J_parallel=J_parallel,
                h=h
            )
            
            # 创建PQC
            pqc = create_pqc_from_config(circuit_config, num_qubits)
            
            # 创建VQE
            vqe = VQE(
                pqc=pqc,
                hamiltonian=hamiltonian,
                optimizer_kwargs={'lr': lr}
            )
            
            # 运行优化
            print(f"开始优化，最大迭代次数: {num_iterations}")
            optimization_result = vqe.optimize(
                num_iterations=num_iterations,
                convergence_threshold=1e-6,
                patience=100
            )
            
            # 获取最终能量和基态
            final_energy = optimization_result['final_energy']
            best_energy = optimization_result['best_energy']
            final_state = vqe.get_ground_state()
            
            # 计算基态能量（用于验证）
            with torch.no_grad():
                ground_state_energy = vqe.expectation_value(final_state).item()
            
            results[num_qubits] = {
                'final_energy': final_energy,
                'best_energy': best_energy,
                'ground_state_energy': ground_state_energy,
                'energy_history': optimization_result['energy_history'],
                'iterations': optimization_result['iterations'],
                'parameters': vqe.pqc.get_parameters().detach().cpu().numpy().tolist()
            }
            
            print(f"✅ {num_qubits} 比特完成:")
            print(f"   最终能量: {final_energy:.6f}")
            print(f"   最佳能量: {best_energy:.6f}")
            print(f"   基态能量: {ground_state_energy:.6f}")
            print(f"   迭代次数: {optimization_result['iterations']}")
            
        except Exception as e:
            print(f"❌ {num_qubits} 比特运行失败: {str(e)}")
            results[num_qubits] = {'error': str(e)}
    
    return results


def compare_with_heisenberg():
    """对比准一维AFM与Heisenberg模型的结果"""
    print("\n" + "="*80)
    print("准一维AFM vs Heisenberg模型能量对比")
    print("="*80)
    
    # 加载Heisenberg结果
    heisenberg_file = "heisenberg_vqe_results.json"
    if os.path.exists(heisenberg_file):
        with open(heisenberg_file, 'r') as f:
            heisenberg_results = json.load(f)
        
        print(f"{'比特数':<8} {'Heisenberg':<12} {'Quasi-1D AFM':<15} {'差异':<10}")
        print("-" * 50)
        
        for num_qubits in [4, 8]:
            if str(num_qubits) in heisenberg_results:
                heisenberg_energy = heisenberg_results[str(num_qubits)]['best_energy']
                print(f"{num_qubits:<8} {heisenberg_energy:<12.6f} {'待计算':<15} {'-':<10}")
    else:
        print("未找到Heisenberg模型结果文件")


def main():
    """主函数"""
    print("准一维反铁磁模型VQE计算（快速版本）")
    
    # 配置参数
    num_qubits_list = [4, 8]  # 只运行4和8比特
    circuit_path = os.path.join(project_root, "data", "circuit_0623_converted.json")
    
    # 准一维AFM参数（典型值）
    J_perp = 0.3    # 横向相互作用（较弱）
    J_parallel = 1.0  # 纵向相互作用（较强）
    h = 0.0         # 外场
    
    # 运行VQE
    results = run_quasi_1d_afm_vqe_quick(
        num_qubits_list=num_qubits_list,
        circuit_path=circuit_path,
        J_perp=J_perp,
        J_parallel=J_parallel,
        h=h,
        num_iterations=500,  # 减少迭代次数
        lr=0.01
    )
    
    # 保存结果
    output_file = "quasi_1d_afm_vqe_quick_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 对比Heisenberg模型
    compare_with_heisenberg()
    
    # 打印总结
    print(f"\n{'='*60}")
    print("计算总结")
    print(f"{'='*60}")
    for num_qubits in num_qubits_list:
        if num_qubits in results and 'error' not in results[num_qubits]:
            print(f"{num_qubits} 比特: 能量 = {results[num_qubits]['best_energy']:.6f}")
        else:
            print(f"{num_qubits} 比特: 计算失败")
    
    print(f"\n参数设置:")
    print(f"  J⊥ (横向) = {J_perp}")
    print(f"  J∥ (纵向) = {J_parallel}")
    print(f"  h (外场) = {h}")
    print(f"  J⊥/J∥ 比值 = {J_perp/J_parallel:.3f}")
    
    print(f"\n物理意义:")
    print(f"  - 当 J⊥/J∥ = {J_perp/J_parallel:.3f} < 1 时，系统表现出准一维特性")
    print(f"  - 纵向相互作用占主导，形成链状磁性结构")
    print(f"  - 横向相互作用较弱，层间耦合有限")


if __name__ == "__main__":
    main() 