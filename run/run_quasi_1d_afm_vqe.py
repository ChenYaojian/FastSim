#!/usr/bin/env python3
"""
运行准一维反铁磁模型的VQE计算
"""

import torch
import numpy as np
import json
import os
import sys
from typing import Dict, List

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
sys.path.append(project_root)
sys.path.append(src_path)

from src.vqe import VQE, PQC, create_quasi_1d_afm_hamiltonian, load_circuit_from_file, create_pqc_from_config
from src.circuit import load_gates_from_config


def run_quasi_1d_afm_vqe(num_qubits_list: List[int], circuit_path: str, 
                         J_perp: float = 0.5, J_parallel: float = 1.0, h: float = 0.0,
                         num_iterations: int = 1000, lr: float = 0.01):
    """
    运行准一维反铁磁模型的VQE计算
    
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
                patience=200
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


def save_results(results: Dict, filename: str):
    """保存结果到JSON文件"""
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = {}
    for num_qubits, result in results.items():
        if 'error' not in result:
            serializable_results[num_qubits] = {
                'final_energy': float(result['final_energy']),
                'best_energy': float(result['best_energy']),
                'ground_state_energy': float(result['ground_state_energy']),
                'energy_history': [float(e) for e in result['energy_history']],
                'iterations': result['iterations'],
                'parameters': result['parameters']
            }
        else:
            serializable_results[num_qubits] = result
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n结果已保存到: {filename}")


def print_comparison_with_heisenberg():
    """打印准一维AFM与Heisenberg模型的对比"""
    print("\n" + "="*80)
    print("准一维反铁磁模型 vs Heisenberg模型对比")
    print("="*80)
    print("Heisenberg模型:")
    print("  H = J Σ(σx_i⊗σx_{i+1} + σy_i⊗σy_{i+1} + σz_i⊗σz_{i+1}) + h Σσz_i")
    print("  - 各向同性相互作用（J_x = J_y = J_z = J）")
    print("  - 适用于各向同性磁性材料")
    print("  - 基态通常是反铁磁态（相邻自旋反平行）")
    print()
    print("准一维反铁磁模型:")
    print("  H = J⊥ Σ(σx_i⊗σx_{i+1} + σy_i⊗σy_{i+1}) + J∥ Σσz_i⊗σz_{i+1} + h Σσz_i")
    print("  - 各向异性相互作用（J⊥ ≠ J∥）")
    print("  - 通常 J⊥ < J∥，形成准一维结构")
    print("  - 适用于层状磁性材料、准一维磁性链")
    print("  - 基态性质取决于 J⊥/J∥ 比值")
    print()
    print("物理意义:")
    print("  - J⊥: 横向相互作用，控制层间耦合")
    print("  - J∥: 纵向相互作用，控制链内耦合")
    print("  - 当 J⊥ << J∥ 时，系统表现出准一维特性")
    print("  - 当 J⊥ ≈ J∥ 时，接近各向同性Heisenberg模型")
    print("="*80)


def main():
    """主函数"""
    print("准一维反铁磁模型VQE计算")
    print_comparison_with_heisenberg()
    
    # 配置参数
    num_qubits_list = [4, 8, 12, 15]  # 量子比特数
    circuit_path = os.path.join(project_root, "data", "circuit_0623_converted.json")
    
    # 准一维AFM参数（典型值）
    J_perp = 0.3    # 横向相互作用（较弱）
    J_parallel = 1.0  # 纵向相互作用（较强）
    h = 0.0         # 外场
    
    # 运行VQE
    results = run_quasi_1d_afm_vqe(
        num_qubits_list=num_qubits_list,
        circuit_path=circuit_path,
        J_perp=J_perp,
        J_parallel=J_parallel,
        h=h,
        num_iterations=1000,
        lr=0.01
    )
    
    # 保存结果
    output_file = "quasi_1d_afm_vqe_results.json"
    save_results(results, output_file)
    
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


if __name__ == "__main__":
    main() 