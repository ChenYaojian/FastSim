#!/usr/bin/env python3
"""
VQE采样示例脚本
演示如何使用VQE采样模块从优化后的量子态中采样比特串
"""

import torch
import torch.optim as optim
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

from src.circuit import load_gates_from_config
from src.vqe import VQE, PQC, create_pqc_from_config, create_heisenberg_hamiltonian
from src.vqe_sampler import VQESampler


def run_vqe_and_sample(num_qubits: int, circuit_config: Dict, 
                      num_shots: int = 1000, device: torch.device = None) -> Dict:
    """运行VQE优化并进行采样"""
    print(f"\n{'='*60}")
    print(f"运行 {num_qubits} 比特VQE优化和采样")
    print(f"{'='*60}")
    
    # 创建扩展的电路配置
    if num_qubits > 4:
        extended_config = create_extended_circuit_config(circuit_config, num_qubits)
        print(f"创建了 {num_qubits} 比特的扩展电路配置")
    else:
        extended_config = circuit_config
    
    # 创建PQC
    print("创建参数化量子电路...")
    pqc = create_pqc_from_config(extended_config, device=device)
    print(f"电路信息:")
    print(f"  - 量子比特数: {pqc.num_qubits}")
    print(f"  - 门数量: {len(pqc.gates)}")
    print(f"  - 参数数量: {pqc.parameter_count}")
    
    # 创建海森堡模型哈密顿量
    print("创建海森堡模型哈密顿量...")
    hamiltonian = create_heisenberg_hamiltonian(num_qubits, J=1.0, h=0.0, device=device)
    print(f"哈密顿量形状: {hamiltonian.shape}")
    
    # 创建VQE
    print("初始化VQE...")
    optimizer_kwargs = {'lr': 0.01}
    vqe = VQE(pqc, hamiltonian, optim.Adam, optimizer_kwargs)
    
    # 执行优化
    print("开始优化...")
    results = vqe.optimize(
        num_iterations=300,  # 减少迭代次数以加快运行
        convergence_threshold=1e-6,
        patience=50
    )
    
    # 打印优化结果
    print(f"\n优化结果:")
    print(f"  - 最终能量: {results['final_energy']:.6f}")
    print(f"  - 最佳能量: {results['best_energy']:.6f}")
    print(f"  - 总迭代次数: {results['iterations']}")
    print(f"  - 是否收敛: {results['iterations'] < 300}")
    
    # 创建VQE采样器
    print("\n创建VQE采样器...")
    sampler = VQESampler(vqe, device=device)
    
    # 进行采样
    print(f"开始采样 {num_shots} 次...")
    bitstrings = sampler.sample_bitstrings(num_shots=num_shots)
    
    # 获取统计信息
    bitstring_freqs = sampler.get_bitstring_statistics(num_shots=num_shots)
    energy_freqs = sampler.get_energy_distribution(num_shots=num_shots)
    entanglement_entropy = sampler.get_entanglement_entropy()
    
    # 打印采样结果
    print(f"\n采样结果:")
    print(f"  - 纠缠熵: {entanglement_entropy:.6f}")
    print(f"  - 最常见的比特串:")
    
    # 显示前5个最常见的比特串
    sorted_bitstrings = sorted(bitstring_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (bitstring, freq) in enumerate(sorted_bitstrings, 1):
        print(f"    {i}. {bitstring}: {freq:.4f} ({freq*100:.2f}%)")
    
    # 保存采样结果
    sampling_results = sampler.save_sampling_results(
        num_shots=num_shots,
        save_path=f"vqe_sampling_{num_qubits}q.json"
    )
    
    return {
        'num_qubits': num_qubits,
        'vqe_results': results,
        'sampling_results': sampling_results,
        'sampler': sampler
    }


def create_extended_circuit_config(base_config: Dict, target_qubits: int) -> Dict:
    """基于4比特电路创建扩展的电路配置"""
    extended_config = {}
    
    # 复制原始层
    for layer_idx in base_config.keys():
        if layer_idx == "5":  # 跳过测量层
            continue
            
        extended_config[layer_idx] = []
        for gate_info in base_config[layer_idx]:
            # 复制门到所有量子比特
            for qubit_offset in range(0, target_qubits, 4):
                new_gate = gate_info.copy()
                new_qubits = [q + qubit_offset for q in gate_info["qubits"]]
                if max(new_qubits) < target_qubits:  # 确保不超出目标量子比特数
                    new_gate["qubits"] = new_qubits
                    extended_config[layer_idx].append(new_gate)
    
    # 添加额外的纠缠层
    extra_layer_idx = len(extended_config)
    for i in range(0, target_qubits - 1, 2):
        if i + 1 < target_qubits:
            extended_config[str(extra_layer_idx)] = [{
                "gate_name": "CZ",
                "parameters": {},
                "qubits": [i, i + 1]
            }]
            extra_layer_idx += 1
    
    return extended_config


def plot_comparison(all_results: List[Dict]):
    """绘制不同比特数的采样结果对比"""
    import matplotlib.pyplot as plt
    
    num_qubits_list = [result['num_qubits'] for result in all_results]
    entanglement_entropies = [result['sampling_results']['entanglement_entropy'] for result in all_results]
    final_energies = [result['vqe_results']['final_energy'] for result in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 纠缠熵对比
    ax1.plot(num_qubits_list, entanglement_entropies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Entanglement Entropy')
    ax1.set_title('Entanglement Entropy vs System Size')
    ax1.grid(True, alpha=0.3)
    
    # 最终能量对比
    ax2.plot(num_qubits_list, final_energies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Final Energy')
    ax2.set_title('Final Energy vs System Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vqe_sampling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    # 加载基础电路配置
    circuit_path = os.path.join(project_root, "data", "circuit.json")
    with open(circuit_path, 'r') as f:
        base_circuit_config = json.load(f)
    
    # 要测试的量子比特数（选择较小的系统以加快运行）
    qubit_counts = [4, 6, 8]
    num_shots = 500  # 采样次数
    
    # 存储所有结果
    all_results = []
    
    # 运行每个比特数的VQE和采样
    for num_qubits in qubit_counts:
        try:
            result = run_vqe_and_sample(num_qubits, base_circuit_config, num_shots, device)
            all_results.append(result)
        except Exception as e:
            print(f"运行 {num_qubits} 比特VQE采样时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制对比图
    if len(all_results) > 1:
        plot_comparison(all_results)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("VQE采样结果总结")
    print(f"{'='*80}")
    print(f"{'量子比特数':<10} {'最终能量':<15} {'纠缠熵':<15} {'采样次数':<10}")
    print("-" * 80)
    
    for result in all_results:
        num_qubits = result['num_qubits']
        final_energy = result['vqe_results']['final_energy']
        entanglement_entropy = result['sampling_results']['entanglement_entropy']
        num_shots = result['sampling_results']['num_shots']
        print(f"{num_qubits:<10} {final_energy:<15.6f} {entanglement_entropy:<15.6f} {num_shots:<10}")


if __name__ == "__main__":
    main() 