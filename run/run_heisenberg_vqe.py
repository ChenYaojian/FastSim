#!/usr/bin/env python3
"""
运行不同比特数的海森堡模型VQE
"""

import torch
import torch.optim as optim
import numpy as np
import json
import os
import sys
from typing import Dict, List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.circuit import load_gates_from_config, QuantumGate
from fastsim.vqe import VQE, PQC, create_pqc_from_config, create_heisenberg_hamiltonian


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


def run_heisenberg_vqe(num_qubits: int, circuit_config: Dict, device: torch.device) -> Dict:
    """运行指定比特数的海森堡模型VQE"""
    print(f"\n{'='*60}")
    print(f"运行 {num_qubits} 比特海森堡模型VQE")
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
        num_iterations=500,  # 减少迭代次数以加快运行
        convergence_threshold=1e-6,
        patience=50
    )
    
    # 打印结果
    print(f"\n优化结果:")
    print(f"  - 最终能量: {results['final_energy']:.6f}")
    print(f"  - 最佳能量: {results['best_energy']:.6f}")
    print(f"  - 总迭代次数: {results['iterations']}")
    print(f"  - 是否收敛: {results['iterations'] < 500}")
    
    return {
        'num_qubits': num_qubits,
        'final_energy': results['final_energy'],
        'best_energy': results['best_energy'],
        'iterations': results['iterations'],
        'converged': results['iterations'] < 500
    }


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    print(f"加载门配置从: {config_path}")
    load_gates_from_config(config_path)
    
    # 检查已注册的门
    print(f"已注册的门: {list(QuantumGate._registry.keys())}")
    
    # 加载基础电路配置
    circuit_path = os.path.join(project_root, "data", "circuit.json")
    with open(circuit_path, 'r') as f:
        base_circuit_config = json.load(f)
    
    # 要测试的量子比特数
    qubit_counts = [4, 8, 12, 15]
    
    # 存储所有结果
    all_results = []
    
    # 运行每个比特数的VQE
    for num_qubits in qubit_counts:
        try:
            result = run_heisenberg_vqe(num_qubits, base_circuit_config, device)
            all_results.append(result)
        except Exception as e:
            print(f"运行 {num_qubits} 比特VQE时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print(f"\n{'='*80}")
    print("海森堡模型VQE结果总结")
    print(f"{'='*80}")
    print(f"{'量子比特数':<10} {'最终能量':<15} {'最佳能量':<15} {'迭代次数':<10} {'收敛':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['num_qubits']:<10} {result['final_energy']:<15.6f} "
              f"{result['best_energy']:<15.6f} {result['iterations']:<10} "
              f"{'是' if result['converged'] else '否':<8}")
    
    # 保存结果到文件
    results_file = os.path.join(project_root, "heisenberg_vqe_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main() 