#!/usr/bin/env python3
"""
VQE采样运行器

允许用户指定各种参数来运行标准化的VQE采样流程
"""

import sys
import os
import argparse
import torch
import json
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from run.standard_vqe_sampling import ExampleVQESampling


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VQE采样运行器')
    
    # 基本参数
    parser.add_argument('--num_qubits', type=int, default=2, help='量子比特数')
    parser.add_argument('--num_shots', type=int, default=10000, help='采样次数')
    parser.add_argument('--num_layers', type=int, default=2, help='PQC层数')
    parser.add_argument('--num_iterations', type=int, default=100, help='VQE优化迭代次数')
    
    # 哈密顿量参数
    parser.add_argument('--J', type=float, default=1.0, help='海森堡耦合强度')
    parser.add_argument('--h', type=float, default=0.0, help='外场强度')
    
    # 采样选项
    parser.add_argument('--qubit_indices', type=str, default=None, 
                       help='要采样的量子比特索引，格式如"0,1"或"0,1,2"')
    parser.add_argument('--no_save_json', action='store_true', 
                       help='不保存JSON文件')
    parser.add_argument('--no_expectation', action='store_true', 
                       help='不计算期望值')
    
    # 输出选项
    parser.add_argument('--filename_prefix', type=str, default=None, 
                       help='输出文件名前缀')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='输出目录')
    
    # 优化参数
    parser.add_argument('--convergence_threshold', type=float, default=1e-6, 
                       help='收敛阈值')
    parser.add_argument('--patience', type=int, default=50, 
                       help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='学习率')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析量子比特索引
    qubit_indices = None
    if args.qubit_indices:
        qubit_indices = [int(x.strip()) for x in args.qubit_indices.split(',')]
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置参数
    hamiltonian_params = {
        'J': args.J,
        'h': args.h
    }
    
    pqc_params = {
        'num_layers': args.num_layers
    }
    
    vqe_params = {
        'num_iterations': args.num_iterations,
        'convergence_threshold': args.convergence_threshold,
        'patience': args.patience
    }
    
    print(f"VQE采样配置:")
    print(f"  量子比特数: {args.num_qubits}")
    print(f"  采样次数: {args.num_shots}")
    print(f"  PQC层数: {args.num_layers}")
    print(f"  优化迭代次数: {args.num_iterations}")
    print(f"  哈密顿量参数: J={args.J}, h={args.h}")
    print(f"  采样量子比特: {qubit_indices if qubit_indices else 'all'}")
    print(f"  输出目录: {args.output_dir}")
    
    # 创建标准化VQE采样实例
    vqe_sampling = ExampleVQESampling(
        num_qubits=args.num_qubits,
        hamiltonian_params=hamiltonian_params,
        pqc_params=pqc_params,
        vqe_params=vqe_params
    )
    
    # 设置文件名前缀
    if args.filename_prefix is None:
        args.filename_prefix = f"vqe_{args.num_qubits}q_J{args.J}_h{args.h}_shots{args.num_shots}"
    
    # 运行完整工作流
    results = vqe_sampling.run_complete_workflow(
        num_shots=args.num_shots,
        qubit_indices=qubit_indices,
        save_json=not args.no_save_json,
        calculate_expectation=not args.no_expectation,
        observable=vqe_sampling.hamiltonian,  # 使用哈密顿量作为可观测量
        filename_prefix=args.filename_prefix
    )
    
    print(f"\nVQE采样完成！")
    print(f"结果已保存在 {args.output_dir} 文件夹下")
    
    # 打印关键结果
    if 'training' in results:
        training = results['training']
        print(f"\n训练结果:")
        print(f"  最终能量: {training.get('final_energy', 'N/A'):.6f}")
        print(f"  最佳能量: {training.get('best_energy', 'N/A'):.6f}")
    
    if 'sampling' in results:
        sampling = results['sampling']
        stats = sampling['statistics']
        print(f"\n采样结果:")
        print(f"  最常见状态: |{stats['most_common_state']['bitstring']}⟩ "
              f"(概率: {stats['most_common_state']['probability']:.4f})")
    
    if 'expectation' in results:
        expectation = results['expectation']
        print(f"\n期望值结果:")
        print(f"  采样期望值: {expectation['sampling_expectation']:.6f}")
        if 'direct_expectation' in expectation:
            print(f"  直接期望值: {expectation['direct_expectation']:.6f}")
            print(f"  相对误差: {expectation['relative_error']:.2f}%")


if __name__ == "__main__":
    main() 