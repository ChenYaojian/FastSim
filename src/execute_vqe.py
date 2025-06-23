#!/usr/bin/env python3
"""
VQE执行脚本
从data文件夹读取电路配置并执行变分量子本征求解器
"""

import torch
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import argparse

# 添加项目根目录到路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.circuit import load_gates_from_config
from src.vqe import VQE, PQC, load_circuit_from_file, create_pqc_from_config, create_random_hamiltonian, create_heisenberg_hamiltonian, create_ising_hamiltonian, create_hubbard_hamiltonian


def load_circuit_data(data_dir: str, circuit_name: str) -> Dict:
    """从data目录加载电路数据"""
    # 确保路径相对于项目根目录
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    
    circuit_path = os.path.join(data_dir, circuit_name)
    
    if os.path.isfile(circuit_path):
        # 如果是单个文件
        return load_circuit_from_file(circuit_path)
    elif os.path.isdir(circuit_path):
        # 如果是目录，查找json文件
        json_files = [f for f in os.listdir(circuit_path) if f.endswith('.json')]
        if json_files:
            return load_circuit_from_file(os.path.join(circuit_path, json_files[0]))
        else:
            raise FileNotFoundError(f"No JSON files found in {circuit_path}")
    else:
        raise FileNotFoundError(f"Circuit path not found: {circuit_path}")


def create_hamiltonian(hamiltonian_type: str, num_qubits: int, 
                      device: torch.device = None, **kwargs) -> torch.Tensor:
    """创建指定类型的哈密顿量"""
    if hamiltonian_type == "random":
        return create_random_hamiltonian(num_qubits, device)
    elif hamiltonian_type == "heisenberg":
        J = kwargs.get('J', 1.0)
        h = kwargs.get('h', 0.0)
        return create_heisenberg_hamiltonian(num_qubits, J, h, device)
    elif hamiltonian_type == "ising":
        J = kwargs.get('J', 1.0)
        h = kwargs.get('h', 0.0)
        return create_ising_hamiltonian(num_qubits, J, h, device)
    elif hamiltonian_type == "hubbard":
        U = kwargs.get('U', 4.0)
        t = kwargs.get('t', 1.0)
        # 现在Hubbard模型直接使用量子比特数，每2个量子比特表示1个格点
        return create_hubbard_hamiltonian(num_qubits, t, U, device)
    elif hamiltonian_type == "custom":
        hamiltonian_path = kwargs.get('hamiltonian_path')
        if hamiltonian_path and os.path.exists(hamiltonian_path):
            hamiltonian_data = np.load(hamiltonian_path)
            return torch.tensor(hamiltonian_data, dtype=torch.complex64, device=device)
        else:
            raise ValueError("Custom hamiltonian path not provided or file not found")
    else:
        raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")


def plot_optimization_history(energy_history: List[float], save_path: Optional[str] = None):
    """绘制优化历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, 'b-', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Optimization History')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization plot saved to {save_path}")
    
    plt.show()


def print_circuit_info(pqc: PQC):
    """打印电路信息"""
    print(f"Circuit Info:")
    print(f"  - Number of qubits: {pqc.num_qubits}")
    print(f"  - Number of gates: {len(pqc.gates)}")
    print(f"  - Number of parameters: {pqc.parameter_count}")
    print(f"  - Parameter names: {pqc.parameter_names}")
    print(f"  - Circuit structure:")
    print(pqc.draw())


def main():
    parser = argparse.ArgumentParser(description='Execute VQE optimization')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing circuit data')
    parser.add_argument('--circuit_name', type=str, default='sim_cir_input_8layers.json',
                       help='Name of the circuit file/directory')
    parser.add_argument('--num_qubits', type=int, default=None,
                       help='Number of qubits (optional, will be auto-detected from circuit)')
    parser.add_argument('--hamiltonian_type', type=str, default='hubbard',
                       choices=['random', 'heisenberg', 'ising', 'hubbard', 'custom'],
                       help='Type of hamiltonian to use')
    parser.add_argument('--U', type=float, default=4.0, help='Hubbard模型的U参数')
    parser.add_argument('--t', type=float, default=1.0, help='Hubbard模型的t参数')
    parser.add_argument('--J', type=float, default=1.0, help='Ising/Heisenberg模型的J参数')
    parser.add_argument('--h', type=float, default=0.0, help='Ising/Heisenberg模型的h参数')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of optimization iterations')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for optimizer')
    parser.add_argument('--convergence_threshold', type=float, default=1e-6,
                       help='Convergence threshold for early stopping')
    parser.add_argument('--patience', type=int, default=100,
                       help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for computation')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save optimization results')
    parser.add_argument('--plot', action='store_true',
                       help='Show optimization plot')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 加载门配置
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)
    
    try:
        # 加载电路配置
        print(f"Loading circuit from {args.data_dir}/{args.circuit_name}")
        circuit_config = load_circuit_data(args.data_dir, args.circuit_name)
        
        # 创建PQC
        print("Creating parameterized quantum circuit...")
        pqc = create_pqc_from_config(circuit_config, device=device)
        print_circuit_info(pqc)
        
        # 创建哈密顿量
        print(f"Creating {args.hamiltonian_type} hamiltonian...")
        # 对于Hubbard模型，使用电路的量子比特数作为格点数
        # 对于其他模型，也使用电路的量子比特数
        hamiltonian_qubits = pqc.num_qubits
        print(f"Using {hamiltonian_qubits} qubits/sites for {args.hamiltonian_type} hamiltonian")
        hamiltonian = create_hamiltonian(
            args.hamiltonian_type, hamiltonian_qubits, device,
            U=args.U, t=args.t, J=args.J, h=args.h
        )
        print(f"Hamiltonian shape: {hamiltonian.shape}")
        
        # 创建VQE
        print("Initializing VQE...")
        optimizer_kwargs = {'lr': args.learning_rate}
        vqe = VQE(pqc, hamiltonian, optim.Adam, optimizer_kwargs)
        
        # 执行优化
        print(f"Starting optimization for {args.iterations} iterations...")
        results = vqe.optimize(
            num_iterations=args.iterations,
            convergence_threshold=args.convergence_threshold,
            patience=args.patience
        )
        
        # 打印结果
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Final energy: {results['final_energy']:.6f}")
        print(f"Best energy: {results['best_energy']:.6f}")
        print(f"Total iterations: {results['iterations']}")
        print(f"Converged: {results['iterations'] < args.iterations}")
        
        # 获取最终参数
        final_params = pqc.get_parameters()
        print(f"\nFinal parameters:")
        for i, (name, param) in enumerate(zip(pqc.parameter_names, final_params)):
            print(f"  {name}: {param.item():.6f}")
        
        # 保存结果
        if args.save_results:
            results_data = {
                'final_energy': results['final_energy'],
                'best_energy': results['best_energy'],
                'iterations': results['iterations'],
                'energy_history': [e.item() if torch.is_tensor(e) else e for e in results['energy_history']],
                'final_parameters': final_params.detach().cpu().numpy().tolist(),
                'parameter_names': pqc.parameter_names
            }
            
            with open(args.save_results, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\nResults saved to {args.save_results}")
        
        # 绘制优化历史
        if args.plot:
            plot_optimization_history(results['energy_history'])
        
    except Exception as e:
        print(f"Error during VQE execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 