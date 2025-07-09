#!/usr/bin/env python3
"""
标准化的VQE采样流程

流程：
1. 训练参数
2. 运行电路得到末态
3. 对末态调用sampler采样
4. 根据需求输出bitstring-count json文件或计算期望值
5. 结果保存在results文件夹下
"""

import sys
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.vqe import VQE, build_double_cz_pqc
from fastsim.hamiltonian import create_heisenberg_hamiltonian
from fastsim.circuit import load_gates_from_config
from fastsim.state import StateVector
from fastsim.sampling import Sampler


class ExampleVQESampling:
    """标准化的VQE采样流程"""
    
    def __init__(self, num_qubits: int, hamiltonian_params: Dict = None, 
                 pqc_params: Dict = None, vqe_params: Dict = None):
        """
        初始化标准化VQE采样
        
        Args:
            num_qubits: 量子比特数
            hamiltonian_params: 哈密顿量参数
            pqc_params: PQC参数
            vqe_params: VQE优化参数
        """
        self.num_qubits = num_qubits
        
        # 默认参数
        self.hamiltonian_params = hamiltonian_params or {'J': 1.0, 'h': 0.0}
        self.pqc_params = pqc_params or {'num_layers': 2}
        self.vqe_params = vqe_params or {
            'num_iterations': 100,
            'convergence_threshold': 1e-6,
            'patience': 50
        }
        
        # 加载门配置
        load_gates_from_config("configs/gates_config.json")
        
        # 初始化组件
        self._initialize_components()
        
        # 结果存储
        self.results = {}
    
    def _initialize_components(self):
        """初始化VQE组件"""
        # 创建哈密顿量
        self.hamiltonian = create_heisenberg_hamiltonian(
            self.num_qubits, 
            **self.hamiltonian_params
        )
        
        # 创建PQC
        self.pqc = build_double_cz_pqc(
            self.num_qubits, 
            **self.pqc_params
        )
        
        # 创建VQE
        self.vqe = VQE(self.pqc, self.hamiltonian)
        
        print(f"初始化完成:")
        print(f"  量子比特数: {self.num_qubits}")
        print(f"  哈密顿量参数: {self.hamiltonian_params}")
        print(f"  PQC参数: {self.pqc_params}")
        print(f"  VQE参数: {self.vqe_params}")
    
    def train_parameters(self) -> Dict:
        """
        训练VQE参数
        
        Returns:
            Dict: 训练结果
        """
        print(f"\n开始VQE参数训练...")
        
        # 创建初始态
        initial_state = torch.zeros(2**self.num_qubits, dtype=torch.complex64)
        initial_state[0] = 1.0  # |0...0⟩态
        initial_state = initial_state.unsqueeze(0)  # 添加batch维度
        
        # 运行优化
        result = self.vqe.optimize(
            input_state=initial_state,
            **self.vqe_params
        )
        
        print(f"训练完成:")
        print(f"  最终能量: {result.get('final_energy', 'N/A'):.6f}")
        print(f"  最佳能量: {result.get('best_energy', 'N/A'):.6f}")
        print(f"  迭代次数: {result.get('iterations', 'N/A')}")
        
        self.results['training'] = result
        return result
    
    def get_final_state(self) -> torch.Tensor:
        """
        运行优化后的电路得到末态
        
        Returns:
            torch.Tensor: 末态向量
        """
        print(f"\n计算末态...")
        
        # 创建初始态
        initial_state = torch.zeros(2**self.num_qubits, dtype=torch.complex64)
        initial_state[0] = 1.0  # |0...0⟩态
        initial_state = initial_state.unsqueeze(0)  # 添加batch维度
        
        # 运行优化后的电路
        final_state = self.vqe.pqc(initial_state)
        final_state_vector = final_state[0]  # 移除batch维度
        
        print(f"末态计算完成")
        print(f"  末态维度: {final_state_vector.shape}")
        print(f"  末态范数: {torch.norm(final_state_vector):.6f}")
        
        self.results['final_state'] = final_state_vector.detach().clone()
        return final_state_vector
    
    def sample_final_state(self, num_shots: int, 
                          qubit_indices: Optional[List[int]] = None,
                          save_json: bool = True,
                          filename_prefix: str = None) -> Dict:
        """
        对末态进行采样
        
        Args:
            num_shots: 采样次数
            qubit_indices: 要采样的量子比特索引，None表示采样所有量子比特
            save_json: 是否保存JSON文件
            filename_prefix: 文件名前缀
            
        Returns:
            Dict: 采样结果
        """
        print(f"\n开始末态采样...")
        print(f"  采样次数: {num_shots}")
        print(f"  量子比特: {qubit_indices if qubit_indices else 'all'}")
        
        # 创建StateVector和Sampler
        state = StateVector(self.num_qubits)
        state.state_vector = self.results['final_state']
        sampler = Sampler(state, default_shots=num_shots)
        
        # 进行采样
        sampling_result = sampler.sample_final_state(
            num_shots=num_shots,
            qubit_indices=qubit_indices,
            return_json=False
        )
        
        # 获取采样统计信息
        stats = sampler.get_sampling_statistics(num_shots, qubit_indices)
        
        # 保存结果
        self.results['sampling'] = {
            'sampling_result': sampling_result,
            'statistics': stats,
            'num_shots': num_shots,
            'qubit_indices': qubit_indices
        }
        
        # 保存JSON文件
        if save_json:
            if filename_prefix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_prefix = f"vqe_sampling_{self.num_qubits}q_{timestamp}"
            
            filename = f"results/{filename_prefix}_shots{num_shots}.json"
            sampler.save_sampling_results(filename, num_shots, qubit_indices)
            print(f"  采样结果已保存到: {filename}")
        
        # 打印采样摘要
        sampler.print_sampling_summary(num_shots, qubit_indices)
        
        return self.results['sampling']
    
    def calculate_expectation_from_sampling(self, observable: torch.Tensor,
                                          num_shots: int,
                                          qubit_indices: Optional[List[int]] = None,
                                          compare_with_direct: bool = True) -> Dict:
        """
        通过采样计算期望值
        
        Args:
            observable: 可观测量算符
            num_shots: 采样次数
            qubit_indices: 作用的量子比特索引
            compare_with_direct: 是否与直接计算比较
            
        Returns:
            Dict: 期望值计算结果
        """
        print(f"\n通过采样计算期望值...")
        print(f"  采样次数: {num_shots}")
        print(f"  可观测量维度: {observable.shape}")
        print(f"  量子比特: {qubit_indices if qubit_indices else 'all'}")
        
        # 创建StateVector和Sampler
        state = StateVector(self.num_qubits)
        state.state_vector = self.results['final_state']
        sampler = Sampler(state, default_shots=num_shots)
        
        # 如果指定了qubit_indices，需要提取对应的可观测量
        if qubit_indices is not None and len(qubit_indices) != self.num_qubits:
            # 对于部分量子比特，我们需要提取对应的可观测量子矩阵
            # 这里简化处理，使用单位矩阵作为可观测量
            observable_dim = 2**len(qubit_indices)
            observable = torch.eye(observable_dim, dtype=torch.complex64)
            print(f"  使用单位矩阵作为可观测量，维度: {observable.shape}")
        
        # 通过采样计算期望值
        sampling_expectation = sampler.calculate_expectation_from_sampling(
            observable, num_shots=num_shots, qubit_indices=qubit_indices
        )
        
        result = {
            'sampling_expectation': sampling_expectation,
            'num_shots': num_shots,
            'qubit_indices': qubit_indices
        }
        
        # 与直接计算比较
        if compare_with_direct:
            direct_expectation = state.get_expectation(observable, qubit_indices=qubit_indices)
            error = abs(sampling_expectation - direct_expectation)
            relative_error = (error / abs(direct_expectation)) * 100 if direct_expectation != 0 else 0
            
            result.update({
                'direct_expectation': direct_expectation,
                'error': error,
                'relative_error': relative_error
            })
            
            print(f"  采样期望值: {sampling_expectation:.6f}")
            print(f"  直接期望值: {direct_expectation:.6f}")
            print(f"  误差: {error:.6f}")
            print(f"  相对误差: {relative_error:.2f}%")
        
        self.results['expectation'] = result
        return result
    
    def run_complete_workflow(self, num_shots: int,
                             qubit_indices: Optional[List[int]] = None,
                             save_json: bool = True,
                             calculate_expectation: bool = False,
                             observable: torch.Tensor = None,
                             filename_prefix: str = None) -> Dict:
        """
        运行完整的VQE采样工作流
        
        Args:
            num_shots: 采样次数
            qubit_indices: 量子比特索引
            save_json: 是否保存JSON文件
            calculate_expectation: 是否计算期望值
            observable: 可观测量算符
            filename_prefix: 文件名前缀
            
        Returns:
            Dict: 完整结果
        """
        print("="*60)
        print("开始标准化VQE采样工作流")
        print("="*60)
        
        # 1. 训练参数
        training_result = self.train_parameters()
        
        # 2. 运行电路得到末态
        final_state = self.get_final_state()
        
        # 3. 对末态进行采样
        sampling_result = self.sample_final_state(
            num_shots=num_shots,
            qubit_indices=qubit_indices,
            save_json=save_json,
            filename_prefix=filename_prefix
        )
        
        # 4. 计算期望值（如果需要）
        if calculate_expectation and observable is not None:
            expectation_result = self.calculate_expectation_from_sampling(
                observable, num_shots, qubit_indices
            )
        
        # 5. 保存完整结果
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"vqe_complete_{self.num_qubits}q_{timestamp}"
        
        complete_filename = f"results/{filename_prefix}_complete.json"
        self._save_complete_results(complete_filename)
        print(f"\n完整结果已保存到: {complete_filename}")
        
        print("="*60)
        print("标准化VQE采样工作流完成")
        print("="*60)
        
        return self.results
    
    def _save_complete_results(self, filename: str):
        """保存完整结果到文件"""
        # 转换tensor为列表以便JSON序列化
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, torch.Tensor):
                results_to_save[key] = value.tolist()
            elif isinstance(value, dict):
                results_to_save[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        results_to_save[key][sub_key] = sub_value.tolist()
                    else:
                        results_to_save[key][sub_key] = sub_value
            else:
                results_to_save[key] = value
        
        # 添加元数据
        results_to_save['metadata'] = {
            'num_qubits': self.num_qubits,
            'hamiltonian_params': self.hamiltonian_params,
            'pqc_params': self.pqc_params,
            'vqe_params': self.vqe_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)


def main():
    """主函数：演示标准化VQE采样流程"""
    
    # 配置参数
    num_qubits = 2
    num_shots = 10000
    
    # 创建标准化VQE采样实例
    vqe_sampling = ExampleVQESampling(
        num_qubits=num_qubits,
        hamiltonian_params={'J': 1.0, 'h': 0.0},
        pqc_params={'num_layers': 2},
        vqe_params={'num_iterations': 100, 'convergence_threshold': 1e-6, 'patience': 50}
    )
    
    # 运行完整工作流
    results = vqe_sampling.run_complete_workflow(
        num_shots=num_shots,
        save_json=True,
        calculate_expectation=True,
        observable=vqe_sampling.hamiltonian,  # 使用哈密顿量作为可观测量
        filename_prefix=f"demo_vqe_{num_qubits}q"
    )
    
    print(f"\n工作流完成！结果已保存在results文件夹下。")


if __name__ == "__main__":
    main() 