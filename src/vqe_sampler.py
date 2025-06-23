#!/usr/bin/env python3
"""
VQE采样模块
从VQE优化后的量子态中采样比特串
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
from collections import Counter


class VQESampler:
    """VQE采样器，用于从优化后的量子态中采样比特串"""
    
    def __init__(self, vqe, device: torch.device = None):
        """
        初始化VQE采样器
        
        Args:
            vqe: VQE实例，包含优化后的PQC
            device: 计算设备
        """
        self.vqe = vqe
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pqc = vqe.pqc
        self.num_qubits = self.pqc.num_qubits
        
    def get_final_state(self, input_state: torch.Tensor = None) -> torch.Tensor:
        """
        获取VQE优化后的最终量子态
        
        Args:
            input_state: 输入量子态，如果为None则使用|0⟩态
            
        Returns:
            torch.Tensor: 最终量子态向量
        """
        with torch.no_grad():
            final_state = self.vqe.get_ground_state(input_state)
        return final_state
    
    def sample_bitstrings(self, num_shots: int = 1000, 
                         input_state: torch.Tensor = None,
                         qubit_indices: Optional[List[int]] = None) -> List[str]:
        """
        从最终量子态中采样比特串
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态，如果为None则使用|0⟩态
            qubit_indices: 要采样的量子比特索引，如果为None则采样所有量子比特
            
        Returns:
            List[str]: 采样得到的比特串列表
        """
        # 获取最终量子态
        final_state = self.get_final_state(input_state)
        
        # 确保是1D向量
        if final_state.ndim > 1:
            final_state = final_state.squeeze()
        
        # 计算测量概率
        probs = torch.abs(final_state) ** 2
        probs = probs / torch.sum(probs)  # 归一化
        
        # 采样测量结果
        sampled_indices = torch.multinomial(probs, num_shots, replacement=True)
        
        # 转换为比特串
        bitstrings = []
        for idx in sampled_indices:
            bitstring = format(idx.item(), f'0{self.num_qubits}b')
            if qubit_indices is not None:
                # 只保留指定量子比特的结果
                bitstring = ''.join([bitstring[i] for i in qubit_indices])
            bitstrings.append(bitstring)
        
        return bitstrings
    
    def get_bitstring_statistics(self, num_shots: int = 1000,
                                input_state: torch.Tensor = None,
                                qubit_indices: Optional[List[int]] = None) -> Dict[str, float]:
        """
        获取比特串的统计信息
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态
            qubit_indices: 要采样的量子比特索引
            
        Returns:
            Dict[str, float]: 比特串及其出现频率
        """
        bitstrings = self.sample_bitstrings(num_shots, input_state, qubit_indices)
        counter = Counter(bitstrings)
        
        # 转换为频率字典
        total = len(bitstrings)
        frequencies = {bitstring: count / total for bitstring, count in counter.items()}
        
        return frequencies
    
    def get_energy_distribution(self, num_shots: int = 1000,
                               input_state: torch.Tensor = None) -> Dict[str, float]:
        """
        获取能量分布（基于比特串对应的能量本征值）
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态
            
        Returns:
            Dict[str, float]: 能量值及其出现频率
        """
        bitstrings = self.sample_bitstrings(num_shots, input_state)
        
        # 计算每个比特串对应的能量
        energy_distribution = {}
        for bitstring in bitstrings:
            # 将比特串转换为计算基态索引
            state_idx = int(bitstring, 2)
            
            # 计算该态的能量（这里使用简化的方法）
            # 对于海森堡模型，可以基于比特串中相邻位的不同来计算能量
            energy = self._calculate_heisenberg_energy(bitstring)
            
            if energy not in energy_distribution:
                energy_distribution[energy] = 0
            energy_distribution[energy] += 1
        
        # 转换为频率
        total = len(bitstrings)
        frequencies = {energy: count / total for energy, count in energy_distribution.items()}
        
        return frequencies
    
    def _calculate_heisenberg_energy(self, bitstring: str) -> float:
        """
        计算海森堡模型中给定比特串的能量
        
        Args:
            bitstring: 比特串
            
        Returns:
            float: 能量值
        """
        energy = 0.0
        J = 1.0  # 相互作用强度
        h = 0.0  # 外场强度
        
        # 相互作用项：J * (σx⊗σx + σy⊗σy + σz⊗σz)
        for i in range(len(bitstring) - 1):
            bit_i = int(bitstring[i])
            bit_j = int(bitstring[i + 1])
            
            # σz⊗σz 项（对角项）
            if bit_i == bit_j:
                energy -= J
            else:
                energy += J
        
        # 外场项：h * σz
        for i in range(len(bitstring)):
            bit_i = int(bitstring[i])
            if bit_i == 0:
                energy += h
            else:
                energy -= h
        
        return energy
    
    def plot_bitstring_distribution(self, num_shots: int = 1000,
                                   input_state: torch.Tensor = None,
                                   qubit_indices: Optional[List[int]] = None,
                                   top_k: int = 10,
                                   save_path: Optional[str] = None):
        """
        绘制比特串分布图
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态
            qubit_indices: 要采样的量子比特索引
            top_k: 显示前k个最常见的比特串
            save_path: 保存路径
        """
        frequencies = self.get_bitstring_statistics(num_shots, input_state, qubit_indices)
        
        # 按频率排序，取前top_k个
        sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
        bitstrings, freqs = zip(*sorted_items)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(bitstrings)), freqs)
        plt.xlabel('Bitstring')
        plt.ylabel('Frequency')
        plt.title(f'Top {top_k} Bitstring Distribution (VQE Sampled)')
        plt.xticks(range(len(bitstrings)), bitstrings, rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, freq in zip(bars, freqs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{freq:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_energy_distribution(self, num_shots: int = 1000,
                                input_state: torch.Tensor = None,
                                save_path: Optional[str] = None):
        """
        绘制能量分布图
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态
            save_path: 保存路径
        """
        energy_freqs = self.get_energy_distribution(num_shots, input_state)
        
        energies = list(energy_freqs.keys())
        frequencies = list(energy_freqs.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(energies, frequencies)
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title('Energy Distribution (VQE Sampled)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy distribution plot saved to {save_path}")
        
        plt.show()
    
    def get_entanglement_entropy(self, input_state: torch.Tensor = None,
                                partition: Optional[List[int]] = None) -> float:
        """
        计算量子态的纠缠熵
        
        Args:
            input_state: 输入量子态
            partition: 子系统A的量子比特索引，如果为None则取前半部分
            
        Returns:
            float: 纠缠熵
        """
        try:
            final_state = self.get_final_state(input_state)
            
            if final_state.ndim > 1:
                final_state = final_state.squeeze()
            
            # 确保态向量是归一化的
            norm = torch.norm(final_state)
            if norm > 0:
                final_state = final_state / norm
            
            if partition is None:
                # 默认取前半部分量子比特
                partition = list(range(self.num_qubits // 2))
            
            # 计算约化密度矩阵
            dim_A = 2 ** len(partition)
            dim_B = 2 ** (self.num_qubits - len(partition))
            
            # 重塑态向量为矩阵形式
            state_matrix = final_state.reshape(dim_A, dim_B)
            
            # 计算约化密度矩阵
            rho_A = torch.matmul(state_matrix, state_matrix.conj().T)
            
            # 确保密度矩阵是厄米的
            rho_A = (rho_A + rho_A.conj().T) / 2
            
            # 添加小的对角项以确保正定性
            rho_A = rho_A + torch.eye(dim_A, dtype=torch.complex64, device=rho_A.device) * 1e-12
            
            # 重新归一化
            trace = torch.trace(rho_A).real
            if trace > 0:
                rho_A = rho_A / trace
            
            # 计算特征值
            eigenvalues = torch.linalg.eigvalsh(rho_A)
            
            # 确保特征值非负且和为1
            eigenvalues = torch.clamp(eigenvalues, min=1e-12)
            eigenvalues = eigenvalues / torch.sum(eigenvalues)
            
            # 计算von Neumann熵
            entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-12))
            
            return entropy.item()
        except Exception as e:
            print(f"计算纠缠熵时出错: {e}")
            return 0.0
    
    def save_sampling_results(self, num_shots: int = 1000,
                             input_state: torch.Tensor = None,
                             save_path: str = "vqe_sampling_results.json"):
        """
        保存采样结果到文件
        
        Args:
            num_shots: 采样次数
            input_state: 输入量子态
            save_path: 保存路径
        """
        import json
        
        # 获取各种统计信息
        bitstring_freqs = self.get_bitstring_statistics(num_shots, input_state)
        energy_freqs = self.get_energy_distribution(num_shots, input_state)
        entanglement_entropy = self.get_entanglement_entropy(input_state)
        
        # 获取最终能量
        with torch.no_grad():
            final_energy = self.vqe.forward(input_state).item()
        
        results = {
            'num_qubits': self.num_qubits,
            'num_shots': num_shots,
            'final_energy': final_energy,
            'entanglement_entropy': entanglement_entropy,
            'bitstring_frequencies': bitstring_freqs,
            'energy_distribution': energy_freqs,
            'top_bitstrings': sorted(bitstring_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Sampling results saved to {save_path}")
        return results


def create_vqe_sampler_from_results(vqe_results_path: str, vqe_instance) -> VQESampler:
    """
    从VQE结果文件创建采样器
    
    Args:
        vqe_results_path: VQE结果文件路径
        vqe_instance: VQE实例
        
    Returns:
        VQESampler: VQE采样器实例
    """
    import json
    
    with open(vqe_results_path, 'r') as f:
        results = json.load(f)
    
    # 设置VQE的最终参数
    if 'final_parameters' in results:
        final_params = torch.tensor(results['final_parameters'])
        vqe_instance.pqc.set_parameters(final_params)
    
    return VQESampler(vqe_instance) 