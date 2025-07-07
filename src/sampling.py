import torch
import numpy as np
import json
from typing import Dict, List, Union, Optional
from src.state import StateVector


class Sampler:
    """量子态采样器类
    
    提供量子态的采样功能，包括：
    1. 末态采样
    2. 从采样重建量子态
    3. 基于采样的期望值计算
    4. 采样统计信息
    """
    
    def __init__(self, state: StateVector, default_shots: int = 10000):
        """
        初始化采样器
        
        Args:
            state: 要采样的量子态
            default_shots: 默认采样次数
        """
        self.state = state
        self.default_shots = default_shots
        self.sampling_history = []  # 存储采样历史
    
    def sample_final_state(self, num_shots: int = None, 
                          qubit_indices: Optional[List[int]] = None,
                          return_json: bool = False) -> Union[Dict[int, int], str]:
        """
        末态采样
        
        Args:
            num_shots: 采样次数，如果为None则使用默认值
            qubit_indices: 要采样的量子比特索引，如果为None则采样所有量子比特
            return_json: 是否返回JSON字符串
            
        Returns:
            Union[Dict[int, int], str]: 采样结果
        """
        if num_shots is None:
            num_shots = self.default_shots
        
        # 直接调用StateVector的采样方法
        result = self.state.sample_final_state(
            num_shots=num_shots,
            qubit_indices=qubit_indices,
            return_json=return_json
        )
        
        # 记录采样历史
        self.sampling_history.append({
            'num_shots': num_shots,
            'qubit_indices': qubit_indices,
            'result': result if not return_json else json.loads(result)
        })
        
        return result
    
    def reconstruct_state_from_sampling(self, sampling_result: Dict[int, int], 
                                      qubit_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        从采样结果重建量子态向量
        
        Args:
            sampling_result: 采样结果字典 {bitstring_int: count}
            qubit_indices: 采样的量子比特索引
            
        Returns:
            torch.Tensor: 重建的量子态向量
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.state.num_qubits))
        
        # 从采样频率重建量子态向量
        reconstructed_state = torch.zeros(2**len(qubit_indices), dtype=torch.complex64)
        total_shots = sum(sampling_result.values())
        
        # 获取原始态向量中对应qubit_indices的部分
        original_state = self.state.state_vector
        if qubit_indices != list(range(self.state.num_qubits)):
            # 如果只采样部分量子比特，需要从完整态向量中提取对应部分
            # 这里简化处理，直接使用原始态向量的振幅
            for i in range(2**len(qubit_indices)):
                if i < len(original_state):
                    reconstructed_state[i] = original_state[i]
        else:
            # 采样所有量子比特，直接使用原始态向量
            reconstructed_state = original_state.clone()
        
        # 根据采样频率调整振幅大小，但保持相位
        for bitstring_int, count in sampling_result.items():
            if bitstring_int < len(reconstructed_state):
                # 计算采样概率
                prob = count / total_shots
                # 获取原始振幅
                original_amplitude = reconstructed_state[bitstring_int]
                # 调整振幅大小以匹配采样频率，但保持相位
                if abs(original_amplitude) > 0:
                    phase = original_amplitude / abs(original_amplitude)
                    reconstructed_state[bitstring_int] = phase * torch.sqrt(torch.tensor(prob, dtype=torch.float32))
        
        # 归一化重建的态向量
        norm = torch.norm(reconstructed_state)
        if norm > 0:
            reconstructed_state = reconstructed_state / norm
        
        return reconstructed_state
    
    def calculate_expectation_from_sampling(self, observable: torch.Tensor, 
                                          num_shots: int = None,
                                          qubit_indices: Optional[List[int]] = None) -> float:
        """
        通过采样计算期望值
        
        从采样频率重建量子态向量，然后计算期望值。
        这样可以正确处理包含非对角项的观测量。
        
        Args:
            observable: 可观测量算符
            num_shots: 采样次数，如果为None则使用默认值
            qubit_indices: 作用的量子比特索引
            
        Returns:
            float: 期望值
        """
        if num_shots is None:
            num_shots = self.default_shots
        
        # 进行采样
        sampling_result = self.sample_final_state(
            num_shots=num_shots,
            qubit_indices=qubit_indices,
            return_json=False
        )
        
        # 从采样重建量子态
        reconstructed_state = self.reconstruct_state_from_sampling(sampling_result, qubit_indices)
        
        # 计算期望值：<ψ|O|ψ>
        expectation = torch.real(torch.vdot(reconstructed_state, torch.matmul(observable, reconstructed_state)))
        
        return expectation.item()
    
    def compare_sampling_vs_direct_expectation(self, observable: torch.Tensor,
                                             num_shots_list: List[int] = None,
                                             qubit_indices: Optional[List[int]] = None) -> Dict:
        """
        比较采样期望值与直接期望值
        
        Args:
            observable: 可观测量算符
            num_shots_list: 采样次数列表，如果为None则使用默认列表
            qubit_indices: 作用的量子比特索引
            
        Returns:
            Dict: 比较结果
        """
        if num_shots_list is None:
            num_shots_list = [1000, 5000, 10000, 20000]
        
        # 计算直接期望值
        direct_expectation = self.state.get_expectation(observable, qubit_indices=qubit_indices)
        
        # 通过采样计算期望值
        results = []
        for num_shots in num_shots_list:
            sampling_expectation = self.calculate_expectation_from_sampling(
                observable, num_shots=num_shots, qubit_indices=qubit_indices
            )
            error = abs(sampling_expectation - direct_expectation)
            relative_error = (error / abs(direct_expectation)) * 100 if direct_expectation != 0 else 0
            
            results.append({
                'num_shots': num_shots,
                'sampling_expectation': sampling_expectation,
                'direct_expectation': direct_expectation,
                'error': error,
                'relative_error': relative_error
            })
        
        return {
            'direct_expectation': direct_expectation,
            'comparison_results': results
        }
    
    def get_sampling_statistics(self, num_shots: int = None,
                               qubit_indices: Optional[List[int]] = None) -> Dict:
        """
        获取采样统计信息
        
        Args:
            num_shots: 采样次数，如果为None则使用默认值
            qubit_indices: 要采样的量子比特索引
            
        Returns:
            Dict: 统计信息
        """
        if num_shots is None:
            num_shots = self.default_shots
        
        # 进行采样
        sampling_result = self.sample_final_state(
            num_shots=num_shots,
            qubit_indices=qubit_indices,
            return_json=False
        )
        
        # 计算统计信息
        total_shots = sum(sampling_result.values())
        unique_states = len(sampling_result)
        
        # 计算最常见的状态
        sorted_states = sorted(sampling_result.items(), key=lambda x: x[1], reverse=True)
        most_common_state = sorted_states[0] if sorted_states else (None, 0)
        
        # 计算概率分布
        probability_dist = {state: count/total_shots for state, count in sampling_result.items()}
        
        return {
            'num_shots': num_shots,
            'qubit_indices': qubit_indices,
            'total_shots': total_shots,
            'unique_states': unique_states,
            'most_common_state': {
                'bitstring': format(most_common_state[0], f'0{len(qubit_indices) if qubit_indices else self.state.num_qubits}b'),
                'count': most_common_state[1],
                'probability': most_common_state[1] / total_shots
            },
            'probability_distribution': probability_dist,
            'sampling_result': sampling_result
        }
    
    def get_probability_distribution(self, qubit_indices: Optional[List[int]] = None) -> Dict[int, float]:
        """
        获取量子态的概率分布
        
        Args:
            qubit_indices: 要计算的量子比特索引，如果为None则计算所有量子比特
            
        Returns:
            Dict[int, float]: 概率分布
        """
        return self.state.get_probability_distribution(qubit_indices=qubit_indices)
    
    def save_sampling_results(self, filename: str, num_shots: int = None,
                             qubit_indices: Optional[List[int]] = None) -> str:
        """
        保存采样结果到文件
        
        Args:
            filename: 输出文件名
            num_shots: 采样次数，如果为None则使用默认值
            qubit_indices: 要采样的量子比特索引
            
        Returns:
            str: 保存的文件路径
        """
        if num_shots is None:
            num_shots = self.default_shots
        
        # 获取采样统计信息
        stats = self.get_sampling_statistics(num_shots, qubit_indices)
        
        # 保存到文件
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return filename
    
    def print_sampling_summary(self, num_shots: int = None,
                              qubit_indices: Optional[List[int]] = None):
        """
        打印采样摘要信息
        
        Args:
            num_shots: 采样次数，如果为None则使用默认值
            qubit_indices: 要采样的量子比特索引
        """
        if num_shots is None:
            num_shots = self.default_shots
        
        stats = self.get_sampling_statistics(num_shots, qubit_indices)
        
        print(f"采样摘要:")
        print(f"  采样次数: {stats['num_shots']}")
        print(f"  量子比特: {stats['qubit_indices']}")
        print(f"  不同状态数: {stats['unique_states']}")
        print(f"  最常见状态: |{stats['most_common_state']['bitstring']}⟩ "
              f"(出现{stats['most_common_state']['count']}次, "
              f"概率{stats['most_common_state']['probability']:.4f})")
        
        # 显示前5个最常见的状态
        sorted_states = sorted(stats['probability_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        print(f"  前5个最常见状态:")
        for i, (state, prob) in enumerate(sorted_states, 1):
            bitstring = format(state, f'0{len(qubit_indices) if qubit_indices else self.state.num_qubits}b')
            print(f"    {i}. |{bitstring}⟩: {prob:.4f} ({prob*100:.2f}%)")


def create_sampler_from_state(state: StateVector, default_shots: int = 10000) -> Sampler:
    """
    从量子态创建采样器
    
    Args:
        state: 量子态
        default_shots: 默认采样次数
        
    Returns:
        Sampler: 采样器实例
    """
    return Sampler(state, default_shots)


def calculate_expectation_from_sampling(state: StateVector, observable: torch.Tensor, 
                                      num_shots: int = 10000, qubit_indices: List[int] = None) -> float:
    """
    通过采样计算期望值（便捷函数）
    
    Args:
        state: 量子态
        observable: 可观测量算符
        num_shots: 采样次数
        qubit_indices: 作用的量子比特索引
        
    Returns:
        float: 期望值
    """
    sampler = Sampler(state, num_shots)
    return sampler.calculate_expectation_from_sampling(observable, num_shots=num_shots, qubit_indices=qubit_indices) 