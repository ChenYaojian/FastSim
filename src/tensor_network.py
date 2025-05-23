import torch
from typing import List, Dict, Tuple, Optional, Union
import cotengra as ctg
import numpy as np
from src.tensor_ops import TensorOps

class TensorNetwork:
    """张量网络类
    
    管理张量网络的结构和操作，包括：
    1. 存储张量和它们的索引
    2. 寻找最优收缩路径
    3. 执行张量网络收缩
    4. 支持变分张量网络
    
    属性：
        arrays: 所有张量构成的列表
        indices: 每个张量的indices构成的列表
        size_dict: 每个index的维度大小
        path: 收缩路径
        output: 收缩完毕后剩余的indices
        variational: 是否为变分张量网络
    """
    
    def __init__(self, arrays: List[torch.Tensor], indices: List[List[str]], 
                 size_dict: Dict[str, int], output: List[str], variational: bool = False):
        """初始化张量网络
        
        Args:
            arrays: 所有张量构成的列表
            indices: 每个张量的indices构成的列表
            size_dict: 每个index的维度大小
            output: 收缩完毕后剩余的indices
            variational: 是否为变分张量网络
        """
        self.arrays = arrays
        self.indices = indices
        self.size_dict = size_dict
        self.output = output
        self.variational = variational
        self.path = None
        
        # 验证输入
        self._validate_input()
        
        # 寻找收缩路径
        self._find_contraction_path()
    
    def _validate_input(self) -> None:
        """验证输入的有效性
        
        Raises:
            ValueError: 如果输入无效
        """
        # 检查数组和索引列表长度是否匹配
        if len(self.arrays) != len(self.indices):
            raise ValueError(f"Number of arrays ({len(self.arrays)}) does not match "
                           f"number of index lists ({len(self.indices)})")
        
        # 检查每个张量的维度是否与其索引列表匹配
        for i, (array, idx_list) in enumerate(zip(self.arrays, self.indices)):
            if len(array.shape) != len(idx_list):
                raise ValueError(f"Array {i} has {len(array.shape)} dimensions but "
                               f"{len(idx_list)} indices")
        
        # 检查每个索引的维度是否在size_dict中定义
        for idx_list in self.indices:
            for idx in idx_list:
                if idx not in self.size_dict:
                    raise ValueError(f"Index {idx} not found in size_dict")
        
        # 检查输出索引是否都在张量网络中出现
        for idx in self.output:
            if not any(idx in idx_list for idx_list in self.indices):
                raise ValueError(f"Output index {idx} not found in tensor network")
    
    def _find_contraction_path(self) -> None:
        """寻找最优收缩路径
        
        使用cotengra包来寻找最优收缩路径。对于非变分网络，
        可以使用full_simplify来简化网络。
        """
        # 创建cotengra的输入格式
        inputs = []
        for idx_list in self.indices:
            # 将索引列表转换为cotengra的格式
            input_indices = []
            for idx in idx_list:
                input_indices.append((idx, self.size_dict[idx]))
            inputs.append(input_indices)
        
        # 创建cotengra的优化器
        optimizer = ctg.HyperOptimizer(
            methods=['greedy', 'kahypar'],
            max_repeats=16,
            max_time=30,
            progbar=True
        )
        
        if not self.variational:
            # 对于非变分网络，使用full_simplify
            path = optimizer.full_simplify(
                inputs=inputs,
                output=self.output,
                size_dict=self.size_dict
            )
        else:
            # 对于变分网络，只寻找收缩路径
            path = optimizer.optimize(
                inputs=inputs,
                output=self.output,
                size_dict=self.size_dict
            )
        
        self.path = path
    
    def contract(self) -> torch.Tensor:
        """执行张量网络收缩
        
        Returns:
            torch.Tensor: 收缩后的张量
            
        Raises:
            RuntimeError: 如果收缩路径未找到
        """
        if self.path is None:
            raise RuntimeError("Contraction path not found")
        
        # 创建张量副本，避免修改原始数据
        arrays = [array.clone() for array in self.arrays]
        
        # 按照路径执行收缩
        for i, j in self.path:
            # 获取要收缩的两个张量
            tensor1 = arrays[i]
            tensor2 = arrays[j]
            
            # 获取它们的索引
            indices1 = self.indices[i]
            indices2 = self.indices[j]
            
            # 找到共同的索引
            common_indices = set(indices1).intersection(set(indices2))
            
            # 找到非共同的索引
            non_common1 = [idx for idx in indices1 if idx not in common_indices]
            non_common2 = [idx for idx in indices2 if idx not in common_indices]
            
            # 执行收缩
            result = TensorOps.contract(
                tensor1, tensor2,
                [indices1.index(idx) for idx in common_indices],
                [indices2.index(idx) for idx in common_indices]
            )
            
            # 更新张量列表
            arrays[i] = result
            arrays.pop(j)
            
            # 更新索引列表
            self.indices[i] = non_common1 + non_common2
            self.indices.pop(j)
        
        # 返回最终结果
        return arrays[0]
    
    def get_contraction_cost(self) -> int:
        """计算收缩的计算复杂度
        
        Returns:
            int: 收缩的计算复杂度（浮点运算次数）
        """
        if self.path is None:
            raise RuntimeError("Contraction path not found")
        
        # 使用cotengra计算复杂度
        return self.path.contraction_cost()
    
    def get_contraction_tree(self) -> str:
        """获取收缩树的字符串表示
        
        Returns:
            str: 收缩树的字符串表示
        """
        if self.path is None:
            raise RuntimeError("Contraction path not found")
        
        return str(self.path)
    
    def simplify(self) -> None:
        """简化张量网络（仅用于非变分网络）
        
        Raises:
            RuntimeError: 如果是变分网络
        """
        if self.variational:
            raise RuntimeError("Cannot simplify variational tensor network")
        
        # 使用cotengra的full_simplify
        optimizer = ctg.HyperOptimizer(
            methods=['greedy', 'kahypar'],
            max_repeats=16,
            max_time=30,
            progbar=True
        )
        
        # 创建cotengra的输入格式
        inputs = []
        for idx_list in self.indices:
            input_indices = []
            for idx in idx_list:
                input_indices.append((idx, self.size_dict[idx]))
            inputs.append(input_indices)
        
        # 简化网络
        path = optimizer.full_simplify(
            inputs=inputs,
            output=self.output,
            size_dict=self.size_dict
        )
        
        self.path = path
    
    def to(self, device: torch.device) -> 'TensorNetwork':
        """将张量网络移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            TensorNetwork: 移动后的张量网络
        """
        self.arrays = [array.to(device) for array in self.arrays]
        return self
    
    def clone(self) -> 'TensorNetwork':
        """创建张量网络的副本
        
        Returns:
            TensorNetwork: 张量网络的副本
        """
        return TensorNetwork(
            arrays=[array.clone() for array in self.arrays],
            indices=[idx_list.copy() for idx_list in self.indices],
            size_dict=self.size_dict.copy(),
            output=self.output.copy(),
            variational=self.variational
        )
