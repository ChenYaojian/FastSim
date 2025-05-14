import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from typing import Dict, List, Union, Callable
from abc import ABC, abstractmethod
import math


class QuantumGate(ABC):
    """量子门基类"""
    _registry: Dict[str, 'QuantumGate'] = {}
    
    def __init__(self, name: str, num_qubits: int, is_parametric: bool = False):
        self.name = name
        self.num_qubits = num_qubits
        self.is_parametric = is_parametric
        self.matrix = None
        self.params = None
        self.matrix_func = None

    @classmethod
    def register(cls, name: str, num_qubits: int, matrix_func: Callable = None, is_parametric: bool = False):
        """注册量子门"""
        def decorator(gate_class):
            cls._registry[name] = gate_class
            return gate_class
        return decorator

    @classmethod
    def get_gate(cls, name: str, **kwargs) -> 'QuantumGate':
        """获取已注册的量子门实例"""
        if name not in cls._registry:
            raise ValueError(f"Gate {name} not registered")
        return cls._registry[name](**kwargs)

    @abstractmethod
    def get_matrix(self, params: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """获取门的矩阵表示"""
        return self.matrix

class ParametricGate(QuantumGate):
    """参数化量子门"""
    def __init__(self, name: str, num_qubits: int, matrix_func_str: str = None, is_parametric: bool = False):
        super().__init__(name, num_qubits, is_parametric=is_parametric)
        if matrix_func_str:
            # 将字符串形式的函数转换为可执行的函数
            self.matrix_func = self._parse_matrix_func(matrix_func_str)
        else:
            self.matrix_func = None

    def _parse_matrix_func(self, func_str: str) -> Callable:
        """解析字符串形式的矩阵函数"""
        # 创建一个包含必要数学函数的命名空间
        namespace = {
            'cos': torch.cos,
            'sin': torch.sin,
            'exp': torch.exp,
            'pi': math.pi,
            'j': 1j,
            'torch': torch,
            'np': np
        }
        
        # 将字符串转换为可执行的函数
        try:
            # 使用eval安全地执行字符串
            func = eval(f"lambda theta: {func_str}", namespace)
            return func
        except Exception as e:
            raise ValueError(f"Error parsing matrix function: {e}")

    def get_matrix(self, params: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """获取门的矩阵表示"""
        if self.is_parametric:
            if params is None:
                raise ValueError(f"Gate {self.name} requires parameters")
            if self.matrix_func:
                # 使用解析后的函数计算矩阵
                matrix = self.matrix_func(params)
                # 确保返回的是torch.Tensor类型
                if not isinstance(matrix, torch.Tensor):
                    matrix = torch.tensor(matrix, dtype=torch.complex64)
                return matrix
            else:
                raise ValueError(f"No matrix function defined for gate {self.name}")
        else:
            if self.matrix is None:
                raise ValueError(f"No matrix defined for gate {self.name}")
            return self.matrix


def _parse_complex_matrix(matrix):
    """解析包含复数的矩阵"""
    def parse_complex(x):
        if isinstance(x, str):
            return eval(x.replace('j', '1j'))
        return x
    return [[parse_complex(x) for x in row] for row in matrix]


# 从配置文件加载门定义
def load_gates_from_config(config_path: str):
    """从配置文件加载门定义"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    for gate_info in config['gates']:
        name = gate_info['name']
        num_qubits = gate_info['num_qubits']
        is_parametric = gate_info.get('is_parametric', False)
        
        # 创建门类
        if is_parametric:
            matrix_func = gate_info.get('matrix_func')
            if not matrix_func:
                raise ValueError(f"Parametric gate {name} must have matrix_func defined")
            
            # 动态创建参数化门类
            def create_parametric_gate_class(gate_name, n_qubits, m_func):
                class DynamicGate(ParametricGate):
                    def __init__(self, **kwargs):
                        super().__init__(gate_name, n_qubits, matrix_func_str=m_func, is_parametric=True)
                    
                    def get_matrix(self, params=None):
                        if params is None:
                            raise ValueError(f"Gate {gate_name} requires parameters")
                        if self.matrix_func:
                            matrix = self.matrix_func(params)
                            if not isinstance(matrix, torch.Tensor):
                                matrix = torch.tensor(matrix, dtype=torch.complex64)
                            return matrix
                        raise ValueError(f"No matrix function defined for gate {gate_name}")
                return DynamicGate
            
            gate_class = create_parametric_gate_class(name, num_qubits, matrix_func)
            
        else:
            matrix = gate_info.get('matrix')
            if not matrix:
                raise ValueError(f"Non-parametric gate {name} must have matrix defined")
            
            # 解析矩阵中的复数
            matrix = _parse_complex_matrix(matrix)
            
            # 动态创建非参数化门类
            def create_non_parametric_gate_class(gate_name, n_qubits, m):
                class DynamicGate(QuantumGate):
                    def __init__(self, **kwargs):
                        super().__init__(gate_name, n_qubits, is_parametric=False)
                        self.matrix = torch.tensor(m, dtype=torch.complex64)
                    
                    def get_matrix(self, params=None):
                        return self.matrix
                return DynamicGate
            
            gate_class = create_non_parametric_gate_class(name, num_qubits, matrix)
        
        # 注册门类
        QuantumGate.register(name, num_qubits, is_parametric=is_parametric)(gate_class)


class Circuit(nn.Module):
    def __init__(self, num_qubits: int):
        super(Circuit, self).__init__()
        self.num_qubits = num_qubits
        self.gates = []  # 存储门操作序列
        self.qubits = torch.zeros((2**num_qubits,), dtype=torch.complex64)
        self.qubits[0] = 1.0  # 初始化为|0⟩态

    def add_gate(self, gate_name: str, qubit_indices: List[int], params: Union[torch.Tensor, None] = None):
        """添加门操作到电路
        
        Args:
            gate_name: 门的名称
            qubit_indices: 门操作作用的量子比特索引
            params: 参数化门的参数（如果有）
        """
        gate = QuantumGate.get_gate(gate_name)
        if gate.is_parametric and params is None:
            raise ValueError(f"Gate {gate_name} requires parameters")
        self.gates.append((gate, qubit_indices, params))

    def forward(self):
        """执行量子电路"""
        state = self.qubits
        for gate, qubit_indices, params in self.gates:
            state = self._apply_gate(state, gate, qubit_indices, params)
        return state

    def _apply_gate(self, state: torch.Tensor, gate: QuantumGate, 
                   qubit_indices: List[int], params: Union[torch.Tensor, None]) -> torch.Tensor:
        """应用门操作到量子态"""
        # 获取门的矩阵表示
        matrix = gate.get_matrix(params)
        
        # 这里需要实现具体的门操作应用逻辑
        # 可以使用张量积和矩阵乘法来实现
        return state  # 临时返回，需要实现具体逻辑

    def draw(self, max_gates: int = 10, show_params: bool = True) -> str:
        """绘制量子电路图
        
        Args:
            max_gates: 显示的最大门数量，超过此数量将使用省略号
            show_params: 是否显示参数化门的参数值
        
        Returns:
            str: 电路图的字符串表示
        """
        if not self.gates:
            return "Empty circuit"

        # 计算需要显示的门
        if len(self.gates) > max_gates:
            first_half = self.gates[:max_gates//2]
            second_half = self.gates[-(max_gates//2):]
            gates_to_show = first_half + [None] + second_half
        else:
            gates_to_show = self.gates

        # 创建电路图的行
        lines = []
        for qubit in range(self.num_qubits):
            line = []
            for gate_info in gates_to_show:
                if gate_info is None:
                    line.append("...")
                else:
                    gate, qubit_indices, params = gate_info
                    if qubit in qubit_indices:
                        # 获取门的表示
                        gate_str = gate.name
                        if gate.is_parametric and show_params and params is not None:
                            if isinstance(params, torch.Tensor):
                                param_value = params.item()
                            else:
                                param_value = params
                            gate_str += f"({param_value:.2f})"
                        
                        # 对于多量子比特门，添加连接线
                        if len(qubit_indices) > 1:
                            if qubit == min(qubit_indices):
                                line.append("┌─" + gate_str + "─┐")
                            elif qubit == max(qubit_indices):
                                line.append("└─" + "─" * len(gate_str) + "─┘")
                            else:
                                line.append("│ " + " " * len(gate_str) + " │")
                        else:
                            line.append("─" + gate_str + "─")
                    else:
                        # 对于不在门操作中的量子比特，添加空线
                        if len(qubit_indices) > 1 and min(qubit_indices) < qubit < max(qubit_indices):
                            line.append("│ " + " " * len(gate.name) + " │")
                        else:
                            line.append("─" + "─" * len(gate.name) + "─")
            lines.append("".join(line))

        # 添加量子比特标签
        labeled_lines = []
        for i, line in enumerate(lines):
            labeled_lines.append(f"q{i}: {line}")

        return "\n".join(labeled_lines)
        
        
        