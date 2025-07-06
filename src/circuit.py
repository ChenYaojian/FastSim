import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from typing import Dict, List, Union, Callable
from abc import ABC, abstractmethod
import math
import inspect


class QuantumGate(ABC, nn.Module):
    """量子门基类"""
    _registry: Dict[str, 'QuantumGate'] = {}
    
    def __init__(self, name: str, num_qubits: int, is_parametric: bool = False):
        ABC.__init__(self)
        nn.Module.__init__(self)
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

    def forward(self, state: torch.Tensor, qubit_indices: List[int]) -> torch.Tensor:
        """将门作为神经网络层应用
        
        Args:
            state: 输入量子态，形状为 [batch_size, 2**num_qubits]
            qubit_indices: 门操作作用的量子比特索引
            
        Returns:
            torch.Tensor: 输出量子态，形状为 [batch_size, 2**num_qubits]
        """
        batch_size = state.size(0)
        total_qubits = int(math.log2(state.size(1)))  # 计算总量子比特数
        
        # 获取门的矩阵表示
        matrix = self.get_matrix()
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.complex64, device=state.device)
        elif matrix.device != state.device:
            matrix = matrix.to(state.device)
            
        gate_dim = 2**len(qubit_indices)
        
        # 对每个样本分别处理
        new_states = []
        for i in range(batch_size):
            # 将单个样本的state重塑为nqubit维的张量
            single_state = state[i].view([2] * total_qubits)
            
            # 根据gate作用的qubits，将state vector permute，然后reshape
            permute_order = list(range(total_qubits))
            for j, qubit in enumerate(qubit_indices):
                permute_order[j], permute_order[qubit] = permute_order[qubit], permute_order[j]
            
            # 计算逆置换
            inverse_permute_order = [0] * total_qubits
            for j, pos in enumerate(permute_order):
                inverse_permute_order[pos] = j
            
            # 重塑state为矩阵形式
            reshaped_state = single_state.permute(permute_order).reshape(gate_dim, -1)
            
            # 执行矩阵乘法
            new_state = torch.matmul(matrix, reshaped_state)
            
            # 将state permute回去（使用逆置换）
            new_state = new_state.reshape([2] * total_qubits).permute(inverse_permute_order)
            new_states.append(new_state.reshape(-1))
        
        # 将处理后的状态堆叠成batch
        return torch.stack(new_states)

    @abstractmethod
    def get_matrix(self, params: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """获取门的矩阵表示"""
        return self.matrix

class ParametricGate(QuantumGate):
    """参数化量子门，支持多参数"""
    def __init__(self, name: str, num_qubits: int, matrix_func_str: str = None, is_parametric: bool = False, param_names: list = None):
        super().__init__(name, num_qubits, is_parametric=is_parametric)
        self.param_names = param_names or []
        if matrix_func_str:
            self.matrix_func, self.param_names = self._parse_matrix_func(matrix_func_str, self.param_names)
            # 为每个参数创建Parameter
            for param_name in self.param_names:
                param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
                setattr(self, param_name, param)
        else:
            self.matrix_func = None

    def _parse_matrix_func(self, func_str: str, param_names_from_json=None):
        """解析字符串形式的矩阵函数，支持lambda表达式和多参数"""
        namespace = {
            'cos': torch.cos,
            'sin': torch.sin,
            'exp': torch.exp,
            'pi': math.pi,
            'j': 1j,
            'torch': torch,
            'np': np
        }
        try:
            func_str = func_str.strip()
            if func_str.startswith('lambda'):
                # 直接eval lambda表达式
                matrix_func = eval(func_str, namespace)
                # 自动提取参数名
                sig = inspect.signature(matrix_func)
                param_names = list(sig.parameters.keys())
                return matrix_func, param_names
            else:
                # 兼容旧格式，假设单参数theta
                def matrix_func(theta):
                    return eval(func_str, {**namespace, 'theta': theta})
                param_names = param_names_from_json or ['theta']
                return matrix_func, param_names
        except Exception as e:
            raise ValueError(f"Error parsing matrix function: {e}")

    def get_matrix(self, params: Union[torch.Tensor, list, tuple, dict, None] = None) -> torch.Tensor:
        """获取门的矩阵表示，支持多参数"""
        if self.is_parametric:
            if params is None:
                # 如果没有提供参数，使用内部Parameter
                params = [getattr(self, name) for name in self.param_names]
            if self.matrix_func:
                try:
                    # 参数预处理：确保参数是torch.Tensor且保持梯度
                    def to_tensor(x):
                        if isinstance(x, torch.Tensor):
                            return x
                        else:
                            return torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
                    
                    if isinstance(params, dict):
                        args = [to_tensor(params[k]) for k in self.param_names]
                        matrix = self.matrix_func(*args)
                    elif isinstance(params, (list, tuple)):
                        args = [to_tensor(x) for x in params]
                        matrix = self.matrix_func(*args)
                    elif isinstance(params, torch.Tensor):
                        if params.ndim == 0:
                            args = [params]
                        else:
                            args = [to_tensor(x) for x in params.tolist()]
                        matrix = self.matrix_func(*args)
                    else:
                        raise ValueError(f"Unsupported parameter type: {type(params)}")
                    
                    # 只在matrix不是tensor时才转为tensor，保证梯度可传递
                    if not isinstance(matrix, torch.Tensor):
                        # 递归地将所有元素转为tensor
                        matrix = torch.stack([
                            torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.complex64, device=next(self.parameters()).device if list(self.parameters()) else 'cpu') for x in row])
                            for row in matrix
                        ])
                    # 保证类型为complex64
                    matrix = matrix.to(torch.complex64)
                    return matrix
                except Exception as e:
                    raise ValueError(f"Error computing matrix for gate {self.name}: {e}")
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
    """从配置文件加载门定义，支持多参数门"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    for gate_info in config['gates']:
        name = gate_info['name']
        num_qubits = gate_info['num_qubits']
        is_parametric = gate_info.get('is_parametric', False)
        
        if is_parametric:
            matrix_func = gate_info.get('matrix_func')
            param_names = gate_info.get('param_names', None)
            if not matrix_func:
                raise ValueError(f"Parametric gate {name} must have matrix_func defined")
            def create_parametric_gate_class(gate_name, n_qubits, m_func, param_names):
                class DynamicGate(ParametricGate):
                    def __init__(self, **kwargs):
                        super().__init__(gate_name, n_qubits, matrix_func_str=m_func, is_parametric=True, param_names=param_names)
                return DynamicGate
            gate_class = create_parametric_gate_class(name, num_qubits, matrix_func, param_names)
        else:
            matrix = gate_info.get('matrix')
            if not matrix:
                raise ValueError(f"Non-parametric gate {name} must have matrix defined")
            matrix = _parse_complex_matrix(matrix)
            def create_non_parametric_gate_class(gate_name, n_qubits, m):
                class DynamicGate(QuantumGate):
                    def __init__(self, **kwargs):
                        super().__init__(gate_name, n_qubits, is_parametric=False)
                        self.matrix = torch.tensor(m, dtype=torch.complex64)
                    def get_matrix(self, params=None):
                        return self.matrix
                return DynamicGate
            gate_class = create_non_parametric_gate_class(name, num_qubits, matrix)
        QuantumGate.register(name, num_qubits, is_parametric=is_parametric)(gate_class)


class Circuit(nn.Module):
    def __init__(self, num_qubits: int, device: torch.device = None):
        super(Circuit, self).__init__()
        self.num_qubits = num_qubits
        self.gates = []  # 存储门操作序列
        
        # 设置device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化量子态为|0⟩态
        self.register_buffer('qubits', torch.zeros(2**num_qubits, dtype=torch.complex64, device=self.device))
        self.qubits[0] = 1.0

    def add_gate(self, gate_name: str, qubit_indices: List[int], params: Union[torch.Tensor, None] = None):
        """添加门操作到电路
        
        Args:
            gate_name: 门的名称
            qubit_indices: 门操作作用的量子比特索引
            params: 参数化门的参数（如果有）
        """
        if qubit_indices[0] >= self.num_qubits:
            raise ValueError(f"Invalid qubit index: {qubit_indices[0]}")
        
        # 获取门类
        gate_class = QuantumGate._registry.get(gate_name)
        if gate_class is None:
            raise ValueError(f"Gate {gate_name} not registered")
        
        # 创建门实例来检查是否为参数化门
        gate_instance = gate_class()
        is_parametric = getattr(gate_instance, 'is_parametric', False)
        
        if is_parametric and params is None:
            raise ValueError(f"Gate {gate_name} requires parameters")
            
        # 如果是参数化门，设置参数
        if is_parametric:
            gate_instance.to(self.device)
            
            # 处理参数
            if isinstance(params, (list, tuple)):
                params = torch.tensor(params, dtype=torch.float32, device=self.device)
            elif isinstance(params, dict):
                # 如果参数是dict，按门定义顺序转为list
                param_names = getattr(gate_instance, 'param_names', list(params.keys()))
                params = [params[k] for k in param_names]
                params = torch.tensor(params, dtype=torch.float32, device=self.device)
            
            # 用register_parameter注册参数，确保梯度可传递
            for i, param_name in enumerate(gate_instance.param_names):
                param_value = params[i] if params.dim() > 0 else params
                if isinstance(param_value, torch.Tensor):
                    param_tensor = param_value.detach().clone().requires_grad_(True)
                else:
                    param_tensor = torch.tensor(param_value, dtype=torch.float32, device=self.device, requires_grad=True)
                param = nn.Parameter(param_tensor)
                gate_instance.register_parameter(param_name, param)
            
            # 将门添加到模块
            gate_module_name = f"{gate_name}_{len(self.gates)}"
            self.add_module(gate_module_name, gate_instance)
            
            self.gates.append((gate_instance, qubit_indices))
        else:
            # 非参数化门
            gate_instance.to(self.device)
            self.gates.append((gate_instance, qubit_indices))

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """执行量子电路
        
        Args:
            x: 输入量子态，形状为 [batch_size, 2**num_qubits]
            
        Returns:
            torch.Tensor: 输出量子态，形状为 [batch_size, 2**num_qubits]
        """
        if x is None:
            # 如果没有输入，使用|0⟩态
            x = self.qubits.unsqueeze(0)  # 添加batch维度
        else:
            # 确保输入在正确的设备上
            x = x.to(self.device)
        
        # 应用量子门
        for gate, qubit_indices in self.gates:
            x = gate(x, qubit_indices)
        
        return x

    def draw(self, max_gates: int = 10, show_params: bool = True) -> str:
        """
        绘制量子电路图（美化版）
        - 每个qubit一行
        - 单比特门用[门名]方框
        - 双比特受控门用●和⊕/Z区分控制和目标
        - 其他多比特门用跨多行的|门名|方框
        """
        if not self.gates:
            return "Empty circuit"
        # 只显示max_gates个门
        gates_to_show = self.gates
        if len(self.gates) > max_gates:
            first_half = self.gates[:max_gates//2]
            second_half = self.gates[-(max_gates//2):]
            gates_to_show = first_half + [None] + second_half
        n = self.num_qubits
        # 初始化每行
        lines = [["──"] * len(gates_to_show) for _ in range(n)]
        # 受控门类型
        controlled_gates = {"CNOT": "⊕", "CZ": "Z"}
        for col, gate_info in enumerate(gates_to_show):
            if gate_info is None:
                for i in range(n):
                    lines[i][col] = " ... "
                continue
            gate, qubit_indices = gate_info
            name = gate.name
            # 参数化门显示参数
            if gate.is_parametric:
                params = [getattr(gate, param_name) for param_name in gate.param_names]
                param_str = ",".join([f"{p.item():.2f}" for p in params])
                display_name = f"{name}({param_str})"
            else:
                display_name = name
            # 单比特门
            if len(qubit_indices) == 1:
                idx = qubit_indices[0]
                lines[idx][col] = f"[{display_name}]"
            # 双比特门
            elif len(qubit_indices) == 2:
                q0, q1 = sorted(qubit_indices)
                # 判断是否为受控门
                if name in controlled_gates:
                    ctrl, tgt = qubit_indices
                    # 画连线
                    for i in range(min(ctrl, tgt)+1, max(ctrl, tgt)):
                        lines[i][col] = "  │  "
                    # 控制位
                    lines[ctrl][col] = "  ●  "
                    # 目标位
                    lines[tgt][col] = f"  {controlled_gates[name]}  "
                else:
                    # 非受控双比特门，画跨两行的方框
                    for i in range(n):
                        if i == q0:
                            lines[i][col] = f"┌─{display_name}─┐"
                        elif i == q1:
                            lines[i][col] = f"└─{'─'*len(display_name)}─┘"
                        elif q0 < i < q1:
                            lines[i][col] = f"│ {' '*len(display_name)} │"
            # 多比特门
            else:
                qmin, qmax = min(qubit_indices), max(qubit_indices)
                for i in range(n):
                    if i == qmin:
                        lines[i][col] = f"┌─{display_name}─┐"
                    elif i == qmax:
                        lines[i][col] = f"└─{'─'*len(display_name)}─┘"
                    elif qmin < i < qmax:
                        lines[i][col] = f"│ {' '*len(display_name)} │"
        # 拼接
        result = []
        for i, line in enumerate(lines):
            result.append(f"q{i}: " + "".join(line))
        return "\n".join(result)
        
    @staticmethod
    def from_json(json_path_or_dict):
        """
        从json文件或dict构建Circuit实例。
        支持两种格式：
        
        1. 分层格式（包含layer信息）：
        {
            "0": [ {"gate_name": ..., "parameters": ..., "qubits": [...]}, ...],
            "1": [...],
            ...
        }
        
        2. 门序列格式（不包含layer信息）：
        [
            {"gate_name": ..., "parameters": ..., "qubits": [...]},
            {"gate_name": ..., "parameters": ..., "qubits": [...]},
            ...
        ]
        """
        # 加载json
        if isinstance(json_path_or_dict, str):
            with open(json_path_or_dict, 'r') as f:
                circuit_data = json.load(f)
        else:
            circuit_data = json_path_or_dict
        
        # 判断是否为分层格式
        is_layered = isinstance(circuit_data, dict) and all(
            isinstance(key, str) and key.isdigit() for key in circuit_data.keys()
        )
        
        # 统计最大qubit数
        max_qubit = -1
        if is_layered:
            # 分层格式：遍历所有层
            for layer in circuit_data.values():
                for gate in layer:
                    max_in_gate = max(gate["qubits"])
                    if max_in_gate > max_qubit:
                        max_qubit = max_in_gate
        else:
            # 门序列格式：直接遍历门列表
            for gate in circuit_data:
                max_in_gate = max(gate["qubits"])
                if max_in_gate > max_qubit:
                    max_qubit = max_in_gate
        
        num_qubits = max_qubit + 1
        circuit = Circuit(num_qubits)
        
        def process_gate(gate):
            """处理单个门"""
            gate_name = gate["gate_name"]
            if gate_name in {"MX", "MY", "MZ"}:
                return  # 跳过测量操作
            
            qubits = gate["qubits"]
            params = gate.get("parameters", None)
            
            # 如果参数是空dict，视为None
            if isinstance(params, dict) and not params:
                params = None
            
            # 如果参数是dict且有内容，按门定义顺序转为list
            if isinstance(params, dict) and params:
                # 获取门的param_names
                gate_obj = QuantumGate.get_gate(gate_name)
                param_names = getattr(gate_obj, 'param_names', list(params.keys()))
                params = [params[k] for k in param_names]
                params = torch.tensor(params, dtype=torch.float32) if len(params) > 0 else None
            elif isinstance(params, (list, tuple)):
                params = torch.tensor(params, dtype=torch.float32)
            
            circuit.add_gate(gate_name, qubits, params)
        
        if is_layered:
            # 分层格式：按层顺序添加门
            print(f"检测到分层格式，按层顺序处理 {len(circuit_data)} 层")
            for layer_idx in sorted(circuit_data, key=lambda x: int(x)):
                for gate in circuit_data[layer_idx]:
                    process_gate(gate)
        else:
            # 门序列格式：直接按顺序添加门
            print(f"检测到门序列格式，按顺序处理 {len(circuit_data)} 个门")
            for gate in circuit_data:
                process_gate(gate)
        
        return circuit
        
        
        