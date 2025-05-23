import torch
from typing import List, Optional, Dict, Tuple
import numpy as np
from src.circuit import Circuit, QuantumGate
import sympy as sp
from sympy import symbols, Matrix
import math
import torch.nn as nn

class AbstractGate:
    """优化后的量子门
    
    这个类表示经过优化（如门融合）后的量子门操作。
    它只关注门的数学特性（矩阵表示）和作用的量子比特，而不关心门的物理意义。
    """
    
    def __init__(self, matrix: torch.Tensor, qubit_indices: List[int], 
                 is_parametric: bool = False, param_names: Optional[List[str]] = None):
        """初始化优化后的量子门
        
        Args:
            matrix: 门的矩阵表示，可以是参数化的函数
            qubit_indices: 门操作作用的量子比特索引
            is_parametric: 是否为参数化门
            param_names: 参数化门的参数名称列表
        """
        self.matrix = matrix
        self.qubit_indices = qubit_indices
        self.num_qubits = len(qubit_indices)
        self.is_parametric = is_parametric
        self.param_names = param_names if param_names else []
    
    @classmethod
    def from_quantum_gate(cls, gate: 'QuantumGate', qubit_indices: List[int]) -> 'AbstractGate':
        """从QuantumGate创建AbstractGate实例
        
        Args:
            gate: 量子门实例
            qubit_indices: 门操作作用的量子比特索引
            
        Returns:
            AbstractGate: 优化后的量子门实例
        """
        if gate.is_parametric:
            # 对于参数化门，我们需要保存参数化函数
            matrix_func = gate.get_matrix
            param_names = [f"theta_{i}" for i in range(len(qubit_indices))]
            return cls(matrix_func, qubit_indices, is_parametric=True, param_names=param_names)
        else:
            # 对于非参数化门，直接使用矩阵
            matrix = gate.get_matrix()
            return cls(matrix, qubit_indices, is_parametric=False)
    
    def get_matrix(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """获取门的矩阵表示
        
        Args:
            params: 参数化门的参数字典
            
        Returns:
            torch.Tensor: 门的矩阵表示
        """
        if self.is_parametric:
            if params is None:
                raise ValueError("Parameterized gate requires parameters")
            if not all(name in params for name in self.param_names):
                raise ValueError(f"Missing parameters. Required: {self.param_names}")
            return self.matrix(params)
        return self.matrix
    
    def fuse(self, other: 'AbstractGate') -> Optional['AbstractGate']:
        """尝试与另一个门融合
        
        Args:
            other: 要融合的另一个门
            
        Returns:
            Optional[AbstractGate]: 融合后的新门，如果无法融合则返回None
        """
        # 检查是否可以融合（例如，作用在相邻的量子比特上）
        if not self._can_fuse(other):
            return None
        
        # 计算融合后的矩阵
        fused_matrix = self._compute_fused_matrix(other)
        
        # 合并量子比特索引
        fused_indices = sorted(set(self.qubit_indices + other.qubit_indices))
        
        # 合并参数（如果是参数化门）
        is_parametric = self.is_parametric or other.is_parametric
        param_names = list(set(self.param_names + other.param_names))
        
        return AbstractGate(fused_matrix, fused_indices, is_parametric, param_names)
    
    def _can_fuse(self, other: 'AbstractGate') -> bool:
        """检查是否可以与另一个门融合
        
        判断标准：融合后的计算复杂度是否不高于融合前的计算复杂度
        对于state vector表示，复杂度由矩阵乘法决定：2^(m+n)，其中m是state vector的比特数，n是门的比特数
        
        Args:
            other: 要检查的另一个门
            
        Returns:
            bool: 是否可以融合
        """
        # 获取两个门作用的量子比特集合
        qubits1 = set(self.qubit_indices)
        qubits2 = set(other.qubit_indices)
        
        # 计算融合后的量子比特数量
        fused_qubits = len(qubits1.union(qubits2))
        
        # 计算融合前的复杂度
        # 对于state vector，复杂度为2^(m+n)，其中m是state vector的比特数，n是门的比特数
        # 这里我们假设state vector的比特数等于门的最大量子比特索引+1
        max_qubit = max(max(qubits1), max(qubits2))
        state_size = max_qubit + 1
        
        # 融合前的复杂度：两个门分别作用
        complexity_before = 2**(state_size + len(qubits1)) + 2**(state_size + len(qubits2))
        
        # 融合后的复杂度：一个更大的门作用
        complexity_after = 2**(state_size + fused_qubits)
        
        # 如果融合后的复杂度不高于融合前，则可以融合
        return complexity_after <= complexity_before
    
    def _compute_fused_matrix(self, other: 'AbstractGate') -> torch.Tensor:
        """计算融合后的矩阵
        
        对于非参数化门，在公共量子比特上进行缩并操作
        对于参数化门，使用sympy计算融合后的函数形式
        
        Args:
            other: 要融合的另一个门
            
        Returns:
            torch.Tensor: 融合后的矩阵
        """
        if not self.is_parametric and not other.is_parametric:
            # 非参数化门的情况：在公共量子比特上进行缩并
            matrix1 = self.get_matrix()
            matrix2 = other.get_matrix()
            
            # 获取两个门的量子比特集合
            qubits1 = set(self.qubit_indices)
            qubits2 = set(other.qubit_indices)
            
            # 计算公共量子比特
            common_qubits = qubits1.intersection(qubits2)
            
            # 计算每个门的维度
            dim1 = 2**len(qubits1)
            dim2 = 2**len(qubits2)
            
            # 重塑矩阵为张量形式
            tensor1 = matrix1.reshape([2] * (2 * len(qubits1)))
            tensor2 = matrix2.reshape([2] * (2 * len(qubits2)))
            
            # 构建索引映射
            indices1 = []
            indices2 = []
            for i, q in enumerate(sorted(qubits1)):
                if q in common_qubits:
                    # 公共量子比特的索引
                    indices1.extend([f'i_{q}', f'o_{q}'])
                else:
                    # 非公共量子比特的索引
                    indices1.extend([f'i_{q}', f'o_{q}'])
            
            for i, q in enumerate(sorted(qubits2)):
                if q in common_qubits:
                    # 公共量子比特的索引
                    indices2.extend([f'i_{q}', f'o_{q}'])
                else:
                    # 非公共量子比特的索引
                    indices2.extend([f'i_{q}', f'o_{q}'])
            
            # 在公共索引上进行缩并
            from src.tensor_ops import TensorOps
            fused_tensor = TensorOps.contract(
                tensor1, tensor2,
                [indices1.index(idx) for idx in indices1 if idx in indices2],
                [indices2.index(idx) for idx in indices2 if idx in indices1]
            )
            
            # 重塑为矩阵形式
            fused_dim = 2**len(qubits1.union(qubits2))
            return fused_tensor.reshape(fused_dim, fused_dim)
        else:
            # 参数化门的情况：使用sympy计算融合后的函数形式
            # 创建符号变量
            params1 = [sp.symbols(f'theta_{i}') for i in range(len(self.param_names))]
            params2 = [sp.symbols(f'theta_{i+len(params1)}') for i in range(len(other.param_names))]
            
            # 获取两个门的矩阵函数
            matrix1 = self.get_matrix({name: param for name, param in zip(self.param_names, params1)})
            matrix2 = other.get_matrix({name: param for name, param in zip(other.param_names, params2)})
            
            # 获取两个门的量子比特集合
            qubits1 = set(self.qubit_indices)
            qubits2 = set(other.qubit_indices)
            
            # 计算公共量子比特
            common_qubits = qubits1.intersection(qubits2)
            
            # 将torch.Tensor转换为sympy矩阵
            matrix1_sym = Matrix([[sp.sympify(str(x.item())) for x in row] for row in matrix1])
            matrix2_sym = Matrix([[sp.sympify(str(x.item())) for x in row] for row in matrix2])
            
            # 计算融合后的函数形式
            def fused_matrix_func(params):
                # 创建参数字典
                param_dict = {str(p): v for p, v in zip(params1 + params2, params)}
                
                # 替换符号并计算
                matrix1 = torch.tensor([[complex(x) for x in row] for row in matrix1_sym.subs(param_dict).tolist()], 
                                     dtype=torch.complex64)
                matrix2 = torch.tensor([[complex(x) for x in row] for row in matrix2_sym.subs(param_dict).tolist()], 
                                     dtype=torch.complex64)
                
                # 重塑为张量形式
                tensor1 = matrix1.reshape([2] * (2 * len(qubits1)))
                tensor2 = matrix2.reshape([2] * (2 * len(qubits2)))
                
                # 构建索引映射
                indices1 = []
                indices2 = []
                for i, q in enumerate(sorted(qubits1)):
                    if q in common_qubits:
                        indices1.extend([f'i_{q}', f'o_{q}'])
                    else:
                        indices1.extend([f'i_{q}', f'o_{q}'])
                
                for i, q in enumerate(sorted(qubits2)):
                    if q in common_qubits:
                        indices2.extend([f'i_{q}', f'o_{q}'])
                    else:
                        indices2.extend([f'i_{q}', f'o_{q}'])
                
                # 在公共索引上进行缩并
                from src.tensor_ops import TensorOps
                fused_tensor = TensorOps.contract(
                    tensor1, tensor2,
                    [indices1.index(idx) for idx in indices1 if idx in indices2],
                    [indices2.index(idx) for idx in indices2 if idx in indices1]
                )
                
                # 重塑为矩阵形式
                fused_dim = 2**len(qubits1.union(qubits2))
                return fused_tensor.reshape(fused_dim, fused_dim)
            
            return fused_matrix_func


class AbstractCircuit(nn.Module):
    """优化后的量子电路
    
    这个类表示经过优化（如门融合）后的量子电路。
    它包含一系列优化后的门操作，这些门可能已经失去了原有的物理意义，
    但保持了正确的数学变换。
    """
    
    def __init__(self, num_qubits: int):
        """初始化优化后的量子电路
        
        Args:
            num_qubits: 量子比特数量
        """
        super().__init__()  # 调用nn.Module的初始化
        self.num_qubits = num_qubits
        self.gates: List[AbstractGate] = []
        self.optimized = False
        self.gate_dependencies = {}  # 存储门之间的依赖关系
        self.gate_layers = []  # 存储优化后的门层
    
    @classmethod
    def from_circuit(cls, circuit: 'Circuit') -> 'AbstractCircuit':
        """从Circuit创建AbstractCircuit实例
        
        Args:
            circuit: 量子电路实例
            
        Returns:
            AbstractCircuit: 优化后的量子电路实例
        """
        abs_circuit = cls(circuit.num_qubits)
        
        # 转换每个门
        for gate, qubit_indices, params in circuit.gates:
            abs_gate = AbstractGate.from_quantum_gate(gate, qubit_indices)
            abs_circuit.add_gate(abs_gate)
        
        return abs_circuit
    
    def add_gate(self, gate: AbstractGate) -> None:
        """添加一个优化后的门
        
        Args:
            gate: 要添加的门
        """
        self.gates.append(gate)
        self.optimized = False
    
    def _build_dependency_graph(self) -> None:
        """构建门之间的依赖关系图
        
        基于量子比特的拓扑关系构建依赖图，考虑：
        1. 作用在相同量子比特上的门
        2. 作用在相邻量子比特上的门（可能可以融合）
        3. 并行执行的可能性
        """
        self.gate_dependencies = {i: set() for i in range(len(self.gates))}
        
        # 构建量子比特到门的映射
        qubit_to_gates = {i: [] for i in range(self.num_qubits)}
        for i, gate in enumerate(self.gates):
            for qubit in gate.qubit_indices:
                qubit_to_gates[qubit].append(i)
        
        # 分析门之间的依赖关系
        for i, gate1 in enumerate(self.gates):
            # 获取gate1作用的所有量子比特
            qubits1 = set(gate1.qubit_indices)
            
            # 检查与后续门的依赖关系
            for j in range(i + 1, len(self.gates)):
                gate2 = self.gates[j]
                qubits2 = set(gate2.qubit_indices)
                
                # 如果两个门作用在相同的量子比特上，它们必须串行执行
                if qubits1.intersection(qubits2):
                    self.gate_dependencies[j].add(i)
                # 如果两个门作用在相邻的量子比特上，且可以融合，则它们可以并行执行
                elif self._are_qubits_adjacent(qubits1, qubits2) and gate1._can_fuse(gate2):
                    # 不添加依赖关系，允许并行执行
                    pass
                # 如果两个门作用在不相邻的量子比特上，它们可以并行执行
                else:
                    # 不添加依赖关系，允许并行执行
                    pass
    
    def _are_qubits_adjacent(self, qubits1: set, qubits2: set) -> bool:
        """检查两组量子比特是否相邻
        
        Args:
            qubits1: 第一组量子比特
            qubits2: 第二组量子比特
            
        Returns:
            bool: 是否相邻
        """
        # 获取所有量子比特的索引
        all_qubits = sorted(list(qubits1.union(qubits2)))
        
        # 检查是否连续
        for i in range(len(all_qubits) - 1):
            if all_qubits[i + 1] - all_qubits[i] > 1:
                return False
        return True
    
    def _schedule_gates(self) -> None:
        """基于依赖关系调度门
        
        使用拓扑排序将门分配到不同的层，每一层的门可以并行执行
        """
        # 计算每个门的入度
        in_degree = {i: len(self.gate_dependencies[i]) for i in range(len(self.gates))}
        
        # 初始化层
        self.gate_layers = []
        current_layer = []
        available_gates = {i for i in range(len(self.gates)) if in_degree[i] == 0}
        
        while available_gates:
            # 尝试将可以融合的门放在同一层
            layer_gates = []
            remaining_gates = set()
            
            for gate_idx in available_gates:
                gate = self.gates[gate_idx]
                can_fuse = False
                
                # 检查是否可以与当前层的某个门融合
                for layer_gate_idx in layer_gates:
                    layer_gate = self.gates[layer_gate_idx]
                    if gate._can_fuse(layer_gate):
                        # 融合门
                        fused_gate = gate.fuse(layer_gate)
                        if fused_gate is not None:
                            # 替换原来的门
                            self.gates[layer_gate_idx] = fused_gate
                            can_fuse = True
                            break
                
                if not can_fuse:
                    layer_gates.append(gate_idx)
            
            # 添加当前层
            self.gate_layers.append(layer_gates)
            
            # 更新可用门
            available_gates = remaining_gates
            for gate_idx in layer_gates:
                # 更新依赖关系
                for j in range(len(self.gates)):
                    if gate_idx in self.gate_dependencies[j]:
                        in_degree[j] -= 1
                        if in_degree[j] == 0:
                            available_gates.add(j)
    
    def optimize(self) -> None:
        """优化电路
        
        优化分为两部分：
        1. 门融合：融合相邻的量子门，降低计算复杂度
        2. 张量网络优化：使用张量网络方法寻找最优的收缩路径
        
        Raises:
            RuntimeError: 如果优化过程中出现错误
        """
        if self.optimized:
            return
        
        try:
            # 第一部分：门融合
            i = 0
            while i < len(self.gates):
                gate1 = self.gates[i]
                fused = False
                
                # 检查与后续门的融合可能性
                for j in range(i + 1, len(self.gates)):
                    gate2 = self.gates[j]
                    
                    # 检查是否相邻且可以融合
                    if self._are_gates_adjacent(gate1, gate2) and self._can_fuse_gates(gate1, gate2):
                        # 融合门
                        fused_gate = gate1.fuse(gate2)
                        if fused_gate is not None:
                            # 替换原来的门
                            self.gates[i] = fused_gate
                            self.gates.pop(j)
                            fused = True
                            break
                
                if not fused:
                    i += 1
            
            # 第二部分：张量网络优化
            # 将电路转换为张量网络
            tn = self.to_tn()
            
            # 获取收缩路径
            path = tn.path
            
            # 遍历收缩路径
            new_gates = []
            state_gate_indices = set()  # 记录已经与state张量收缩的门
            
            for i, j in path:
                if i == 0 or j == 0:  # 与state张量收缩
                    gate_idx = max(i, j) - 1  # 减1是因为state张量在索引0
                    if gate_idx not in state_gate_indices:
                        new_gates.append(self.gates[gate_idx])
                        state_gate_indices.add(gate_idx)
                else:  # 两个量子门张量收缩
                    gate1_idx = i - 1
                    gate2_idx = j - 1
                    if gate1_idx not in state_gate_indices and gate2_idx not in state_gate_indices:
                        # 融合门
                        fused_gate = self.gates[gate1_idx].fuse(self.gates[gate2_idx])
                        if fused_gate is not None:
                            new_gates.append(fused_gate)
                            state_gate_indices.add(gate1_idx)
                            state_gate_indices.add(gate2_idx)
            
            # 更新门列表
            self.gates = new_gates
            
            self.optimized = True
            
        except Exception as e:
            raise RuntimeError(f"Error during circuit optimization: {e}")
    
    def forward(self, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """前向传播，支持自动求导
        
        Args:
            params: 参数化门的参数字典
            
        Returns:
            torch.Tensor: 电路的矩阵表示
        """
        if not self.optimized:
            self.optimize()
        
        # 计算整个电路的矩阵
        circuit_matrix = torch.eye(2**self.num_qubits, dtype=torch.complex64)
        for gate in self.gates:
            gate_matrix = gate.get_matrix(params)
            # 将门矩阵扩展到整个电路空间
            expanded_matrix = self._expand_gate_matrix(gate_matrix, gate.qubit_indices)
            circuit_matrix = expanded_matrix @ circuit_matrix
        
        return circuit_matrix
    
    def _expand_gate_matrix(self, gate_matrix: torch.Tensor, 
                           qubit_indices: List[int]) -> torch.Tensor:
        """将门矩阵扩展到整个电路空间
        
        Args:
            gate_matrix: 门的矩阵
            qubit_indices: 门作用的量子比特索引
            
        Returns:
            torch.Tensor: 扩展后的矩阵
        """
        # 计算门的维度
        gate_dim = 2**len(qubit_indices)
        
        # 创建单位矩阵
        identity = torch.eye(2, dtype=torch.complex64)
        
        # 计算张量积
        expanded_matrix = torch.eye(2**self.num_qubits, dtype=torch.complex64)
        for i in range(self.num_qubits):
            if i in qubit_indices:
                # 对于门作用的量子比特，使用门矩阵的对应部分
                idx = qubit_indices.index(i)
                matrix = gate_matrix.reshape([2] * (2 * len(qubit_indices)))
                matrix = matrix.transpose(*[idx, idx + len(qubit_indices)])
                matrix = matrix.reshape(2, 2)
                expanded_matrix = torch.kron(expanded_matrix, matrix)
            else:
                # 对于其他量子比特，使用单位矩阵
                expanded_matrix = torch.kron(expanded_matrix, identity)
        
        return expanded_matrix

    def to_tn(self) -> 'TensorNetwork':
        """将量子电路转换为张量网络
        
        每个量子门都转换为一个张量，初态state vector视为一个具有和量子比特数相同维度的张量，
        末态视为open index。
        
        Returns:
            TensorNetwork: 转换后的张量网络
        """
        from src.tensor_network import TensorNetwork
        
        # 创建初始态张量
        initial_state = torch.zeros([2] * self.num_qubits, dtype=torch.complex64)
        initial_state[tuple([0] * self.num_qubits)] = 1.0
        
        # 创建张量列表和索引列表
        arrays = [initial_state]
        indices = [['i' + str(i) for i in range(self.num_qubits)]]
        
        # 为每个量子门创建张量和索引
        for i, gate in enumerate(self.gates):
            # 获取门的矩阵
            if gate.is_parametric:
                if not hasattr(gate, 'params') or gate.params is None:
                    raise ValueError("Parameterized gate requires parameters")
                matrix = gate.get_matrix(gate.params)
            else:
                matrix = gate.get_matrix()
            
            # 创建门的张量
            gate_tensor = matrix.reshape([2] * (2 * len(gate.qubit_indices)))
            
            # 创建门的索引
            gate_indices = []
            for j, qubit in enumerate(gate.qubit_indices):
                # 输入索引
                gate_indices.append('i' + str(qubit))
                # 输出索引
                gate_indices.append('o' + str(qubit))
            
            arrays.append(gate_tensor)
            indices.append(gate_indices)
        
        # 创建size_dict
        size_dict = {idx: 2 for idx in set([idx for idx_list in indices for idx in idx_list])}
        
        # 创建输出索引（末态的open index）
        output = ['o' + str(i) for i in range(self.num_qubits)]
        
        return TensorNetwork(arrays, indices, size_dict, output, variational=True)

    def _are_gates_adjacent(self, gate1: AbstractGate, gate2: AbstractGate) -> bool:
        """判断两个量子门是否相邻
        
        两个量子门相邻的条件：
        1. 它们有公共的量子比特
        2. 在所有的公共比特上，两个门之间没有其他的量子门
        
        Args:
            gate1: 第一个量子门
            gate2: 第二个量子门
            
        Returns:
            bool: 是否相邻
        """
        # 获取两个门作用的量子比特集合
        qubits1 = set(gate1.qubit_indices)
        qubits2 = set(gate2.qubit_indices)
        
        # 检查是否有公共量子比特
        common_qubits = qubits1.intersection(qubits2)
        if not common_qubits:
            return False
        
        # 获取两个门在电路中的位置
        idx1 = self.gates.index(gate1)
        idx2 = self.gates.index(gate2)
        
        # 确保gate1在gate2之前
        if idx1 > idx2:
            gate1, gate2 = gate2, gate1
            idx1, idx2 = idx2, idx1
        
        # 检查两个门之间是否有其他门作用在公共量子比特上
        for i in range(idx1 + 1, idx2):
            other_gate = self.gates[i]
            other_qubits = set(other_gate.qubit_indices)
            if common_qubits.intersection(other_qubits):
                return False
        
        return True

    def _can_fuse_gates(self, gate1: AbstractGate, gate2: AbstractGate) -> bool:
        """判断两个量子门是否可以融合
        
        判断标准：融合后计算复杂度不增
        
        Args:
            gate1: 第一个量子门
            gate2: 第二个量子门
            
        Returns:
            bool: 是否可以融合
        """
        # 获取两个门作用的量子比特集合
        qubits1 = set(gate1.qubit_indices)
        qubits2 = set(gate2.qubit_indices)
        
        # 计算融合后的量子比特数量
        fused_qubits = len(qubits1.union(qubits2))
        
        # 计算融合前的复杂度
        # 对于state vector，复杂度为2^(m+n)，其中m是state vector的比特数，n是门的比特数
        max_qubit = max(max(qubits1), max(qubits2))
        state_size = max_qubit + 1
        
        # 融合前的复杂度：两个门分别作用
        complexity_before = 2**(state_size + len(qubits1)) + 2**(state_size + len(qubits2))
        
        # 融合后的复杂度：一个更大的门作用
        complexity_after = 2**(state_size + fused_qubits)
        
        # 如果融合后的复杂度不高于融合前，则可以融合
        return complexity_after <= complexity_before
