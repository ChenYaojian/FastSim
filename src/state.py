import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Type
from enum import Enum
import numpy as np

class StateType(Enum):
    """量子态表示类型"""
    STATE_VECTOR = "state_vector"
    MPS = "mps"
    TNS = "tns"
    NNQS = "nnqs"
    DENSITY_MATRIX = "density_matrix"

class AbstractState(ABC):
    """量子态抽象基类
    
    这个类定义了量子态的基本接口，包括：
    1. 初始化量子态
    2. 应用量子门
    3. 测量操作
    4. 获取量子态信息
    """
    
    _state_classes: Dict[StateType, Type['AbstractState']] = {}
    
    def __init__(self, num_qubits: int):
        """初始化量子态
        
        Args:
            num_qubits: 量子比特数量
        """
        self.num_qubits = num_qubits
    
    @classmethod
    def register_state_class(cls, state_type: StateType):
        """注册量子态类
        
        Args:
            state_type: 量子态类型
        """
        def decorator(state_class: Type['AbstractState']):
            cls._state_classes[state_type] = state_class
            return state_class
        return decorator
    
    @classmethod
    def create_state(cls, num_qubits: int, state_type: Union[str, StateType]=StateType.STATE_VECTOR,  
                    bitstring: Optional[str] = None, initial_state: Optional[torch.Tensor] = None,
                    max_bond_dim: Optional[int] = None, **kwargs) -> 'AbstractState':
        """创建量子态实例
        
        Args:
            state_type: 量子态类型（字符串或StateType枚举）
            num_qubits: 量子比特数量
            bitstring: 比特串，用于初始化量子态（例如："101"表示|101⟩态）
            initial_state: 初始量子态向量，形状为(2^num_qubits,)
            max_bond_dim: MPS的最大键维数，仅在state_type为MPS时使用
            **kwargs: 其他初始化参数
            
        Returns:
            AbstractState: 量子态实例
            
        Raises:
            ValueError: 如果指定的状态类型不存在或参数不匹配
        """
        if isinstance(state_type, str):
            try:
                state_type = StateType(state_type)
            except ValueError:
                raise ValueError(f"Unknown state type: {state_type}. "
                               f"Available types: {[t.value for t in StateType]}")
        
        if state_type not in cls._state_classes:
            raise ValueError(f"State type {state_type.value} not registered")
        
        # 检查参数
        if bitstring is not None and initial_state is not None:
            raise ValueError("Cannot specify both bitstring and initial_state")
        
        if initial_state is not None:
            if initial_state.shape != (2**num_qubits,):
                raise ValueError(f"Initial state shape {initial_state.shape} does not match "
                               f"expected shape {(2**num_qubits,)}")
        
        # 创建状态实例
        state_class = cls._state_classes[state_type]
        
        # 根据状态类型设置参数
        if state_type == StateType.MPS:
            if max_bond_dim is None:
                max_bond_dim = 2  # 默认值
            state = state_class(num_qubits, max_bond_dim=max_bond_dim, **kwargs)
        else:
            state = state_class(num_qubits, **kwargs)
        
        # 初始化状态
        if bitstring is not None:
            state.initialize_from_bitstring(bitstring)
        elif initial_state is not None:
            state.initialize(initial_state)
        else:
            state.initialize()
            
        return state
    
    @abstractmethod
    def initialize_from_bitstring(self, bitstring: str) -> None:
        """从比特串初始化量子态
        
        Args:
            bitstring: 比特串，例如："101"表示|101⟩态
            
        Raises:
            ValueError: 如果比特串长度不等于量子比特数量
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} does not match "
                           f"number of qubits {self.num_qubits}")
        pass
    
    @abstractmethod
    def initialize(self, initial_state: Optional[torch.Tensor] = None) -> None:
        """初始化量子态
        
        Args:
            initial_state: 初始量子态，如果为None则初始化为|0⟩态
        """
        pass
    
    @abstractmethod
    def apply_gate(self, gate: 'QuantumGate', qubit_indices: List[int], 
                  params: Optional[torch.Tensor] = None) -> None:
        """应用量子门
        
        Args:
            gate: 要应用的量子门
            qubit_indices: 门操作作用的量子比特索引
            params: 参数化门的参数（如果有）
        """
        pass
    
    @abstractmethod
    def measure(self, qubit_indices: Optional[List[int]] = None, 
               num_shots: int = 1) -> torch.Tensor:
        """测量量子态
        
        Args:
            qubit_indices: 要测量的量子比特索引，如果为None则测量所有量子比特
            num_shots: 测量次数
            
        Returns:
            torch.Tensor: 测量结果
        """
        pass
    
    @abstractmethod
    def get_state_vector(self) -> torch.Tensor:
        """获取量子态向量表示
        
        Returns:
            torch.Tensor: 量子态向量
        """
        pass
    
    @abstractmethod
    def get_density_matrix(self) -> torch.Tensor:
        """获取密度矩阵表示
        
        Returns:
            torch.Tensor: 密度矩阵
        """
        pass
    
    @abstractmethod
    def get_expectation(self, observable: torch.Tensor, 
                       qubit_indices: Optional[List[int]] = None) -> torch.Tensor:
        """计算可观测量期望值
        
        Args:
            observable: 可观测量算符
            qubit_indices: 作用的量子比特索引，如果为None则作用于所有量子比特
            
        Returns:
            torch.Tensor: 期望值
        """
        pass
    
    @abstractmethod
    def get_entanglement_entropy(self, partition: List[int]) -> torch.Tensor:
        """计算纠缠熵
        
        Args:
            partition: 子系统A的量子比特索引
            
        Returns:
            torch.Tensor: 纠缠熵
        """
        pass
    
    @abstractmethod
    def get_fidelity(self, other_state: 'AbstractState') -> torch.Tensor:
        """计算与另一个量子态的保真度
        
        Args:
            other_state: 另一个量子态
            
        Returns:
            torch.Tensor: 保真度
        """
        pass

# 注册具体的状态类
@AbstractState.register_state_class(StateType.STATE_VECTOR)
class StateVector(AbstractState):
    """态向量表示的量子态
    
    使用复数向量表示量子态，支持任意精度。
    """
    
    def __init__(self, num_qubits: int, dtype: torch.dtype = torch.complex64):
        """初始化态向量量子态
        
        Args:
            num_qubits: 量子比特数量
            dtype: 数据类型，默认为torch.complex64
        """
        super().__init__(num_qubits)
        self.dtype = dtype
        self.state_type = StateType.STATE_VECTOR
        self.state_vector = None
        self.initialize()
    
    def initialize(self, initial_state: Optional[torch.Tensor] = None) -> None:
        """初始化量子态
        
        Args:
            initial_state: 初始量子态，如果为None则初始化为|0⟩态
        """
        if initial_state is None:
            self.state_vector = torch.zeros(2**self.num_qubits, dtype=self.dtype)
            self.state_vector[0] = 1.0
        else:
            if initial_state.shape != (2**self.num_qubits,):
                raise ValueError(f"Initial state shape {initial_state.shape} does not match "
                               f"expected shape {(2**self.num_qubits,)}")
            self.state_vector = initial_state.to(dtype=self.dtype)
    
    def initialize_from_bitstring(self, bitstring: str) -> None:
        """从比特串初始化量子态
        
        Args:
            bitstring: 比特串，例如："101"表示|101⟩态
        """
        super().initialize_from_bitstring(bitstring)  # 检查比特串长度
        index = int(bitstring, 2)
        self.state_vector = torch.zeros(2**self.num_qubits, dtype=self.dtype)
        self.state_vector[index] = 1.0
    
    def apply_gate(self, gate: 'QuantumGate', qubit_indices: List[int], 
                  params: Optional[torch.Tensor] = None) -> None:
        """应用量子门（待实现）
        """
        pass
    
    def measure(self, qubit_indices: Optional[List[int]] = None, 
               num_shots: int = 1) -> torch.Tensor:
        """测量量子态
        
        Args:
            qubit_indices: 要测量的量子比特索引，如果为None则测量所有量子比特
            num_shots: 测量次数
            
        Returns:
            torch.Tensor: 测量结果，形状为(num_shots, len(qubit_indices))
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算测量概率
        probs = torch.abs(self.state_vector) ** 2
        
        # 进行测量
        results = []
        for _ in range(num_shots):
            # 采样测量结果
            measured_state = torch.multinomial(probs, 1).item()
            # 将测量结果转换为比特串
            bitstring = format(measured_state, f'0{self.num_qubits}b')
            # 提取指定量子比特的测量结果
            result = [int(bitstring[i]) for i in qubit_indices]
            results.append(result)
        
        return torch.tensor(results, dtype=torch.int64)
    
    def sample_final_state(self, num_shots: int = 1000, 
                          qubit_indices: Optional[List[int]] = None,
                          return_json: bool = True) -> Union[Dict[int, int], str]:
        """末态采样
        
        根据量子态的振幅计算概率分布，然后进行采样。
        
        Args:
            num_shots: 采样次数
            qubit_indices: 要采样的量子比特索引，如果为None则采样所有量子比特
            return_json: 是否返回JSON字符串，如果为False则返回字典
            
        Returns:
            Union[Dict[int, int], str]: 采样结果
                - 如果return_json=True: 返回JSON字符串
                - 如果return_json=False: 返回字典，键为比特串的整数表示，值为采样次数
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算测量概率
        probs = torch.abs(self.state_vector) ** 2
        
        # 进行采样
        sampled_states = torch.multinomial(probs, num_shots, replacement=True)
        
        # 统计每个比特串的出现次数
        bitstring_counts = {}
        for state_idx in sampled_states:
            # 将状态索引转换为比特串
            bitstring = format(state_idx.item(), f'0{self.num_qubits}b')
            
            # 提取指定量子比特的比特串
            if qubit_indices != list(range(self.num_qubits)):
                # 如果只采样部分量子比特，需要重新构建比特串
                partial_bitstring = ''.join([bitstring[i] for i in qubit_indices])
                # 将部分比特串转换为整数
                bitstring_int = int(partial_bitstring, 2)
            else:
                # 采样所有量子比特，直接使用状态索引
                bitstring_int = state_idx.item()
            
            # 更新计数
            bitstring_counts[bitstring_int] = bitstring_counts.get(bitstring_int, 0) + 1
        
        if return_json:
            import json
            return json.dumps(bitstring_counts, indent=2)
        else:
            return bitstring_counts
    
    def get_probability_distribution(self, qubit_indices: Optional[List[int]] = None) -> Dict[int, float]:
        """获取量子态的概率分布
        
        Args:
            qubit_indices: 要计算的量子比特索引，如果为None则计算所有量子比特
            
        Returns:
            Dict[int, float]: 概率分布，键为比特串的整数表示，值为概率
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算测量概率
        probs = torch.abs(self.state_vector) ** 2
        
        # 构建概率分布
        prob_dist = {}
        for state_idx, prob in enumerate(probs):
            if prob > 1e-10:  # 只保留非零概率
                # 将状态索引转换为比特串
                bitstring = format(state_idx, f'0{self.num_qubits}b')
                
                # 提取指定量子比特的比特串
                if qubit_indices != list(range(self.num_qubits)):
                    # 如果只计算部分量子比特，需要重新构建比特串
                    partial_bitstring = ''.join([bitstring[i] for i in qubit_indices])
                    # 将部分比特串转换为整数
                    bitstring_int = int(partial_bitstring, 2)
                else:
                    # 计算所有量子比特，直接使用状态索引
                    bitstring_int = state_idx
                
                # 累加概率（如果有多个状态映射到同一个部分比特串）
                prob_dist[bitstring_int] = prob_dist.get(bitstring_int, 0.0) + prob.item()
        
        return prob_dist
    
    def get_state_vector(self) -> torch.Tensor:
        """获取量子态向量表示
        
        Returns:
            torch.Tensor: 量子态向量
        """
        return self.state_vector
    
    def get_density_matrix(self) -> torch.Tensor:
        """获取密度矩阵表示
        
        Returns:
            torch.Tensor: 密度矩阵
        """
        return torch.outer(self.state_vector, self.state_vector.conj())
    
    def get_expectation(self, observable: torch.Tensor, 
                       qubit_indices: Optional[List[int]] = None) -> torch.Tensor:
        """计算可观测量期望值
        
        Args:
            observable: 可观测量算符
            qubit_indices: 作用的量子比特索引，如果为None则作用于所有量子比特
            
        Returns:
            torch.Tensor: 期望值
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算期望值
        # 从态向量中获取子态向量
        full_indices = list(range(self.num_qubits))
        sub_state_vector = self.state_vector.clone()
        
        # 仅保留指定量子比特的态
        for i in reversed(full_indices):
            if i not in qubit_indices:
                sub_state_vector = sub_state_vector.view(-1, 2).select(1, 0)
        
        # 计算子态向量的期望值
        expectation = torch.vdot(sub_state_vector, 
                                 torch.matmul(observable, sub_state_vector))
        return expectation.real
    
    def get_entanglement_entropy(self, partition: List[int]) -> torch.Tensor:
        """计算纠缠熵
        
        Args:
            partition: 子系统A的量子比特索引
            
        Returns:
            torch.Tensor: 纠缠熵
        """
        # 将态向量重塑为矩阵形式
        dim_A = 2**len(partition)
        dim_B = 2**(self.num_qubits - len(partition))
        state_matrix = self.state_vector.reshape(dim_A, dim_B)
        
        # 计算约化密度矩阵
        rho_A = torch.matmul(state_matrix, state_matrix.conj().T)
        
        # 计算特征值
        eigenvalues = torch.linalg.eigvalsh(rho_A)
        
        # 计算von Neumann熵
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-10))
        return entropy
    
    def get_fidelity(self, other_state: 'AbstractState') -> torch.Tensor:
        """计算与另一个量子态的保真度
        
        Args:
            other_state: 另一个量子态
            
        Returns:
            torch.Tensor: 保真度
        """
        if not isinstance(other_state, StateVector):
            raise TypeError("Fidelity can only be calculated between state vectors")
        
        # 计算保真度 |⟨ψ|φ⟩|²
        overlap = torch.vdot(self.state_vector, other_state.state_vector)
        fidelity = torch.abs(overlap) ** 2
        return fidelity

    def tofile(self, filename: str) -> None:
        """将量子态保存到二进制文件
        
        Args:
            filename: 输出文件名
            
        文件格式：
        - 前4字节：量子比特数量（uint32）
        - 接下来8字节：数据类型信息（uint64，表示torch.dtype的数值）
        - 剩余字节：态向量数据（复数数组）
        """
        with open(filename, 'wb') as f:
            # 写入量子比特数量（uint32）
            f.write(np.uint32(self.num_qubits).tobytes())
            
            # 写入数据类型信息（uint64）
            # 将dtype转换为字符串，然后提取数值部分
            dtype_str = str(self.dtype)
            if 'complex64' in dtype_str:
                dtype_value = 64
            elif 'complex128' in dtype_str:
                dtype_value = 128
            else:
                raise ValueError(f"不支持的数据类型: {dtype_str}")
            f.write(np.uint64(dtype_value).tobytes())
            
            # 写入态向量数据
            f.write(self.state_vector.numpy().tobytes())
    
    @classmethod
    def fromfile(cls, filename: str) -> 'StateVector':
        """从二进制文件加载量子态
        
        Args:
            filename: 输入文件名
            
        Returns:
            StateVector: 加载的量子态实例
            
        Raises:
            ValueError: 如果文件格式不正确
        """
        with open(filename, 'rb') as f:
            # 读取量子比特数量
            num_qubits = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            # 读取数据类型信息
            dtype_value = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            # 根据数值确定dtype
            if dtype_value == 64:
                dtype = torch.complex64
            elif dtype_value == 128:
                dtype = torch.complex128
            else:
                raise ValueError(f"不支持的数据类型值: {dtype_value}")
            
            # 创建StateVector实例
            state = cls(num_qubits, dtype=dtype)
            
            # 读取态向量数据
            data = np.frombuffer(f.read(), dtype=np.complex64)
            if len(data) != 2**num_qubits:
                raise ValueError(f"文件数据长度 {len(data)} 与量子比特数量 {num_qubits} 不匹配")
            
            # 初始化态向量
            state.state_vector = torch.from_numpy(data).to(dtype=dtype)
            
            return state

@AbstractState.register_state_class(StateType.MPS)
class MPS(AbstractState):
    """矩阵乘积态表示的量子态
    
    使用矩阵乘积态(MPS)表示量子态，支持SVD截断。
    """
    
    def __init__(self, num_qubits: int, max_bond_dim: int = 2, dtype: torch.dtype = torch.complex64):
        """初始化MPS量子态
        
        Args:
            num_qubits: 量子比特数量
            max_bond_dim: 最大键维数，用于SVD截断
            dtype: 数据类型，默认为torch.complex64
        """
        super().__init__(num_qubits)
        self.dtype = dtype
        self.state_type = StateType.MPS
        self.max_bond_dim = max_bond_dim
        self.tensors = None  # 存储MPS张量
        self.initialize()
    
    def initialize(self, initial_state: Optional[torch.Tensor] = None) -> None:
        """初始化量子态
        
        Args:
            initial_state: 初始量子态，如果为None则初始化为|0⟩态
        """
        if initial_state is None:
            # 初始化|0⟩态
            self.tensors = []
            for i in range(self.num_qubits):
                if i == 0:
                    tensor = torch.zeros((1, 2, 1), dtype=self.dtype)
                    tensor[0, 0, 0] = 1.0
                else:
                    tensor = torch.zeros((1, 2, 1), dtype=self.dtype)
                    tensor[0, 0, 0] = 1.0
                self.tensors.append(tensor)
        else:
            if initial_state.shape != (2**self.num_qubits,):
                raise ValueError(f"Initial state shape {initial_state.shape} does not match "
                               f"expected shape {(2**self.num_qubits,)}")
            # 将态向量转换为MPS表示
            self._state_vector_to_mps(initial_state)
    
    def initialize_from_bitstring(self, bitstring: str) -> None:
        """从比特串初始化量子态
        
        Args:
            bitstring: 比特串，例如："101"表示|101⟩态
        """
        super().initialize_from_bitstring(bitstring)  # 检查比特串长度
        
        # 初始化MPS张量
        self.tensors = []
        for i, bit in enumerate(bitstring):
            if i == 0:
                tensor = torch.zeros((1, 2, 1), dtype=self.dtype)
                tensor[0, int(bit), 0] = 1.0
            else:
                tensor = torch.zeros((1, 2, 1), dtype=self.dtype)
                tensor[0, int(bit), 0] = 1.0
            self.tensors.append(tensor)
    
    def _state_vector_to_mps(self, state_vector: torch.Tensor) -> None:
        """将态向量转换为MPS表示
        
        Args:
            state_vector: 态向量
        """
        # 将态向量重塑为矩阵形式
        state_matrix = state_vector.reshape(2, -1)
        
        # 使用SVD分解
        self.tensors = []
        for i in range(self.num_qubits - 1):
            # 执行SVD
            U, S, V = torch.linalg.svd(state_matrix, full_matrices=False)
            
            # 截断奇异值
            if len(S) > self.max_bond_dim:
                U = U[:, :self.max_bond_dim]
                S = S[:self.max_bond_dim]
                V = V[:self.max_bond_dim, :]
            
            # 存储左奇异向量
            tensor = U.reshape(-1, 2, U.shape[1])
            self.tensors.append(tensor)
            
            # 更新状态矩阵
            state_matrix = torch.diag(S) @ V
        
        # 存储最后一个张量
        self.tensors.append(state_matrix.reshape(-1, 2, 1))
    
    def get_state_vector(self) -> torch.Tensor:
        """获取量子态向量表示
        
        Returns:
            torch.Tensor: 量子态向量
        """
        # 从MPS重建态向量
        state = self.tensors[0]
        for i in range(1, self.num_qubits):
            state = torch.tensordot(state, self.tensors[i], dims=([-1], [0]))
        
        # 重塑为向量形式
        return state.reshape(-1)
    
    def get_density_matrix(self) -> torch.Tensor:
        """获取密度矩阵表示
        
        Returns:
            torch.Tensor: 密度矩阵
        """
        state_vector = self.get_state_vector()
        return torch.outer(state_vector, state_vector.conj())
    
    def get_expectation(self, observable: torch.Tensor, 
                       qubit_indices: Optional[List[int]] = None) -> torch.Tensor:
        """计算可观测量期望值
        
        Args:
            observable: 可观测量算符
            qubit_indices: 作用的量子比特索引，如果为None则作用于所有量子比特
            
        Returns:
            torch.Tensor: 期望值
        """
        state_vector = self.get_state_vector()
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算期望值
        # 从态向量中获取子态向量
        full_indices = list(range(self.num_qubits))
        sub_state_vector = state_vector.clone()
        
        # 仅保留指定量子比特的态
        for i in reversed(full_indices):
            if i not in qubit_indices:
                sub_state_vector = sub_state_vector.view(-1, 2).select(1, 0)
        
        # 计算子态向量的期望值
        expectation = torch.vdot(sub_state_vector, 
                                 torch.matmul(observable, sub_state_vector))
        return expectation.real
    
    def get_entanglement_entropy(self, partition: List[int]) -> torch.Tensor:
        """计算纠缠熵
        
        Args:
            partition: 子系统A的量子比特索引
            
        Returns:
            torch.Tensor: 纠缠熵
        """
        # 获取态向量
        state_vector = self.get_state_vector()
        
        # 将态向量重塑为矩阵形式
        dim_A = 2**len(partition)
        dim_B = 2**(self.num_qubits - len(partition))
        state_matrix = state_vector.reshape(dim_A, dim_B)
        
        # 计算约化密度矩阵
        rho_A = torch.matmul(state_matrix, state_matrix.conj().T)
        
        # 计算特征值
        eigenvalues = torch.linalg.eigvalsh(rho_A)
        
        # 计算von Neumann熵
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-10))
        return entropy
    
    def get_fidelity(self, other_state: 'AbstractState') -> torch.Tensor:
        """计算与另一个量子态的保真度
        
        Args:
            other_state: 另一个量子态
            
        Returns:
            torch.Tensor: 保真度
        """
        # 获取两个态的态向量
        state_vector1 = self.get_state_vector()
        state_vector2 = other_state.get_state_vector()
        
        # 计算保真度 |⟨ψ|φ⟩|²
        overlap = torch.vdot(state_vector1, state_vector2)
        fidelity = torch.abs(overlap) ** 2
        return fidelity
    
    def apply_gate(self, gate: 'QuantumGate', qubit_indices: List[int], 
                  params: Optional[torch.Tensor] = None) -> None:
        """应用量子门（待实现）
        """
        pass
    
    def measure(self, qubit_indices: Optional[List[int]] = None, 
               num_shots: int = 1) -> torch.Tensor:
        """测量量子态
        
        Args:
            qubit_indices: 要测量的量子比特索引，如果为None则测量所有量子比特
            num_shots: 测量次数
            
        Returns:
            torch.Tensor: 测量结果，形状为(num_shots, len(qubit_indices))
        """
        # 获取态向量进行测量
        state_vector = self.get_state_vector()
        
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))
        
        # 计算测量概率
        probs = torch.abs(state_vector) ** 2
        
        # 进行测量
        results = []
        for _ in range(num_shots):
            # 采样测量结果
            measured_state = torch.multinomial(probs, 1).item()
            # 将测量结果转换为比特串
            bitstring = format(measured_state, f'0{self.num_qubits}b')
            # 提取指定量子比特的测量结果
            result = [int(bitstring[i]) for i in qubit_indices]
            results.append(result)
        
        return torch.tensor(results, dtype=torch.int64)

