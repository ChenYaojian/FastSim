import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Callable, Optional, Tuple
import math
from abc import ABC, abstractmethod
import json
import re


class Operator(nn.Module):
    """通用算符基类，支持任意量子比特数和索引"""
    
    def __init__(self, num_qubits: int, qubit_indices: List[int], 
                 use_sparse: bool = False, use_decomposed: bool = False,
                 device: torch.device = None):
        super().__init__()
        self.num_qubits = num_qubits
        self.qubit_indices = qubit_indices
        self.use_sparse = use_sparse
        self.use_decomposed = use_decomposed
        self.device = device or torch.device('cpu')
        self.dim = 2 ** num_qubits
        
        # 存储不同表示形式
        self.dense_matrix = None
        self.sparse_matrix = None
        self.terms = nn.ModuleList()  # 算符项列表
        self.coefficient = 1.0  # 系数
        
        # 形状属性，用于兼容性
        self.shape = (self.dim, self.dim)
        
    def add_term(self, term: 'Operator'):
        """添加算符项"""
        self.terms.append(term)
        
    def __add__(self, other):
        """重载加法操作"""
        if not isinstance(other, Operator):
            raise TypeError("Can only add Operator instances")
        
        if self.num_qubits != other.num_qubits:
            raise ValueError("Operators must have the same number of qubits")
        
        result = Operator(self.num_qubits, self.qubit_indices, 
                         self.use_sparse, self.use_decomposed, self.device)
        
        # 复制所有项
        for term in self.terms:
            result.add_term(term)
        for term in other.terms:
            result.add_term(term)
            
        return result
    
    def __mul__(self, scalar):
        """重载标量乘法"""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Can only multiply by scalar")
        
        result = Operator(self.num_qubits, self.qubit_indices,
                         self.use_sparse, self.use_decomposed, self.device)
        
        # 复制所有项，乘以标量
        for term in self.terms:
            new_term = term * scalar
            result.add_term(new_term)
            
        return result
    
    def __rmul__(self, scalar):
        """重载右乘标量"""
        return self.__mul__(scalar)
        
    def build_dense_matrix(self):
        """构建密集矩阵表示"""
        if self.dense_matrix is not None:
            return self.dense_matrix
            
        H = torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)
        
        # 如果有terms，使用terms
        if len(self.terms) > 0:
            for term in self.terms:
                term_matrix = term.get_matrix()
                H += term_matrix
        # 否则使用自身的矩阵表示
        else:
            H = self._get_matrix()
            
        self.dense_matrix = H
        return H
        
    def build_sparse_matrix(self):
        """构建稀疏矩阵表示"""
        if self.sparse_matrix is not None:
            return self.sparse_matrix
            
        try:
            import scipy.sparse as sp
            from scipy.sparse import csr_matrix
        except ImportError:
            print("Warning: scipy not available, falling back to dense matrix")
            return self.build_dense_matrix()
        
        # 对于大系统，使用黑盒方式
        if self.num_qubits > 10:
            print(f"Large system detected ({self.num_qubits} qubits), using black-box matrix-vector multiplication")
            return self
        
        # 构建稀疏矩阵
        dense_matrix = self.build_dense_matrix()
        
        # 转换为稀疏矩阵
        dense_np = dense_matrix.detach().cpu().numpy()
        self.sparse_matrix = csr_matrix(dense_np)
        
        return self.sparse_matrix
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，计算 O @ state"""
        # 确保state是2D张量 [batch_size, dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加batch维度
        
        if self.use_decomposed and len(self.terms) > 0:
            # 使用分解形式，分别计算每个term
            result = torch.zeros_like(state)
            for term in self.terms:
                # 确保PauliStringOperator使用其分解形式
                if hasattr(term, 'use_decomposed') and term.use_decomposed:
                    result += term(state)
                else:
                    # 如果term不是分解形式，强制使用分解形式
                    term.use_decomposed = True
                    result += term(state)
            return result
        elif self.use_sparse:
            # 使用稀疏形式
            if self.sparse_matrix is None:
                self.build_sparse_matrix()
            
            if hasattr(self.sparse_matrix, '__matmul__'):
                # 黑盒形式
                return self.sparse_matrix @ state
            else:
                # 稀疏矩阵形式
                state_np = state.detach().cpu().numpy()
                result_np = self.sparse_matrix @ state_np
                return torch.tensor(result_np, dtype=torch.complex64, device=self.device)
        else:
            # 使用密集矩阵形式
            if self.dense_matrix is None:
                self.build_dense_matrix()
            return torch.matmul(self.dense_matrix, state.transpose(0, 1)).transpose(0, 1)
    
    def expectation(self, state: torch.Tensor) -> torch.Tensor:
        """计算期望值 <ψ|O|ψ>"""
        O_state = self.forward(state)
        return torch.sum(state.conj() * O_state, dim=-1)
    
    def __matmul__(self, other):
        """支持矩阵乘法语法 O @ v"""
        return self.forward(other)
    
    def get_matrix(self):
        """获取矩阵表示"""
        if self.use_sparse:
            return self.build_sparse_matrix()
        else:
            return self.build_dense_matrix()
    
    def _get_matrix(self):
        """基类的默认矩阵获取方法"""
        # 如果没有terms，返回零矩阵
        if len(self.terms) == 0:
            return torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)
        
        # 否则返回terms的矩阵表示
        H = torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)
        for term in self.terms:
            H += term.get_matrix()
        return H
    
    def is_hermitian(self, tolerance: float = 1e-10) -> bool:
        """检查算符是否为厄米算符"""
        matrix = self.get_matrix()
        if hasattr(matrix, 'to_dense'):
            matrix = matrix.to_dense()
        return torch.allclose(matrix, matrix.conj().transpose(-2, -1), atol=tolerance)


class PauliStringOperator(Operator):
    """泡利字符串算符，支持高效的分解形式计算"""
    
    def __init__(self, pauli_string: str, qubit_indices: List[int], 
                 coefficient: complex = 1.0, num_qubits: int = None,
                 use_sparse: bool = False, use_decomposed: bool = True,
                 device: torch.device = None):
        # 如果没有指定num_qubits，从qubit_indices推断
        if num_qubits is None:
            num_qubits = max(qubit_indices) + 1 if qubit_indices else 0
            
        super().__init__(num_qubits, qubit_indices, use_sparse, use_decomposed, device)
        self.pauli_string = pauli_string
        self.coefficient = coefficient
        
        # 泡利矩阵（压缩表示）
        self.pauli_matrices = {
            'I': torch.eye(2, dtype=torch.complex64, device=self.device),
            'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.device),
            'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.device),
            'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.device)
        }
        
    def _get_matrix(self):
        """获取泡利算符的完整矩阵表示"""
        # 构建完整的张量积矩阵
        matrices = [torch.eye(2, dtype=torch.complex64, device=self.device)] * self.num_qubits
        
        # 将每个泡利算符的矩阵放到正确的位置
        for i, (pauli, qubit_idx) in enumerate(zip(self.pauli_string, self.qubit_indices)):
            if i < len(self.qubit_indices):
                matrices[qubit_idx] = self.pauli_matrices[pauli]
        
        # 计算张量积
        result_matrix = matrices[0]
        for matrix in matrices[1:]:
            result_matrix = torch.kron(result_matrix, matrix)
        
        return self.coefficient * result_matrix
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """高效的泡利算符应用（分解形式）"""
        if self.use_decomposed:
            return self._forward_decomposed(state)
        else:
            return super().forward(state)
    
    def _forward_decomposed(self, state: torch.Tensor) -> torch.Tensor:
        """高效的分解形式泡利算符应用 - 延迟置换和相位操作"""
        # 确保state是2D张量 [batch_size, dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        total_qubits = int(math.log2(state.size(1)))
        
        # 对每个样本分别处理
        new_states = []
        for i in range(batch_size):
            # 将单个样本的state重塑为1D向量
            single_state = state[i]
            
            # 使用优化的泡利串合并处理
            result_state = self._apply_pauli_sequence_optimized(single_state, total_qubits)
            new_states.append(result_state)
        
        # 将处理后的状态堆叠成batch并应用系数
        result = torch.stack(new_states)
        return self.coefficient * result
    
    def _apply_pauli_sequence_optimized(self, state: torch.Tensor, total_qubits: int) -> torch.Tensor:
        """优化的泡利算符序列应用 - 合并处理方案"""
        import numpy as np
        
        # 转换为numpy数组以提高性能
        state_np = state.detach().cpu().numpy()
        num_states = len(state_np)
        
        # 1. 为每个qubit维护翻转标志和相位
        transposition_flags = [False] * total_qubits  # 翻转标志
        phase_factors = [(1.0, 1.0)] * total_qubits  # 相位因子 (p0, p1)
        global_phase = 1.0  # 全局相位
        
        # 2. 逐个处理泡利算符，累积翻转和相位信息
        for pauli, qubit_idx in zip(self.pauli_string, self.qubit_indices):
            if qubit_idx >= total_qubits:
                continue
                
            p0, p1 = phase_factors[qubit_idx]
            
            if pauli == 'X':
                # X算符：翻转比特，交换相位分量
                transposition_flags[qubit_idx] = not transposition_flags[qubit_idx]
                phase_factors[qubit_idx] = (p1, p0)
                
            elif pauli == 'Y':
                # Y算符：翻转比特，交换相位分量并乘以i/-i
                transposition_flags[qubit_idx] = not transposition_flags[qubit_idx]
                phase_factors[qubit_idx] = (-1j * p1, 1j * p0)
                
            elif pauli == 'Z':
                # Z算符：只改变相位，第二个分量乘以-1
                phase_factors[qubit_idx] = (p0, -p1)
                
            # I算符：不改变任何东西
        
        # 3. 统一进行置换操作
        # 计算基于翻转标志的掩码（注意：需要反转比特顺序以匹配张量积顺序）
        mask = sum(flag << (total_qubits - 1 - i) for i, flag in enumerate(transposition_flags))
        
        # 生成置换索引
        perm_indices = np.arange(num_states) ^ mask
        
        # 应用置换
        new_state_vector = state_np[perm_indices]
        
        # 4. 统一进行相位操作
        indices = np.arange(num_states)
        total_phase = np.ones(num_states, dtype=np.complex128)
        
        for i in range(total_qubits):
            p0, p1 = phase_factors[i]
            
            # 优化1：如果p0 == p1，则只需将相位乘到global_phase上
            if p0 == p1:
                if p0 != 1.0:  # 优化2：如果相位为1，跳过
                    global_phase *= p0
                continue
            
            # 优化2：如果某个分量的相位为1，则跳过该分量
            if p0 == 1.0 and p1 == 1.0:
                continue
            
            # 提取第i个比特的值（注意：需要反转比特顺序）
            bit_values = (indices >> (total_qubits - 1 - i)) & 1
            
            # 计算相位贡献：bit=0时用p0，bit=1时用p1
            phase_contrib = p0 + (p1 - p0) * bit_values
            
            # 累积相位
            total_phase *= phase_contrib
        
        # 应用全局相位和总相位
        final_phase = global_phase * total_phase
        
        # 应用相位到状态向量
        new_state_vector = final_phase * new_state_vector
        
        # 转换回torch张量
        return torch.tensor(new_state_vector, dtype=torch.complex64, device=state.device)


class Hamiltonian(Operator):
    """哈密顿量类，继承自Operator"""
    
    def __init__(self, num_qubits: int, use_sparse: bool = False, 
                 use_decomposed: bool = False, device: torch.device = None):
        super().__init__(num_qubits, [], use_sparse, use_decomposed, device)
        
    def add_pauli_term(self, pauli_string: str, qubit_indices: List[int], coefficient: complex = 1.0):
        """添加泡利项"""
        pauli_op = PauliStringOperator(pauli_string, qubit_indices, coefficient, 
                                      self.num_qubits, self.use_sparse, self.use_decomposed, self.device)
        self.add_term(pauli_op)
        
    def _get_matrix(self):
        """哈密顿量的矩阵表示"""
        if len(self.terms) == 0:
            return torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)
        
        H = torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)
        for term in self.terms:
            H += term.get_matrix()
        return H
    
    @classmethod
    def from_string(cls, hamiltonian_str: str, num_qubits: int, use_sparse: bool = False, 
                   use_decomposed: bool = False, device: torch.device = None):
        """从字符串构建哈密顿量，以加法为分隔符分成不同的term"""
        H = cls(num_qubits, use_sparse, use_decomposed, device)
        
        # 解析哈密顿量字符串
        # 格式示例: "J*XX[0,1] + J*YY[0,1] + J*ZZ[0,1] + h*Z[0] + h*Z[1]"
        
        # 按加法分割（支持+号和空格分隔）
        terms = re.split(r'\s*\+\s*|\s+', hamiltonian_str.strip())
        terms = [term for term in terms if term.strip()]  # 移除空项
        
        for term_str in terms:
            # 解析项: "J*XX[0,1]" -> coefficient="J", pauli="XX", indices=[0,1]
            match = re.match(r'([^[]*)\*([A-Z]+)\[([^\]]+)\]', term_str.strip())
            if match:
                coefficient_str, pauli_str, indices_str = match.groups()
                
                # 解析系数
                coefficient = 1.0
                if coefficient_str.strip():
                    try:
                        coefficient = complex(coefficient_str.strip())
                    except ValueError:
                        # 如果无法解析为复数，使用默认值1.0
                        coefficient = 1.0
                
                # 解析量子比特索引
                indices = [int(x.strip()) for x in indices_str.split(',')]
                
                # 添加泡利项（每个项作为一个独立的term）
                H.add_pauli_term(pauli_str, indices, coefficient)
        
        return H


# 具体的哈密顿量模型类
class HeisenbergHamiltonian(Hamiltonian):
    """海森堡模型哈密顿量"""
    
    def __init__(self, num_qubits: int, J: float = 1.0, h: float = 0.0,
                 use_sparse: bool = False, use_decomposed: bool = False, 
                 device: torch.device = None):
        super().__init__(num_qubits, use_sparse, use_decomposed, device)
        
        # 添加相互作用项 J * (σx⊗σx + σy⊗σy + σz⊗σz)
        for i in range(num_qubits - 1):
            # σx⊗σx 项
            self.add_pauli_term('XX', [i, i+1], J)
            # σy⊗σy 项
            self.add_pauli_term('YY', [i, i+1], J)
            # σz⊗σz 项
            self.add_pauli_term('ZZ', [i, i+1], J)
        
        # 添加外场项 h * σz
        for i in range(num_qubits):
            self.add_pauli_term('Z', [i], h)


class IsingHamiltonian(Hamiltonian):
    """一维横场Ising模型哈密顿量"""
    
    def __init__(self, num_qubits: int, J: float = 1.0, h: float = 0.0,
                 use_sparse: bool = False, use_decomposed: bool = False, 
                 device: torch.device = None):
        super().__init__(num_qubits, use_sparse, use_decomposed, device)
        
        # 添加相互作用项 -J * σz⊗σz
        for i in range(num_qubits - 1):
            self.add_pauli_term('ZZ', [i, i+1], -J)
        
        # 添加横场项 -h * σx
        for i in range(num_qubits):
            self.add_pauli_term('X', [i], -h)


class HubbardHamiltonian(Hamiltonian):
    """Hubbard模型哈密顿量"""
    
    def __init__(self, num_qubits: int, t: float = 1.0, U: float = 4.0,
                 use_sparse: bool = False, use_decomposed: bool = False, 
                 device: torch.device = None):
        super().__init__(num_qubits, use_sparse, use_decomposed, device)
        
        num_sites = num_qubits // 2
        
        # 添加相互作用项 U * n↑ * n↓ (双占据能量)
        for site in range(num_sites):
            q1, q2 = 2*site, 2*site+1
            if q2 < num_qubits:
                # n↑ * n↓ = (I + Z↑)/2 * (I + Z↓)/2 = (I + Z↑ + Z↓ + Z↑Z↓)/4
                # 只保留 Z↑Z↓ 项，因为其他项是常数
                self.add_pauli_term('ZZ', [q1, q2], U/4)
        
        # 添加跳跃项（最近邻）
        for site in range(num_sites - 1):
            q1_up, q1_down = 2*site, 2*site+1
            q2_up, q2_down = 2*(site+1), 2*(site+1)+1
            
            if q2_down < num_qubits:
                # 上自旋跳跃：t * (c↑_i† c↑_{i+1} + h.c.)
                # c↑_i† c↑_{i+1} = (X↑_i - iY↑_i)(X↑_{i+1} + iY↑_{i+1})/4
                # 展开后：XX + YY + i(XY - YX)
                # 由于哈密顿量必须是厄米的，我们只保留 XX + YY 项
                self.add_pauli_term('XX', [q1_up, q2_up], t/4)
                self.add_pauli_term('YY', [q1_up, q2_up], t/4)
                
                # 下自旋跳跃：t * (c↓_i† c↓_{i+1} + h.c.)
                self.add_pauli_term('XX', [q1_down, q2_down], t/4)
                self.add_pauli_term('YY', [q1_down, q2_down], t/4)


class Quasi1DAFMHamiltonian(Hamiltonian):
    """准一维反铁磁模型哈密顿量"""
    
    def __init__(self, num_qubits: int, J_perp: float = 0.5, J_parallel: float = 1.0, h: float = 0.0,
                 use_sparse: bool = False, use_decomposed: bool = False, 
                 device: torch.device = None):
        super().__init__(num_qubits, use_sparse, use_decomposed, device)
        
        # 添加相互作用项
        for i in range(num_qubits - 1):
            # 横向相互作用：J⊥ * (σx⊗σx + σy⊗σy)
            self.add_pauli_term('XX', [i, i+1], J_perp)
            self.add_pauli_term('YY', [i, i+1], J_perp)
            
            # 纵向相互作用：J∥ * σz⊗σz
            self.add_pauli_term('ZZ', [i, i+1], J_parallel)
        
        # 添加外场项：h * σz
        for i in range(num_qubits):
            self.add_pauli_term('Z', [i], h)


class Paper4NHeisenbergHamiltonian(Hamiltonian):
    """arXiv:2007.10917v2 结构的 4*N 海森堡哈密顿量"""
    
    def __init__(self, N: int, use_sparse: bool = False, use_decomposed: bool = False, 
                 device: torch.device = None):
        num_qubits = 4 * N
        super().__init__(num_qubits, use_sparse, use_decomposed, device)
        
        # 子系统内部边
        for block in range(N):
            offset = block * 4
            edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
            for (a, b) in edges:
                q1 = offset + a
                q2 = offset + b
                # 添加海森堡相互作用
                self.add_pauli_term('XX', [q1, q2], 1.0)
                self.add_pauli_term('YY', [q1, q2], 1.0)
                self.add_pauli_term('ZZ', [q1, q2], 1.0)
        
        # 子系统之间的2-0连接
        for block in range(N-1):
            q1 = block * 4 + 2
            q2 = (block + 1) * 4 + 0
            # 添加海森堡相互作用（负号）
            self.add_pauli_term('XX', [q1, q2], -1.0)
            self.add_pauli_term('YY', [q1, q2], -1.0)
            self.add_pauli_term('ZZ', [q1, q2], -1.0)


def create_hamiltonian(hamiltonian_type: str, **kwargs) -> Hamiltonian:
    """工厂函数，创建指定类型的哈密顿量
    
    支持两种方式：
    1. 使用预定义类型：'heisenberg', 'ising', 'hubbard', 'quasi_1d_afm', 'paper_4n_heisenberg'
    2. 使用字符串表达式：如 "-1.0*ZZ[0,1] -0.5*X[0] -0.5*X[1]"
    """
    hamiltonian_classes = {
        'heisenberg': HeisenbergHamiltonian,
        'ising': IsingHamiltonian,
        'hubbard': HubbardHamiltonian,
        'quasi_1d_afm': Quasi1DAFMHamiltonian,
        'paper_4n_heisenberg': Paper4NHeisenbergHamiltonian
    }
    
    # 检查是否是预定义类型
    if hamiltonian_type in hamiltonian_classes:
        return hamiltonian_classes[hamiltonian_type](**kwargs)
    
    # 检查是否是字符串表达式（包含*和[）
    if '*' in hamiltonian_type and '[' in hamiltonian_type:
        # 从kwargs中获取num_qubits，如果没有则尝试推断
        num_qubits = kwargs.get('num_qubits')
        if num_qubits is None:
            # 尝试从字符串中推断量子比特数
            indices = re.findall(r'\[([^\]]+)\]', hamiltonian_type)
            max_index = 0
            for index_str in indices:
                indices_list = [int(x.strip()) for x in index_str.split(',')]
                max_index = max(max_index, max(indices_list))
            num_qubits = max_index + 1
        
        use_sparse = kwargs.get('use_sparse', False)
        use_decomposed = kwargs.get('use_decomposed', False)
        device = kwargs.get('device', None)
        
        return Hamiltonian.from_string(hamiltonian_type, num_qubits, use_sparse, use_decomposed, device)
    
    raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")