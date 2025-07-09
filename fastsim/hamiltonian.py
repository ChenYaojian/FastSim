import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Callable, Optional


def create_random_hamiltonian(num_qubits: int, device: torch.device = None) -> torch.Tensor:
    """创建随机哈密顿量（用于测试）"""
    dim = 2 ** num_qubits
    # 创建随机厄米矩阵
    H = torch.randn(dim, dim, dtype=torch.complex64, device=device)
    H = (H + H.conj().transpose(0, 1)) / 2  # 确保厄米性
    return H


def create_heisenberg_hamiltonian(num_qubits: int, J: float = 1.0, 
                                 h: float = 0.0, device: torch.device = None):
    """创建海森堡模型哈密顿量，大系统使用黑盒矩阵-向量乘法"""
    dim = 2 ** num_qubits
    
    # 对于大系统（>10比特），使用黑盒方式
    if num_qubits > 10:
        print(f"Large system detected ({num_qubits} qubits), using black-box matrix-vector multiplication")
        return HeisenbergHamiltonianOperator(num_qubits, J, h, device)
    
    # 小系统使用密集矩阵
    print(f"Small system ({num_qubits} qubits), using dense matrix")
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    
    # Pauli矩阵
    I = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    
    # 添加相互作用项 J * (σx⊗σx + σy⊗σy + σz⊗σz)
    for i in range(num_qubits - 1):
        # σx⊗σx 项
        ops = [I] * num_qubits
        ops[i] = X
        ops[i+1] = X
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J * term
        
        # σy⊗σy 项
        ops = [I] * num_qubits
        ops[i] = Y
        ops[i+1] = Y
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J * term
        
        # σz⊗σz 项
        ops = [I] * num_qubits
        ops[i] = Z
        ops[i+1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J * term
    
    # 添加外场项 h * σz
    for i in range(num_qubits):
        ops = [I] * num_qubits
        ops[i] = Z
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += h * term
    
    return H


class HeisenbergHamiltonianOperator:
    """
    海森堡模型哈密顿量-向量乘法黑盒，不存储矩阵，只实现 H @ v
    """
    def __init__(self, num_qubits, J=1.0, h=0.0, device=None):
        self.num_qubits = num_qubits
        self.J = J
        self.h = h
        self.device = device
        self.dim = 2 ** num_qubits
        self.shape = (self.dim, self.dim)  # 兼容主流程打印

    def __matmul__(self, v):
        if isinstance(v, torch.Tensor):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        
        # 确保输入是2D数组
        if v_np.ndim == 1:
            v_np = v_np.reshape(1, -1)  # [1, dim]
        
        batch, dim = v_np.shape
        if dim != self.dim:
            raise ValueError(f"Vector dimension {dim} does not match hamiltonian dimension {self.dim}")
        
        out = np.zeros_like(v_np, dtype=np.complex64)
        
        for b in range(batch):
            for state in range(dim):
                val = 0.0
                
                # 相互作用项：J * (σx⊗σx + σy⊗σy + σz⊗σz)
                for i in range(self.num_qubits - 1):
                    # σx⊗σx 项
                    flipped_state = state ^ ((1 << i) | (1 << (i+1)))
                    val += self.J * v_np[b, flipped_state]
                    
                    # σy⊗σy 项（带相位）
                    # σy = [[0, -i], [i, 0]]，所以相位因子是±i
                    phase = 1.0j
                    if ((state >> i) & 1) != ((state >> (i+1)) & 1):
                        phase = -1.0j
                    val += self.J * phase * v_np[b, flipped_state]
                    
                    # σz⊗σz 项
                    # 对角项，根据第i和i+1位的值确定符号
                    sign = 1.0
                    if ((state >> i) & 1) == ((state >> (i+1)) & 1):
                        sign = -1.0
                    val += self.J * sign * v_np[b, state]
                
                # 外场项：h * σz
                for i in range(self.num_qubits):
                    sign = 1.0 if ((state >> i) & 1) == 0 else -1.0
                    val += self.h * sign * v_np[b, state]
                
                out[b, state] = val
        
        out_tensor = torch.tensor(out, dtype=torch.complex64, device=self.device)
        if v.ndim == 1:
            return out_tensor[0]
        return out_tensor


def create_ising_hamiltonian(num_qubits: int, J: float = 1.0, h: float = 0.0, device: torch.device = None) -> torch.Tensor:
    """创建一维横场Ising模型哈密顿量: H = -J Σ σz_i σz_{i+1} - h Σ σx_i"""
    dim = 2 ** num_qubits
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    # Pauli matrices
    I = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    # -J Σ σz_i σz_{i+1}
    for i in range(num_qubits - 1):
        ops = [I] * num_qubits
        ops[i] = Z
        ops[i+1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H -= J * term
    # -h Σ σx_i
    for i in range(num_qubits):
        ops = [I] * num_qubits
        ops[i] = X
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H -= h * term
    return H


def create_sparse_hubbard_hamiltonian(num_qubits: int, t: float = 1.0, U: float = 4.0, device: torch.device = None) -> torch.Tensor:
    """创建稀疏Hubbard模型哈密顿量（适用于大系统）"""
    try:
        import scipy.sparse as sp
        from scipy.sparse import csr_matrix
    except ImportError:
        print("Warning: scipy not available, falling back to dense matrix")
        return create_hubbard_hamiltonian(num_qubits, t, U, device)
    
    dim = 2 ** num_qubits
    print(f"Creating sparse Hubbard hamiltonian with dimension {dim}")
    
    # 计算格点数（每2个量子比特表示1个格点）
    num_sites = num_qubits // 2
    if num_qubits % 2 != 0:
        print(f"Warning: Odd number of qubits ({num_qubits}), using {num_sites} sites")
    
    # 使用列表存储非零元素
    rows, cols, values = [], [], []
    
    # 添加相互作用项 U * n↑ * n↓ (双占据能量)
    for site in range(num_sites):
        q1, q2 = 2*site, 2*site+1
        if q2 < num_qubits:
            # 找到所有双占据态 |↑↓⟩
            for state in range(dim):
                # 检查第q1和q2位是否都是1
                if (state >> q1) & 1 and (state >> q2) & 1:
                    rows.append(state)
                    cols.append(state)
                    values.append(U)
    
    # 添加跳跃项（简化版本，只考虑最近邻）
    for site in range(num_sites - 1):
        q1_up, q1_down = 2*site, 2*site+1
        q2_up, q2_down = 2*(site+1), 2*(site+1)+1
        
        if q2_down < num_qubits:
            # 上自旋跳跃
            for state in range(dim):
                # 检查是否可以跳跃
                if not ((state >> q1_up) & 1) and ((state >> q2_up) & 1):
                    # 创建新态
                    new_state = state | (1 << q1_up)  # 在q1_up位置1
                    new_state = new_state & ~(1 << q2_up)  # 在q2_up位置0
                    rows.append(state)
                    cols.append(new_state)
                    values.append(t)
                    # 厄米共轭
                    rows.append(new_state)
                    cols.append(state)
                    values.append(t)
            
            # 下自旋跳跃
            for state in range(dim):
                if not ((state >> q1_down) & 1) and ((state >> q2_down) & 1):
                    new_state = state | (1 << q1_down)
                    new_state = new_state & ~(1 << q2_down)
                    rows.append(state)
                    cols.append(new_state)
                    values.append(t)
                    rows.append(new_state)
                    cols.append(state)
                    values.append(t)
    
    # 创建稀疏矩阵
    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(dim, dim), dtype=np.complex64)
    
    # 对于大系统，我们使用对角近似
    if dim > 10000:  # 对于非常大的系统，只保留主要项
        print("Large system detected, using diagonal approximation")
        # 只保留对角线和最近的几个非对角元素
        diagonal = sparse_matrix.diagonal()
        # 创建一个简化的哈密顿量，只包含对角项
        hamiltonian = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
        hamiltonian.fill_diagonal_(torch.tensor(diagonal, dtype=torch.complex64, device=device))
        return hamiltonian
    else:
        # 转换为密集矩阵
        dense_matrix = sparse_matrix.toarray()
        return torch.tensor(dense_matrix, dtype=torch.complex64, device=device)


class SparseHamiltonian:
    """稀疏哈密顿量类，用于处理大系统"""
    
    def __init__(self, sparse_matrix, device=None):
        self.sparse_matrix = sparse_matrix
        self.device = device
        self.dim = sparse_matrix.shape[0]
    
    def __matmul__(self, other):
        """矩阵乘法，支持稀疏矩阵与向量的乘法"""
        if isinstance(other, torch.Tensor):
            # 转换为numpy进行稀疏矩阵乘法
            other_np = other.detach().cpu().numpy()
            result_np = self.sparse_matrix @ other_np
            return torch.tensor(result_np, dtype=torch.complex64, device=self.device)
        else:
            return self.sparse_matrix @ other


def create_sparse_hubbard_hamiltonian_v2(num_qubits: int, t: float = 1.0, U: float = 4.0, device: torch.device = None):
    """创建稀疏Hubbard模型哈密顿量（版本2，返回稀疏矩阵对象）"""
    try:
        import scipy.sparse as sp
        from scipy.sparse import csr_matrix
    except ImportError:
        print("Warning: scipy not available, falling back to dense matrix")
        return create_hubbard_hamiltonian(num_qubits, t, U, device)
    
    dim = 2 ** num_qubits
    print(f"Creating sparse Hubbard hamiltonian with dimension {dim}")
    
    # 计算格点数（每2个量子比特表示1个格点）
    num_sites = num_qubits // 2
    if num_qubits % 2 != 0:
        print(f"Warning: Odd number of qubits ({num_qubits}), using {num_sites} sites")
    
    # 使用列表存储非零元素
    rows, cols, values = [], [], []
    
    # 添加相互作用项 U * n↑ * n↓ (双占据能量)
    for site in range(num_sites):
        q1, q2 = 2*site, 2*site+1
        if q2 < num_qubits:
            # 找到所有双占据态 |↑↓⟩
            for state in range(dim):
                # 检查第q1和q2位是否都是1
                if (state >> q1) & 1 and (state >> q2) & 1:
                    rows.append(state)
                    cols.append(state)
                    values.append(U)
    
    # 添加跳跃项（简化版本，只考虑最近邻）
    for site in range(num_sites - 1):
        q1_up, q1_down = 2*site, 2*site+1
        q2_up, q2_down = 2*(site+1), 2*(site+1)+1
        
        if q2_down < num_qubits:
            # 上自旋跳跃
            for state in range(dim):
                # 检查是否可以跳跃
                if not ((state >> q1_up) & 1) and ((state >> q2_up) & 1):
                    # 创建新态
                    new_state = state | (1 << q1_up)  # 在q1_up位置1
                    new_state = new_state & ~(1 << q2_up)  # 在q2_up位置0
                    rows.append(state)
                    cols.append(new_state)
                    values.append(t)
                    # 厄米共轭
                    rows.append(new_state)
                    cols.append(state)
                    values.append(t)
            
            # 下自旋跳跃
            for state in range(dim):
                if not ((state >> q1_down) & 1) and ((state >> q2_down) & 1):
                    new_state = state | (1 << q1_down)
                    new_state = new_state & ~(1 << q2_down)
                    rows.append(state)
                    cols.append(new_state)
                    values.append(t)
                    rows.append(new_state)
                    cols.append(state)
                    values.append(t)
    
    # 创建稀疏矩阵
    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(dim, dim), dtype=np.complex64)
    
    # 返回稀疏矩阵对象
    return SparseHamiltonian(sparse_matrix, device)


class HubbardHamiltonianOperator:
    """
    哈密顿量-向量乘法黑盒，不存储矩阵，只实现 H @ v
    """
    def __init__(self, num_qubits, t=1.0, U=4.0, device=None):
        self.num_qubits = num_qubits
        self.t = t
        self.U = U
        self.device = device
        self.dim = 2 ** num_qubits
        self.num_sites = num_qubits // 2
        self.shape = (self.dim, self.dim)  # 兼容主流程打印

    def __matmul__(self, v):
        if isinstance(v, torch.Tensor):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        if v_np.ndim == 1:
            v_np = v_np[None, :]  # [1, dim]
        batch, dim = v_np.shape
        out = np.zeros_like(v_np, dtype=np.complex64)
        for b in range(batch):
            for state in range(dim):
                val = 0.0
                # U项：双占据
                for site in range(self.num_sites):
                    q1, q2 = 2*site, 2*site+1
                    if q2 < self.num_qubits:
                        if ((state >> q1) & 1) and ((state >> q2) & 1):
                            val += self.U
                # 跳跃项（只考虑最近邻）
                for site in range(self.num_sites - 1):
                    q1_up, q1_down = 2*site, 2*site+1
                    q2_up, q2_down = 2*(site+1), 2*(site+1)+1
                    # 上自旋跳跃
                    if q2_up < self.num_qubits:
                        if not ((state >> q1_up) & 1) and ((state >> q2_up) & 1):
                            new_state = state | (1 << q1_up)
                            new_state = new_state & ~(1 << q2_up)
                            val += self.t * v_np[b, new_state]
                        if not ((state >> q2_up) & 1) and ((state >> q1_up) & 1):
                            new_state = state | (1 << q2_up)
                            new_state = new_state & ~(1 << q1_up)
                            val += self.t * v_np[b, new_state]
                    # 下自旋跳跃
                    if q2_down < self.num_qubits:
                        if not ((state >> q1_down) & 1) and ((state >> q2_down) & 1):
                            new_state = state | (1 << q1_down)
                            new_state = new_state & ~(1 << q2_down)
                            val += self.t * v_np[b, new_state]
                        if not ((state >> q2_down) & 1) and ((state >> q1_down) & 1):
                            new_state = state | (1 << q2_down)
                            new_state = new_state & ~(1 << q1_down)
                            val += self.t * v_np[b, new_state]
                out[b, state] = val
        out_tensor = torch.tensor(out, dtype=torch.complex64, device=self.device)
        if v.ndim == 1:
            return out_tensor[0]
        return out_tensor


def create_hubbard_hamiltonian(num_qubits: int, t: float = 1.0, U: float = 4.0, device: torch.device = None):
    """只返回HubbardHamiltonianOperator黑盒，不存储矩阵"""
    return HubbardHamiltonianOperator(num_qubits, t, U, device)


class Quasi1DAFMHamiltonianOperator:
    """
    准一维反铁磁模型哈密顿量-向量乘法黑盒，不存储矩阵，只实现 H @ v
    哈密顿量：H = J⊥ Σ(σx_i⊗σx_{i+1} + σy_i⊗σy_{i+1}) + J∥ Σσz_i⊗σz_{i+1} + h Σσz_i
    """
    def __init__(self, num_qubits, J_perp=0.5, J_parallel=1.0, h=0.0, device=None):
        self.num_qubits = num_qubits
        self.J_perp = J_perp  # 横向相互作用强度
        self.J_parallel = J_parallel  # 纵向相互作用强度
        self.h = h  # 外场强度
        self.device = device
        self.dim = 2 ** num_qubits
        self.shape = (self.dim, self.dim)  # 兼容主流程打印

    def __matmul__(self, v):
        if isinstance(v, torch.Tensor):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        
        # 确保输入是2D数组
        if v_np.ndim == 1:
            v_np = v_np.reshape(1, -1)  # [1, dim]
        
        batch, dim = v_np.shape
        if dim != self.dim:
            raise ValueError(f"Vector dimension {dim} does not match hamiltonian dimension {self.dim}")
        
        out = np.zeros_like(v_np, dtype=np.complex64)
        
        for b in range(batch):
            for state in range(dim):
                val = 0.0
                
                # 相互作用项
                for i in range(self.num_qubits - 1):
                    # 横向相互作用：J⊥ * (σx⊗σx + σy⊗σy)
                    # σx⊗σx 项
                    flipped_state = state ^ ((1 << i) | (1 << (i+1)))
                    val += self.J_perp * v_np[b, flipped_state]
                    
                    # σy⊗σy 项（带相位）
                    # σy = [[0, -i], [i, 0]]，所以相位因子是±i
                    phase = 1.0j
                    if ((state >> i) & 1) != ((state >> (i+1)) & 1):
                        phase = -1.0j
                    val += self.J_perp * phase * v_np[b, flipped_state]
                    
                    # 纵向相互作用：J∥ * σz⊗σz
                    sign = 1.0
                    if ((state >> i) & 1) == ((state >> (i+1)) & 1):
                        sign = -1.0
                    val += self.J_parallel * sign * v_np[b, state]
                
                # 外场项：h * σz
                for i in range(self.num_qubits):
                    sign = 1.0 if ((state >> i) & 1) == 0 else -1.0
                    val += self.h * sign * v_np[b, state]
                
                out[b, state] = val
        
        out_tensor = torch.tensor(out, dtype=torch.complex64, device=self.device)
        if v.ndim == 1:
            return out_tensor[0]
        return out_tensor


def create_quasi_1d_afm_hamiltonian(num_qubits: int, J_perp: float = 0.5, 
                                   J_parallel: float = 1.0, h: float = 0.0, 
                                   device: torch.device = None):
    """
    创建准一维反铁磁模型哈密顿量
    
    Args:
        num_qubits: 量子比特数
        J_perp: 横向相互作用强度（σx⊗σx + σy⊗σy项）
        J_parallel: 纵向相互作用强度（σz⊗σz项）
        h: 外场强度
        device: 计算设备
        
    Returns:
        哈密顿量（密集矩阵或黑盒操作符）
    """
    dim = 2 ** num_qubits
    
    # 对于大系统（>10比特），使用黑盒方式
    if num_qubits > 10:
        print(f"Large system detected ({num_qubits} qubits), using black-box matrix-vector multiplication")
        return Quasi1DAFMHamiltonianOperator(num_qubits, J_perp, J_parallel, h, device)
    
    # 小系统使用密集矩阵
    print(f"Small system ({num_qubits} qubits), using dense matrix")
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    
    # Pauli矩阵
    I = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    
    # 添加相互作用项
    for i in range(num_qubits - 1):
        # 横向相互作用：J⊥ * (σx⊗σx + σy⊗σy)
        # σx⊗σx 项
        ops = [I] * num_qubits
        ops[i] = X
        ops[i+1] = X
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J_perp * term
        
        # σy⊗σy 项
        ops = [I] * num_qubits
        ops[i] = Y
        ops[i+1] = Y
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J_perp * term
        
        # 纵向相互作用：J∥ * σz⊗σz
        ops = [I] * num_qubits
        ops[i] = Z
        ops[i+1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += J_parallel * term
    
    # 添加外场项：h * σz
    for i in range(num_qubits):
        ops = [I] * num_qubits
        ops[i] = Z
        term = ops[0]
        for op in ops[1:]:
            term = torch.kron(term, op)
        H += h * term
    
    return H


def create_2d_heisenberg_hamiltonian(Lx: int, Ly: int, J: float = 1.0, 
                                    h: float = 0.0, device: torch.device = None):
    """
    创建2D方形格子海森堡模型哈密顿量
    
    Args:
        Lx: x方向格子大小
        Ly: y方向格子大小  
        J: 相互作用强度
        h: 外场强度
        device: 计算设备
        
    Returns:
        哈密顿量（密集矩阵或黑盒操作符）
    """
    num_qubits = Lx * Ly
    
    # 对于大系统（>10比特），使用黑盒方式
    if num_qubits > 10:
        print(f"Large 2D system detected ({Lx}×{Ly} = {num_qubits} qubits), using black-box matrix-vector multiplication")
        return Heisenberg2DHamiltonianOperator(Lx, Ly, J, h, device)
    
    # 小系统使用密集矩阵
    print(f"Small 2D system ({Lx}×{Ly} = {num_qubits} qubits), using dense matrix")
    dim = 2 ** num_qubits
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    
    # Pauli矩阵
    I = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    
    # 添加最近邻相互作用项
    for i in range(Lx):
        for j in range(Ly):
            current_site = i * Ly + j
            
            # 水平方向耦合（向右）
            if i < Lx - 1:
                right_site = (i + 1) * Ly + j
                # σx⊗σx 项
                ops = [I] * num_qubits
                ops[current_site] = X
                ops[right_site] = X
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
                
                # σy⊗σy 项
                ops = [I] * num_qubits
                ops[current_site] = Y
                ops[right_site] = Y
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
                
                # σz⊗σz 项
                ops = [I] * num_qubits
                ops[current_site] = Z
                ops[right_site] = Z
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
            
            # 垂直方向耦合（向下）
            if j < Ly - 1:
                down_site = i * Ly + (j + 1)
                # σx⊗σx 项
                ops = [I] * num_qubits
                ops[current_site] = X
                ops[down_site] = X
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
                
                # σy⊗σy 项
                ops = [I] * num_qubits
                ops[current_site] = Y
                ops[down_site] = Y
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
                
                # σz⊗σz 项
                ops = [I] * num_qubits
                ops[current_site] = Z
                ops[down_site] = Z
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += J * term
    
    # 添加外场项：h * σz
    for i in range(Lx):
        for j in range(Ly):
            site = i * Ly + j
            ops = [I] * num_qubits
            ops[site] = Z
            term = ops[0]
            for op in ops[1:]:
                term = torch.kron(term, op)
            H += h * term
    
    return H


class Heisenberg2DHamiltonianOperator:
    """
    2D海森堡模型哈密顿量-向量乘法黑盒，不存储矩阵，只实现 H @ v
    """
    def __init__(self, Lx, Ly, J=1.0, h=0.0, device=None):
        self.Lx = Lx
        self.Ly = Ly
        self.J = J
        self.h = h
        self.device = device
        self.num_qubits = Lx * Ly
        self.dim = 2 ** self.num_qubits
        self.shape = (self.dim, self.dim)  # 兼容主流程打印

    def __matmul__(self, v):
        if isinstance(v, torch.Tensor):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        
        # 确保输入是2D数组
        if v_np.ndim == 1:
            v_np = v_np.reshape(1, -1)  # [1, dim]
        
        batch, dim = v_np.shape
        if dim != self.dim:
            raise ValueError(f"Vector dimension {dim} does not match hamiltonian dimension {self.dim}")
        
        out = np.zeros_like(v_np, dtype=np.complex64)
        
        for b in range(batch):
            for state in range(dim):
                val = 0.0
                
                # 相互作用项：J * (σx⊗σx + σy⊗σy + σz⊗σz)
                for i in range(self.Lx):
                    for j in range(self.Ly):
                        current_site = i * self.Ly + j
                        
                        # 水平方向耦合（向右）
                        if i < self.Lx - 1:
                            right_site = (i + 1) * self.Ly + j
                            # σx⊗σx 项
                            flipped_state = state ^ ((1 << current_site) | (1 << right_site))
                            val += self.J * v_np[b, flipped_state]
                            
                            # σy⊗σy 项（带相位）
                            phase = 1.0j
                            if ((state >> current_site) & 1) != ((state >> right_site) & 1):
                                phase = -1.0j
                            val += self.J * phase * v_np[b, flipped_state]
                            
                            # σz⊗σz 项
                            sign = 1.0
                            if ((state >> current_site) & 1) == ((state >> right_site) & 1):
                                sign = -1.0
                            val += self.J * sign * v_np[b, state]
                        
                        # 垂直方向耦合（向下）
                        if j < self.Ly - 1:
                            down_site = i * self.Ly + (j + 1)
                            # σx⊗σx 项
                            flipped_state = state ^ ((1 << current_site) | (1 << down_site))
                            val += self.J * v_np[b, flipped_state]
                            
                            # σy⊗σy 项（带相位）
                            phase = 1.0j
                            if ((state >> current_site) & 1) != ((state >> down_site) & 1):
                                phase = -1.0j
                            val += self.J * phase * v_np[b, flipped_state]
                            
                            # σz⊗σz 项
                            sign = 1.0
                            if ((state >> current_site) & 1) == ((state >> down_site) & 1):
                                sign = -1.0
                            val += self.J * sign * v_np[b, state]
                
                # 外场项：h * σz
                for i in range(self.Lx):
                    for j in range(self.Ly):
                        site = i * self.Ly + j
                        sign = 1.0 if ((state >> site) & 1) == 0 else -1.0
                        val += self.h * sign * v_np[b, state]
                
                out[b, state] = val
        
        out_tensor = torch.tensor(out, dtype=torch.complex64, device=self.device)
        if v.ndim == 1:
            return out_tensor[0]
        return out_tensor


def create_quasi_1d_heisenberg_hamiltonian(width: int, length: int, J: float = 1.0, 
                                          h: float = 0.0, device: torch.device = None):
    """
    创建quasi-1D海森堡模型哈密顿量（宽度固定，长度可变）
    
    Args:
        width: 固定宽度（如文献中的4）
        length: 可变长度（如文献中的2,3,4,5,6,8）
        J: 相互作用强度
        h: 外场强度
        device: 计算设备
        
    Returns:
        哈密顿量（密集矩阵或黑盒操作符）
    """
    return create_2d_heisenberg_hamiltonian(width, length, J, h, device)


def create_paper_4N_heisenberg_hamiltonian(N, device=None):
    """
    构造与arXiv:2007.10917v2一致的4*N准一维Heisenberg哈密顿量，
    包含每个子系统内部5条边和子系统间2-0连接。
    N: 子系统数，总比特数为4*N
    """
    num_qubits = 4 * N
    dim = 2 ** num_qubits
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    # Pauli矩阵
    I = torch.eye(2, dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    # 子系统内部边
    for block in range(N):
        offset = block * 4
        edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
        for (a, b) in edges:
            q1 = offset + a
            q2 = offset + b
            for pauli in [X, Y, Z]:
                ops = [I] * num_qubits
                ops[q1] = pauli
                ops[q2] = pauli
                term = ops[0]
                for op in ops[1:]:
                    term = torch.kron(term, op)
                H += term

    # 子系统之间的2-0连接
    for block in range(N-1):
        q1 = block * 4 + 2
        q2 = (block + 1) * 4 + 0
        for pauli in [X, Y, Z]:
            ops = [I] * num_qubits
            ops[q1] = pauli
            ops[q2] = pauli
            term = ops[0]
            for op in ops[1:]:
                term = torch.kron(term, op)
            H -= term

    return H


class Paper4NHeisenbergHamiltonianOperator:
    """
    黑盒实现 arXiv:2007.10917v2 结构的 4*N 海森堡哈密顿量，只实现 H @ v，不存储矩阵。
    """
    def __init__(self, N, device=None):
        self.N = N
        self.num_qubits = 4 * N
        self.device = device
        self.dim = 2 ** self.num_qubits
        self.shape = (self.dim, self.dim)

        # 预先生成所有边
        self.edges = []
        for block in range(N):
            offset = block * 4
            # 子系统内部边
            for (a, b) in [(0,1), (1,2), (2,3), (3,0), (0,2)]:
                self.edges.append((offset + a, offset + b))
        # 子系统间2-0连接
        for block in range(N-1):
            q1 = block * 4 + 2
            q2 = (block + 1) * 4 + 0
            self.edges.append((q1, q2))

    def __matmul__(self, v):
        if isinstance(v, torch.Tensor):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        if v_np.ndim == 1:
            v_np = v_np.reshape(1, -1)
        batch, dim = v_np.shape
        if dim != self.dim:
            raise ValueError(f"Vector dimension {dim} does not match hamiltonian dimension {self.dim}")
        out = np.zeros_like(v_np, dtype=np.complex64)
        for b in range(batch):
            for state in range(dim):
                val = 0.0
                for (q1, q2) in self.edges:
                    # σx⊗σx
                    flipped = state ^ ((1 << q1) | (1 << q2))
                    val += v_np[b, flipped]
                    # σy⊗σy（带相位）
                    phase = 1.0j
                    if ((state >> q1) & 1) != ((state >> q2) & 1):
                        phase = -1.0j
                    val += phase * v_np[b, flipped]
                    # σz⊗σz
                    sign = 1.0
                    if ((state >> q1) & 1) == ((state >> q2) & 1):
                        sign = -1.0
                    val += sign * v_np[b, state]
                out[b, state] = val
        out_tensor = torch.tensor(out, dtype=torch.complex64, device=self.device)
        if v.ndim == 1:
            return out_tensor[0]
        return out_tensor


def create_paper_4N_heisenberg_hamiltonian_operator(N, device=None):
    """
    返回黑盒 Paper4NHeisenbergHamiltonianOperator
    """
    return Paper4NHeisenbergHamiltonianOperator(N, device) 