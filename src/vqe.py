import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Dict, List, Union, Callable, Optional
from src.circuit import Circuit, load_gates_from_config, QuantumGate


class PQC(Circuit):
    """参数化量子电路（Parameterized Quantum Circuit）"""
    
    def __init__(self, num_qubits: int, device: torch.device = None):
        super().__init__(num_qubits, device)
        self.parameter_count = 0
        self.parameter_names = []
    
    def add_parametric_gate(self, gate_name: str, qubit_indices: List[int], 
                           initial_params: Union[List[float], torch.Tensor] = None):
        """添加参数化门并自动初始化参数"""
        if initial_params is None:
            # 如果没有提供初始参数，使用随机值
            gate_class = QuantumGate._registry.get(gate_name)
            if gate_class is None:
                raise ValueError(f"Gate {gate_name} not registered")
            
            # 创建临时实例来获取参数数量
            temp_gate = gate_class()
            num_params = len(getattr(temp_gate, 'param_names', []))
            initial_params = torch.randn(num_params) * 0.1  # 小随机值初始化
        
        self.add_gate(gate_name, qubit_indices, initial_params)
        
        # 更新参数计数
        gate, _ = self.gates[-1]
        if hasattr(gate, 'param_names'):
            for param_name in gate.param_names:
                self.parameter_names.append(f"{gate_name}_{len(self.gates)-1}.{param_name}")
                self.parameter_count += 1
    
    def get_parameters(self) -> torch.Tensor:
        """获取所有参数的向量表示"""
        params = []
        for gate, _ in self.gates:
            if hasattr(gate, 'param_names'):
                for param_name in gate.param_names:
                    param = getattr(gate, param_name)
                    params.append(param)
        return torch.stack(params) if params else torch.tensor([])
    
    def set_parameters(self, params: torch.Tensor):
        """设置所有参数"""
        param_idx = 0
        for gate, _ in self.gates:
            if hasattr(gate, 'param_names'):
                for param_name in gate.param_names:
                    if param_idx < len(params):
                        param = getattr(gate, param_name)
                        param.data = params[param_idx].data
                        param_idx += 1
    
    def parameter_count(self) -> int:
        """返回参数总数"""
        return self.parameter_count


class VQE(nn.Module):
    """变分量子本征求解器（Variational Quantum Eigensolver）"""
    
    def __init__(self, pqc: PQC, hamiltonian: torch.Tensor, 
                 optimizer_class=optim.Adam, optimizer_kwargs: Dict = None):
        """
        初始化VQE
        
        Args:
            pqc: 参数化量子电路
            hamiltonian: 哈密顿量矩阵
            optimizer_class: 优化器类
            optimizer_kwargs: 优化器参数
        """
        super().__init__()
        self.pqc = pqc
        self.hamiltonian = hamiltonian
        
        # 设置优化器
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.01}
        self.optimizer = optimizer_class(self.pqc.parameters(), **optimizer_kwargs)
        
        # 优化历史
        self.energy_history = []
        self.parameter_history = []
    
    def expectation_value(self, state: torch.Tensor) -> torch.Tensor:
        """计算期望值 <ψ|H|ψ>"""
        # 确保state的shape正确
        if state.ndim == 1:
            # 如果是1D向量，转换为(batch=1, dim)
            state = state.unsqueeze(0)
        
        # 检查哈密顿量类型
        if hasattr(self.hamiltonian, '__matmul__') and not isinstance(self.hamiltonian, torch.Tensor):
            # 黑盒哈密顿量
            # 直接使用state，不需要transpose
            h_state = self.hamiltonian @ state
        else:
            # 密集哈密顿量
            h_state = torch.matmul(self.hamiltonian, state.transpose(0, 1)).transpose(0, 1)
        
        # 计算 <ψ|H|ψ> = <ψ|(H|ψ>)>
        expectation = torch.sum(torch.conj(state) * h_state, dim=1)
        return expectation.real
    
    def forward(self, input_state: torch.Tensor = None) -> torch.Tensor:
        """前向传播，返回能量期望值"""
        # 通过PQC
        output_state = self.pqc(input_state)
        # 计算期望值
        energy = self.expectation_value(output_state)
        return energy
    
    def optimize_step(self, input_state: torch.Tensor = None) -> float:
        """执行一步优化"""
        self.optimizer.zero_grad()
        
        # 前向传播
        energy = self.forward(input_state)
        loss = energy.mean()  # 如果有batch，取平均
        
        # 反向传播
        loss.backward()
        
        # 优化器步进
        self.optimizer.step()
        
        # 记录历史
        energy_val = loss.item()
        self.energy_history.append(energy_val)
        self.parameter_history.append(self.pqc.get_parameters().detach().clone())
        
        return energy_val
    
    def optimize(self, num_iterations: int, input_state: torch.Tensor = None,
                 convergence_threshold: float = 1e-6, patience: int = 100) -> Dict:
        """
        优化VQE
        
        Args:
            num_iterations: 最大迭代次数
            input_state: 输入量子态
            convergence_threshold: 收敛阈值
            patience: 早停耐心值
            
        Returns:
            优化结果字典
        """
        best_energy = float('inf')
        patience_counter = 0
        
        for iteration in range(num_iterations):
            energy = self.optimize_step(input_state)
            
            # 检查收敛
            if energy < best_energy - convergence_threshold:
                best_energy = energy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iteration}")
                break
            
            # 打印进度
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f}")
        
        return {
            'final_energy': energy,
            'best_energy': best_energy,
            'energy_history': self.energy_history,
            'parameter_history': self.parameter_history,
            'iterations': len(self.energy_history)
        }
    
    def get_ground_state(self, input_state: torch.Tensor = None) -> torch.Tensor:
        """获取基态"""
        with torch.no_grad():
            return self.pqc(input_state)


def load_circuit_from_file(file_path: str) -> Dict:
    """从文件加载电路配置"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_pqc_from_config(circuit_config: Dict, num_qubits: int = None, 
                          device: torch.device = None) -> PQC:
    """从配置创建PQC"""
    # 自动检测需要的量子比特数量
    if num_qubits is None:
        max_qubit = -1
        for layer in circuit_config.values():
            for gate in layer:
                max_in_gate = max(gate["qubits"])
                if max_in_gate > max_qubit:
                    max_qubit = max_in_gate
        num_qubits = max_qubit + 1
        print(f"Auto-detected number of qubits: {num_qubits}")
    
    pqc = PQC(num_qubits, device)
    
    # 按层添加门
    for layer_idx in sorted(circuit_config.keys(), key=lambda x: int(x)):
        layer = circuit_config[layer_idx]
        for gate_info in layer:
            gate_name = gate_info["gate_name"]
            qubits = gate_info["qubits"]
            params = gate_info.get("parameters", None)
            
            # 跳过测量门
            if gate_name in {"MX", "MY", "MZ"}:
                continue
            
            # 检查是否是参数化门
            gate_class = QuantumGate._registry.get(gate_name)
            is_parametric = gate_class and hasattr(gate_class(), 'param_names')
            
            # 处理参数
            if isinstance(params, dict) and params:
                # 将dict参数转换为list
                if gate_class:
                    temp_gate = gate_class()
                    param_names = getattr(temp_gate, 'param_names', list(params.keys()))
                    param_list = [params[k] for k in param_names]
                    pqc.add_parametric_gate(gate_name, qubits, param_list)
                else:
                    pqc.add_gate(gate_name, qubits, None)
            elif isinstance(params, (list, tuple)):
                pqc.add_parametric_gate(gate_name, qubits, params)
            else:
                # 对于参数化门，即使没有参数也要使用add_parametric_gate
                if is_parametric:
                    pqc.add_parametric_gate(gate_name, qubits, None)
                else:
                    pqc.add_gate(gate_name, qubits, None)
    
    return pqc


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
    
    # 小系统使用密集矩阵（保持原有逻辑）
    print(f"Small system ({num_qubits} qubits), using dense matrix")
    H = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    
    # 这里简化实现，实际应用中需要更复杂的构造
    # 对于小系统，可以直接构造完整的哈密顿量矩阵
    
    # 添加相互作用项 J * (σx⊗σx + σy⊗σy + σz⊗σz)
    for i in range(num_qubits - 1):
        # 这里需要实现具体的泡利矩阵张量积
        # 简化版本：使用随机矩阵模拟
        interaction = torch.randn(dim, dim, dtype=torch.complex64, device=device) * J
        interaction = (interaction + interaction.conj().transpose(0, 1)) / 2
        H += interaction
    
    # 添加外场项 h * σz
    for i in range(num_qubits):
        # 简化版本
        field_term = torch.randn(dim, dim, dtype=torch.complex64, device=device) * h
        field_term = (field_term + field_term.conj().transpose(0, 1)) / 2
        H += field_term
    
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
                    # 翻转第i和i+1位
                    flipped_state = state ^ ((1 << i) | (1 << (i+1)))
                    val += self.J * v_np[b, flipped_state]
                    
                    # σy⊗σy 项（带相位）
                    # 翻转第i和i+1位，并添加相位因子
                    phase = 1.0
                    if ((state >> i) & 1) != ((state >> (i+1)) & 1):
                        phase = -1.0
                    val += self.J * phase * v_np[b, flipped_state]
                    
                    # σz⊗σz 项
                    # 对角项，根据第i和i+1位的值确定符号
                    sign = 1.0
                    if ((state >> i) & 1) == ((state >> (i+1)) & 1):
                        sign = -1.0
                    val += self.J * sign * v_np[b, state]
                
                # 外场项：h * σz
                for i in range(self.num_qubits):
                    # 根据第i位的值确定符号
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