import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Dict, List, Union, Callable, Optional
from .circuit import Circuit, load_gates_from_config, QuantumGate
from .hamiltonian import *
from .tool import get_hf_init_state


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
    
    def randomize_parameters(self, scale: float = 0.1, seed: int = None):
        """
        随机初始化所有参数
        
        Args:
            scale: 随机化的尺度因子
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        for gate, _ in self.gates:
            if hasattr(gate, 'param_names'):
                for param_name in gate.param_names:
                    param = getattr(gate, param_name)
                    param.data = torch.randn_like(param.data) * scale
    
    def reset_parameters(self, initial_params: torch.Tensor = None):
        """
        重置参数到指定值或随机值
        
        Args:
            initial_params: 初始参数值，如果为None则随机初始化
        """
        if initial_params is not None:
            self.set_parameters(initial_params)
        else:
            self.randomize_parameters()


class VQE(nn.Module):
    """变分量子本征求解器（Variational Quantum Eigensolver）"""
    
    def __init__(self, pqc: PQC, hamiltonian: torch.Tensor, 
                 optimizer_class=optim.Adam, optimizer_kwargs: Dict = None,
                 store_best_state: bool = False, save_dir: str = "vqe_results"):
        """
        初始化VQE
        
        Args:
            pqc: 参数化量子电路
            hamiltonian: 哈密顿量矩阵
            optimizer_class: 优化器类
            optimizer_kwargs: 优化器参数
            store_best_state: 是否存储最佳状态
            save_dir: 保存结果的目录
        """
        super().__init__()
        self.pqc = pqc
        
        # 确保哈密顿量和PQC在同一个设备上
        device = next(self.pqc.parameters()).device if list(self.pqc.parameters()) else torch.device('cpu')
        if isinstance(hamiltonian, torch.Tensor):
            self.hamiltonian = hamiltonian.to(device)
        else:
            self.hamiltonian = hamiltonian
        
        # 设置优化器
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.01}
        self.optimizer = optimizer_class(self.pqc.parameters(), **optimizer_kwargs)
        
        # 优化历史
        self.energy_history = []
        self.parameter_history = []
        
        # 存储最佳状态的设置
        self.store_best_state = store_best_state
        self.save_dir = save_dir
        self.best_energy = float('inf')
        self.best_parameters = None
        self.best_state_vector = None
        
        # 创建保存目录
        if self.store_best_state:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def expectation_value(self, state: torch.Tensor) -> torch.Tensor:
        """计算期望值 <ψ|H|ψ>"""
        # 确保state的shape正确
        if state.ndim == 1:
            # 如果是1D向量，转换为(batch=1, dim)
            state = state.unsqueeze(0)
        
        # 确保状态向量和哈密顿量在同一个设备上
        if isinstance(self.hamiltonian, torch.Tensor):
            state = state.to(self.hamiltonian.device)
        
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
                
                # 如果启用了状态存储，更新最佳状态
                if self.store_best_state:
                    self.update_best_state(best_energy, input_state)
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
    
    def update_best_state(self, energy: float, input_state: torch.Tensor = None):
        """
        更新最佳状态（如果当前能量更低）
        
        Args:
            energy: 当前能量
            input_state: 输入量子态
        """
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_parameters = self.pqc.get_parameters().detach().clone()
            
            # 计算并存储最佳状态向量
            if input_state is not None:
                with torch.no_grad():
                    self.best_state_vector = self.pqc(input_state).detach().clone()
            else:
                with torch.no_grad():
                    self.best_state_vector = self.pqc().detach().clone()
    
    def save_best_state(self, filename_prefix: str = "best_state"):
        """
        保存最佳状态到文件
        
        Args:
            filename_prefix: 文件名前缀
        """
        if not self.store_best_state or self.best_parameters is None:
            print("警告: 未启用状态存储或没有最佳状态可保存")
            return
        
        timestamp = torch.tensor([torch.tensor(0.0).item()])  # 简单的占位符
        if hasattr(torch, 'datetime'):
            timestamp = torch.tensor([torch.datetime.now().timestamp()])
        
        # 保存参数
        param_file = os.path.join(self.save_dir, f"{filename_prefix}_parameters.pt")
        torch.save({
            'parameters': self.best_parameters,
            'energy': self.best_energy,
            'timestamp': timestamp,
            'num_qubits': self.pqc.num_qubits,
            'parameter_count': self.pqc.parameter_count
        }, param_file)
        
        # 保存状态向量
        if self.best_state_vector is not None:
            state_file = os.path.join(self.save_dir, f"{filename_prefix}_state_vector.pt")
            torch.save({
                'state_vector': self.best_state_vector,
                'energy': self.best_energy,
                'timestamp': timestamp,
                'num_qubits': self.pqc.num_qubits
            }, state_file)
        
        # 保存详细信息到JSON文件
        info_file = os.path.join(self.save_dir, f"{filename_prefix}_info.json")
        info = {
            'best_energy': self.best_energy,
            'num_qubits': self.pqc.num_qubits,
            'parameter_count': self.pqc.parameter_count,
            'timestamp': timestamp.item() if hasattr(timestamp, 'item') else timestamp[0].item(),
            'files': {
                'parameters': f"{filename_prefix}_parameters.pt",
                'state_vector': f"{filename_prefix}_state_vector.pt" if self.best_state_vector is not None else None
            }
        }
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"最佳状态已保存到 {self.save_dir}/")
        print(f"  - 参数文件: {param_file}")
        if self.best_state_vector is not None:
            print(f"  - 状态向量文件: {state_file}")
        print(f"  - 信息文件: {info_file}")
    
    def load_best_state(self, filename_prefix: str = "best_state") -> bool:
        """
        从文件加载最佳状态
        
        Args:
            filename_prefix: 文件名前缀
            
        Returns:
            bool: 是否成功加载
        """
        param_file = os.path.join(self.save_dir, f"{filename_prefix}_parameters.pt")
        
        if not os.path.exists(param_file):
            print(f"错误: 参数文件不存在 {param_file}")
            return False
        
        try:
            # 加载参数
            checkpoint = torch.load(param_file)
            self.best_parameters = checkpoint['parameters']
            self.best_energy = checkpoint['energy']
            
            # 设置电路参数
            self.pqc.set_parameters(self.best_parameters)
            
            # 尝试加载状态向量
            state_file = os.path.join(self.save_dir, f"{filename_prefix}_state_vector.pt")
            if os.path.exists(state_file):
                state_checkpoint = torch.load(state_file)
                self.best_state_vector = state_checkpoint['state_vector']
            
            print(f"成功加载最佳状态:")
            print(f"  - 最佳能量: {self.best_energy:.6f}")
            print(f"  - 量子比特数: {checkpoint['num_qubits']}")
            print(f"  - 参数数量: {checkpoint['parameter_count']}")
            
            return True
            
        except Exception as e:
            print(f"加载最佳状态时出错: {e}")
            return False
    
    def get_best_state_info(self) -> Dict:
        """
        获取最佳状态信息
        
        Returns:
            Dict: 包含最佳状态信息的字典
        """
        if self.best_parameters is None:
            return {'error': '没有保存的最佳状态'}
        
        info = {
            'best_energy': self.best_energy,
            'num_qubits': self.pqc.num_qubits,
            'parameter_count': self.pqc.parameter_count,
            'has_state_vector': self.best_state_vector is not None
        }
        
        if self.best_state_vector is not None:
            info['state_vector_shape'] = list(self.best_state_vector.shape)
            info['state_vector_norm'] = torch.norm(self.best_state_vector).item()
        
        return info
    
    def optimize_with_random_restarts(self, num_epochs: int, iterations_per_epoch: int, 
                                    input_state: torch.Tensor = None,
                                    convergence_threshold: float = 1e-6, 
                                    patience: int = 100,
                                    random_scale: float = 0.1,
                                    use_seed: bool = True) -> Dict:
        """
        使用随机重启优化VQE
        
        Args:
            num_epochs: 总epoch数
            iterations_per_epoch: 每个epoch的迭代次数
            input_state: 输入量子态
            convergence_threshold: 收敛阈值
            patience: 早停耐心值
            random_scale: 随机初始化尺度
            use_seed: 是否使用种子确保可重复性
            
        Returns:
            优化结果字典
        """
        best_energy = float('inf')
        best_parameters = None
        all_epoch_results = []
        
        print(f"开始随机重启优化: {num_epochs} 个epoch，每个epoch {iterations_per_epoch} 次迭代")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # 随机初始化参数
            if use_seed:
                self.pqc.randomize_parameters(scale=random_scale, seed=epoch)
            else:
                self.pqc.randomize_parameters(scale=random_scale)
            
            # 重置优化器
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = None
            
            # 重置历史记录
            self.energy_history = []
            self.parameter_history = []
            
            # 执行单个epoch的优化
            epoch_best_energy = float('inf')
            patience_counter = 0
            
            for iteration in range(iterations_per_epoch):
                energy = self.optimize_step(input_state)
                
                # 检查收敛
                if energy < epoch_best_energy - convergence_threshold:
                    epoch_best_energy = energy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= patience:
                    print(f"  Epoch {epoch + 1} 早停于迭代 {iteration}")
                    break
                
                # 打印进度
                if iteration % 100 == 0:
                    print(f"  Iteration {iteration}: Energy = {energy:.6f}")
            
            # 记录epoch结果
            epoch_result = {
                'epoch': epoch + 1,
                'final_energy': energy,
                'best_energy': epoch_best_energy,
                'iterations': len(self.energy_history),
                'energy_history': self.energy_history.copy(),
                'parameter_history': [p.clone() for p in self.parameter_history]
            }
            all_epoch_results.append(epoch_result)
            
            print(f"  Epoch {epoch + 1} 结果: 最终能量 = {energy:.6f}, 最优能量 = {epoch_best_energy:.6f}")
            
            # 更新全局最优
            if epoch_best_energy < best_energy:
                best_energy = epoch_best_energy
                best_parameters = self.pqc.get_parameters().detach().clone()
                print(f"  *** 新的全局最优能量: {best_energy:.6f} ***")
                
                # 如果启用了状态存储，更新最佳状态
                if self.store_best_state:
                    self.update_best_state(best_energy, input_state)
        
        # 恢复最佳参数
        if best_parameters is not None:
            self.pqc.set_parameters(best_parameters)
        
        print(f"\n随机重启优化完成!")
        print(f"全局最优能量: {best_energy:.6f}")
        
        return {
            'final_energy': energy,
            'best_energy': best_energy,
            'best_parameters': best_parameters,
            'epoch_results': all_epoch_results,
            'total_iterations': sum(r['iterations'] for r in all_epoch_results)
        }


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


def create_random_hamiltonian(num_qubits: int, device: torch.device = None):
    """创建随机哈密顿量用于测试"""
    dim = 2 ** num_qubits
    if device is None:
        device = torch.device('cpu')
    
    # 创建随机厄米矩阵
    H_real = torch.randn(dim, dim, device=device)
    H_imag = torch.randn(dim, dim, device=device)
    H = H_real + 1j * H_imag
    
    # 确保厄米性
    H = (H + H.conj().T) / 2
    
    return H


def create_quasi_1d_afm_hamiltonian(num_qubits: int, J_perp: float = 0.5, 
                                   J_parallel: float = 1.0, h: float = 0.0, 
                                   device: torch.device = None):
    """创建准一维反铁磁模型哈密顿量"""
    return Quasi1DAFMHamiltonian(num_qubits, J_perp, J_parallel, h, device)


def create_heisenberg_hamiltonian(num_qubits: int, J: float = 1.0, 
                                 h: float = 0.0, device: torch.device = None):
    """创建海森堡模型哈密顿量，大系统使用黑盒矩阵-向量乘法"""
    dim = 2 ** num_qubits
    
    # 对于大系统（>10比特），使用黑盒方式
    if num_qubits > 10:
        print(f"Large system detected ({num_qubits} qubits), using black-box matrix-vector multiplication")
        return HeisenbergHamiltonian(num_qubits, J, h, device)
    
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


def create_paper_4N_heisenberg_hamiltonian(N, device=None):
    """
    构造与arXiv:2007.10917v2一致的4*N准一维Heisenberg哈密顿量，
    包含每个子系统内部5条边和子系统间2-0连接。
    N: 子系统数，总比特数为4*N
    """
    num_qubits = 4 * N
    
    # 对于大系统（>10比特），使用分解模式
    if num_qubits > 10:
        print(f"Large system detected ({num_qubits} qubits), using decomposed Hamiltonian")
        from .hamiltonian import Paper4NHeisenbergHamiltonian
        return Paper4NHeisenbergHamiltonian(N, use_decomposed=True, device=device)
    
    # 小系统使用密集矩阵
    print(f"Small system ({num_qubits} qubits), using dense matrix")
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
            H += term

    return H 


 


def build_pqc_u_cz(num_qubits: int, num_layers: int = 2, closed_chain: bool = False, 
                   device: torch.device = None) -> PQC:
    """
    构建标准PQC：每层所有比特U门+相邻CZ门
    
    Args:
        num_qubits: 量子比特数
        num_layers: 层数
        closed_chain: 是否首尾相连（环形链）
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    pqc = PQC(num_qubits, device)
    
    for layer in range(num_layers):
        # 单比特U门
        for i in range(num_qubits):
            pqc.add_parametric_gate("U", [i])
        
        # 相邻CZ门
        for i in range(num_qubits - 1):
            pqc.add_gate("CZ", [i, i+1])
        
        # 如果是环形链且比特数>2，添加首尾连接
        if closed_chain and num_qubits > 2:
            pqc.add_gate("CZ", [num_qubits-1, 0])
    
    return pqc


def build_pqc_rx_rz_cnot(num_qubits: int, num_layers: int = 2, closed_chain: bool = False,
                         device: torch.device = None) -> PQC:
    """
    构建标准PQC：每层RX+RZ+CNOT结构
    
    Args:
        num_qubits: 量子比特数
        num_layers: 层数
        closed_chain: 是否首尾相连（环形链）
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    pqc = PQC(num_qubits, device)
    
    for layer in range(num_layers):
        # 单比特旋转门
        for i in range(num_qubits):
            pqc.add_parametric_gate("RX", [i])
            pqc.add_parametric_gate("RZ", [i])
        
        # 相邻CNOT门
        for i in range(num_qubits - 1):
            pqc.add_gate("CNOT", [i, i+1])
        
        # 如果是环形链且比特数>2，添加首尾连接
        if closed_chain and num_qubits > 2:
            pqc.add_gate("CNOT", [num_qubits-1, 0])
    
    return pqc


def build_double_cz_pqc(num_qubits: int, num_layers: int = 2, device: torch.device = None) -> PQC:
    """
    构建双层PQC：每层所有比特RX门+相邻CZ门+每层所有比特RZ门
    """
    pqc = PQC(num_qubits, device)
    for layer in range(num_layers):
        for ii in range(num_qubits):
            pqc.add_parametric_gate("RX", [ii], [0.0])
        for ii in range(0, num_qubits - 1, 2):
            pqc.add_gate("CZ", [ii, ii + 1])
        for ii in range(1, num_qubits - 1, 2):
            pqc.add_gate("CZ", [ii, ii + 1])
        for ii in range(num_qubits):
            pqc.add_parametric_gate("RZ", [ii], [0.0])
    return pqc

def build_pqc_alternating(num_qubits: int, num_layers: int = 2, closed_chain: bool = False,
                         device: torch.device = None) -> PQC:
    """
    构建交替纠缠PQC：相邻层使用不同方向的纠缠
    
    Args:
        num_qubits: 量子比特数
        num_layers: 层数
        closed_chain: 是否首尾相连（环形链）
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    pqc = PQC(num_qubits, device)
    
    for layer in range(num_layers):
        # 单比特U门
        for i in range(num_qubits):
            pqc.add_parametric_gate("U", [i])
        
        # 交替方向的纠缠
        if layer % 2 == 0:
            # 正向纠缠
            for i in range(num_qubits - 1):
                pqc.add_gate("CZ", [i, i+1])
            if closed_chain and num_qubits > 2:
                pqc.add_gate("CZ", [num_qubits-1, 0])
        else:
            # 反向纠缠
            for i in range(num_qubits - 1, 0, -1):
                pqc.add_gate("CZ", [i, i-1])
            if closed_chain and num_qubits > 2:
                pqc.add_gate("CZ", [0, num_qubits-1])
    
    return pqc


def build_pqc_adaptive(num_qubits: int, device: torch.device = None) -> PQC:
    """
    根据比特数自适应构建PQC
    
    Args:
        num_qubits: 量子比特数
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    if num_qubits <= 4:
        # 小系统：2层U+CZ
        return build_pqc_u_cz(num_qubits, num_layers=2, device=device)
    elif num_qubits <= 8:
        # 中等系统：3层RX+RZ+CNOT
        return build_pqc_rx_rz_cnot(num_qubits, num_layers=3, device=device)
    else:
        # 大系统：4层交替纠缠
        return build_pqc_alternating(num_qubits, num_layers=4, device=device) 


def build_pqc_hi_paper(num_qubits: int, num_cycles: int = 2, device: torch.device = None) -> PQC:
    """
    构建论文arXiv:2007.10917v2中的HI电路结构
    
    Args:
        num_qubits: 量子比特数（应该是4的倍数）
        num_cycles: cycle数量，每个cycle包含一层单比特门和一层双比特门
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    if num_qubits % 4 != 0:
        raise ValueError(f"量子比特数必须是4的倍数，当前为{num_qubits}")
    
    pqc = PQC(num_qubits, device)
    
    # 定义4比特子系统的边连接方式 E = {(0,1), (1,2), (2,3), (3,0), (0,2)}
    edges_4qubit = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    
    for cycle in range(num_cycles):
        # 每个cycle包含：
        # 1. 一层单比特U门
        for i in range(num_qubits):
            pqc.add_parametric_gate("U", [i])
        
        # 2. 一层双比特HI门，按照论文的连接方式
        for block in range(num_qubits // 4):
            offset = block * 4
            for edge in edges_4qubit:
                q1 = offset + edge[0]
                q2 = offset + edge[1]
                if q2 < num_qubits:  # 确保不超出范围   
                    pqc.add_parametric_gate("HI_ZZ", [q1, q2])
                    pqc.add_parametric_gate("HI_YY", [q1, q2])
                    pqc.add_parametric_gate("HI_XX", [q1, q2])

    
    return pqc


def build_pqc_hi_paper_4N(num_qubits: int, num_cycles: int = 2, device: torch.device = None) -> PQC:
    """
    构建4*N的PQC：将4*N个比特分成N组，每组4个比特按照HI论文方式构建，
    然后添加组间连接
    
    Args:
        num_qubits: 量子比特数（应该是4的倍数）
        num_cycles: cycle数量，每个cycle包含一层单比特门和一层双比特门
        device: 计算设备
        
    Returns:
        PQC: 参数化量子电路
    """
    if num_qubits % 4 != 0:
        raise ValueError(f"量子比特数必须是4的倍数，当前为{num_qubits}")
    
    N = num_qubits // 4  # 子系统数量
    pqc = PQC(num_qubits, device)
    
    # 定义4比特子系统的边连接方式 E = {(0,1), (1,2), (2,3), (3,0), (0,2)}
    edges_4qubit = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    
    for cycle in range(num_cycles):
        # 每个cycle包含：
        # 1. 一层单比特U门
        for i in range(num_qubits):
            pqc.add_parametric_gate("U", [i])
        
        # 2. 一层双比特HI门，按照论文的连接方式（组内连接）
        for block in range(N):
            offset = block * 4
            for edge in edges_4qubit:
                q1 = offset + edge[0]
                q2 = offset + edge[1]
                pqc.add_parametric_gate("HI_ZZ", [q1, q2])
                pqc.add_parametric_gate("HI_YY", [q1, q2])
                pqc.add_parametric_gate("HI_XX", [q1, q2])
    
        # 构建完毕后，在每个比特上加一层单比特U门
        for i in range(num_qubits):
            pqc.add_parametric_gate("U", [i])
    
        # 将第i组的2号比特和第i+1组的0号比特用HI门连接起来
        for i in range(N - 1):
            q1 = i * 4 + 2  # 第i组的2号比特
            q2 = (i + 1) * 4 + 0  # 第i+1组的0号比特
            pqc.add_parametric_gate("HI_ZZ", [q1, q2])
            pqc.add_parametric_gate("HI_YY", [q1, q2])
            pqc.add_parametric_gate("HI_XX", [q1, q2])
    
    return pqc


def compute_ground_state_4qubit(hamiltonian, device):
    """
    计算4比特系统的基态
    
    Args:
        hamiltonian: 4比特哈密顿量
        device: 计算设备
        
    Returns:
        ground_state: 基态向量
        ground_energy: 基态能量
    """
    # 对角化哈密顿量
    eigenvals, eigenvecs = torch.linalg.eigh(hamiltonian)
    ground_energy = eigenvals[0].real.item()
    ground_state = eigenvecs[:, 0]
    
    # 确保基态向量是2D的 (batch_size=1, state_dim)
    if ground_state.ndim == 1:
        ground_state = ground_state.unsqueeze(0)
    
    return ground_state, ground_energy


def create_product_state_from_4qubit_ground(ground_state_4qubit, num_blocks, device):
    """
    从4比特基态创建直积态
    
    Args:
        ground_state_4qubit: 4比特基态向量
        num_blocks: 块数（总比特数 = 4 * num_blocks）
        device: 计算设备
        
    Returns:
        product_state: 直积态向量
    """
    # 计算总比特数
    num_qubits = 4 * num_blocks
    
    # 创建直积态：|ψ⟩⊗|ψ⟩⊗...⊗|ψ⟩
    product_state = ground_state_4qubit.clone()
    
    for _ in range(num_blocks - 1):
        # 计算当前直积态的维度
        current_dim = product_state.size(0)
        # 与4比特基态做张量积
        product_state = torch.kron(product_state, ground_state_4qubit)
    
    return product_state.to(device)
