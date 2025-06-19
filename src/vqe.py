import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Dict, List, Union, Callable, Optional
from .circuit import Circuit, load_gates_from_config, QuantumGate


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
        # 计算 H|ψ>
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


def create_pqc_from_config(circuit_config: Dict, num_qubits: int, 
                          device: torch.device = None) -> PQC:
    """从配置创建PQC"""
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
            
            # 处理参数
            if isinstance(params, dict) and params:
                # 将dict参数转换为list
                gate_class = QuantumGate._registry.get(gate_name)
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
                                 h: float = 0.0, device: torch.device = None) -> torch.Tensor:
    """创建海森堡模型哈密顿量"""
    dim = 2 ** num_qubits
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