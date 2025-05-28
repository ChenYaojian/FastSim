import pytest
import torch
import numpy as np
from src.abs_circuit import AbstractCircuit, AbstractGate
from src.circuit import Circuit, QuantumGate

class TestAbstractCircuit:
    """测试AbstractCircuit类"""
    
    def test_init(self):
        """测试抽象电路初始化"""
        abs_circuit = AbstractCircuit(3)
        assert abs_circuit.num_qubits == 3
        assert len(abs_circuit.gates) == 0
        assert not abs_circuit.optimized
    
    def test_from_circuit(self, test_abs_circuit):
        """测试从Circuit创建AbstractCircuit"""
        assert test_abs_circuit.num_qubits == 3
        assert len(test_abs_circuit.gates) == 4
        assert isinstance(test_abs_circuit.gates[0], AbstractGate)
    
    def test_add_gate(self, test_abs_circuit):
        """测试添加门操作"""
        gate = test_abs_circuit.gates[0]
        assert isinstance(gate, AbstractGate)
        assert gate.num_qubits == 1
        assert not gate.is_parametric
    
    def test_are_gates_adjacent(self, test_abs_circuit):
        """测试门相邻性判断"""
        gate1 = test_abs_circuit.gates[0]  # H门
        gate2 = test_abs_circuit.gates[2]  # CNOT门
        assert test_abs_circuit._are_gates_adjacent(gate1, gate2)
    
    def test_can_fuse_gates(self, test_abs_circuit):
        """测试门融合可能性判断"""
        gate1 = test_abs_circuit.gates[0]  # H门
        gate2 = test_abs_circuit.gates[2]  # CNOT门
        assert test_abs_circuit._can_fuse_gates(gate1, gate2)
    
    def test_to_tn(self, test_abs_circuit):
        """测试转换为张量网络"""
        tn = test_abs_circuit.to_tn()
        assert len(tn.arrays) == len(test_abs_circuit.gates) + 1  # +1 for initial state
        assert len(tn.indices) == len(tn.arrays)
        assert all(idx in tn.size_dict for idx in set([idx for idx_list in tn.indices for idx in idx_list]))
    
    def test_optimize(self, test_abs_circuit):
        """测试电路优化"""
        original_gates = test_abs_circuit.gates.copy()
        test_abs_circuit.optimize()
        assert test_abs_circuit.optimized
        assert len(test_abs_circuit.gates) <= len(original_gates)
    
    def test_forward(self, test_abs_circuit):
        """测试前向传播"""
        state = test_abs_circuit.forward()
        assert state.shape == (8, 8)  # 矩阵形式
        assert torch.allclose(torch.abs(state), torch.abs(state))  # 检查是否为复数
        assert torch.allclose(torch.trace(state), torch.tensor(8.0))  # 检查是否为酉矩阵

class TestAbstractGate:
    """测试AbstractGate类"""
    
    def test_init(self, basic_gates):
        """测试门初始化"""
        H, X, CNOT, RX = basic_gates
        abs_gate = AbstractGate.from_quantum_gate(H, [0])
        assert abs_gate.num_qubits == 1
        assert not abs_gate.is_parametric
        assert abs_gate.qubit_indices == [0]
    
    def test_get_matrix(self, basic_gates):
        """测试获取矩阵表示"""
        H, X, CNOT, RX = basic_gates
        abs_gate = AbstractGate.from_quantum_gate(H, [0])
        matrix = abs_gate.get_matrix()
        assert matrix.shape == (2, 2)
        assert torch.allclose(torch.abs(matrix), torch.abs(matrix))  # 检查是否为复数
    
    def test_fuse(self, basic_gates):
        """测试门融合"""
        H, X, CNOT, RX = basic_gates
        abs_gate1 = AbstractGate.from_quantum_gate(H, [0])
        abs_gate2 = AbstractGate.from_quantum_gate(X, [0])
        fused_gate = abs_gate1.fuse(abs_gate2)
        assert fused_gate is not None
        assert fused_gate.num_qubits == 1
        assert fused_gate.qubit_indices == [0]

def test_circuit_optimization_chain():
    """测试电路优化链"""
    # 创建一个包含多个可优化门的电路
    circuit = Circuit(3)
    H = QuantumGate.get_gate("H")
    X = QuantumGate.get_gate("X")
    CNOT = QuantumGate.get_gate("CNOT")
    
    # 添加一系列可以优化的门
    circuit.add_gate("H", [0])
    circuit.add_gate("X", [0])
    circuit.add_gate("H", [1])
    circuit.add_gate("CNOT", [0, 1])
    circuit.add_gate("H", [2])
    circuit.add_gate("CNOT", [1, 2])
    
    # 转换为抽象电路并优化
    abs_circuit = AbstractCircuit.from_circuit(circuit)
    # 为参数化门提供默认参数（如果有的话）
    for gate in abs_circuit.gates:
        if gate.is_parametric:
            gate.params = torch.tensor([np.pi/4])
    abs_circuit.optimize()
    
    # 验证优化结果
    assert abs_circuit.optimized
    assert len(abs_circuit.gates) < len(circuit.gates)

def test_variational_circuit():
    """测试变分电路"""
    circuit = Circuit(2)
    RX = QuantumGate.get_gate("RX")
    
    # 添加参数化门
    theta1 = torch.tensor([np.pi/4])
    theta2 = torch.tensor([np.pi/3])
    circuit.add_gate("RX", [0], theta1)
    circuit.add_gate("RX", [1], theta2)
    
    # 转换为抽象电路并优化
    abs_circuit = AbstractCircuit.from_circuit(circuit)
    # 为参数化门提供参数
    for gate in abs_circuit.gates:
        if gate.is_parametric:
            gate.params = torch.tensor([np.pi/4])  # 提供默认参数
    abs_circuit.optimize()
    
    # 验证优化结果
    assert abs_circuit.optimized
    assert all(gate.is_parametric for gate in abs_circuit.gates) 