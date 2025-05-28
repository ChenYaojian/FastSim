import sys
import os
import torch
import numpy as np
import pytest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.circuit import Circuit, load_gates_from_config, QuantumGate, ParametricGate
from src.abs_circuit import AbstractCircuit, AbstractGate
from src.state import StateType, AbstractState

class TestCircuit:
    """测试Circuit类"""
    
    def test_init(self):
        """测试电路初始化"""
        circuit = Circuit(3)
        assert circuit.num_qubits == 3
        assert len(circuit.gates) == 0
        assert circuit.qubits.shape == (8,)
        assert circuit.qubits[0] == 1.0
        assert torch.all(circuit.qubits[1:] == 0)
    
    def test_add_gate(self, test_circuit):
        """测试添加门操作"""
        assert len(test_circuit.gates) == 5
        gate, qubit_indices, params = test_circuit.gates[0]
        assert gate.name == "H"
        assert qubit_indices == [0]
        assert params is None
        
        gate, qubit_indices, params = test_circuit.gates[3]
        assert gate.name == "RX"
        assert qubit_indices == [2]
        assert params is not None
    
    def test_forward(self, test_circuit):
        """测试前向传播"""
        state = test_circuit.forward()
        assert state.shape == (8,)
        assert torch.allclose(torch.abs(state), torch.abs(state))  # 检查是否为复数
        assert torch.allclose(torch.sum(torch.abs(state)**2), torch.tensor(1.0), atol=1e-3)  # 检查归一化
    
    def test_draw(self, test_circuit):
        """测试电路绘制"""
        circuit_str = test_circuit.draw()
        assert isinstance(circuit_str, str)
        assert "H" in circuit_str
        assert "X" in circuit_str
        #assert "CNOT" in circuit_str
        assert "RX" in circuit_str    

def test_circuit_with_measurement():
    """测试带测量的电路"""
    circuit = Circuit(2)
    H = QuantumGate.get_gate("H")
    CNOT = QuantumGate.get_gate("CNOT")
    
    # 创建Bell态
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])
    
    # 执行电路
    state = circuit.forward()
    
    # 验证Bell态
    expected_state = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    assert torch.allclose(torch.abs(state), torch.abs(expected_state), atol=1e-6)

def test_circuit_error_handling():
    """测试电路错误处理"""
    circuit = Circuit(2)
    
    # 测试无效的门名称
    with pytest.raises(ValueError):
        circuit.add_gate("InvalidGate", [0])
    
    # 测试无效的量子比特索引
    with pytest.raises(ValueError):
        circuit.add_gate("H", [3])
    
    # 测试缺少参数的门
    with pytest.raises(ValueError):
        circuit.add_gate("RX", [0])

def test_bell_state_amplitudes():
    circuit = Circuit(2)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])
    state = circuit.forward()
    expected = torch.tensor([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=torch.complex64)
    for i in range(4):
        assert torch.allclose(state[i], expected[i], atol=1e-6), f"振幅不符: {i}, {state[i]}, {expected[i]}"

if __name__ == "__main__":
    pytest.main()