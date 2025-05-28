import sys
import os
import torch
import numpy as np
import pytest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.circuit import Circuit, load_gates_from_config, QuantumGate
from src.state import StateType, AbstractState

@pytest.fixture(scope="session", autouse=True)
def register_all_gates():
    """自动从config文件注册所有门"""
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)

@pytest.fixture
def basic_gates(register_all_gates):
    """返回常用门的实例，已自动注册，无需手动class"""
    H = QuantumGate.get_gate("H")
    X = QuantumGate.get_gate("X")
    CNOT = QuantumGate.get_gate("CNOT")
    RX = QuantumGate.get_gate("RX")
    U = QuantumGate.get_gate("U")
    return H, X, CNOT, RX, U

@pytest.fixture
def test_circuit(basic_gates):
    """创建测试用的量子电路"""
    circuit = Circuit(3)
    H, X, CNOT, RX, U = basic_gates
    
    # 添加一些门操作
    circuit.add_gate("H", [0])
    circuit.add_gate("X", [1])
    circuit.add_gate("CNOT", [0, 1])
    circuit.add_gate("RX", [2], torch.tensor([np.pi/4]))
    circuit.add_gate("U", [0], torch.tensor([np.pi/4, np.pi/4, np.pi/4]))
    return circuit

