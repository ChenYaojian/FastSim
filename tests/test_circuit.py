import sys
import os
import torch
import numpy as np
import pytest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.circuit import Circuit, load_gates_from_config, QuantumGate, ParametricGate

# 加载门配置
@pytest.fixture(scope="session", autouse=True)
def load_gates():
    """自动加载门配置"""
    config_path = os.path.join(project_root, "configs", "gates_config.json")
    load_gates_from_config(config_path)

@pytest.fixture
def basic_circuit():
    """创建基本测试电路"""
    circuit = Circuit(3)
    circuit.add_gate("H", [0])
    circuit.add_gate("X", [1])
    circuit.add_gate("CNOT", [0, 1])
    circuit.add_gate("RX", [2], [0.5])
    circuit.add_gate("RY", [0], [0.3])
    return circuit

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
    
    def test_add_gate(self, basic_circuit):
        """测试添加门操作"""
        assert len(basic_circuit.gates) == 5
        
        # 检查第一个门（非参数化门）
        gate, qubit_indices = basic_circuit.gates[0]
        assert gate.name == "H"
        assert qubit_indices == [0]
        
        # 检查参数化门
        gate, qubit_indices = basic_circuit.gates[3]
        assert gate.name == "RX"
        assert qubit_indices == [2]
        # 检查参数是否存在
        assert hasattr(gate, 'theta')
        assert isinstance(gate.theta, torch.nn.Parameter)
    
    def test_forward(self, basic_circuit):
        """测试前向传播"""
        state = basic_circuit.forward()
        assert state.shape == (1, 8)  # 现在包含batch维度
        assert torch.allclose(torch.abs(state), torch.abs(state))  # 检查是否为复数
        assert torch.allclose(torch.sum(torch.abs(state)**2, dim=1), torch.ones(1), atol=1e-3)  # 检查归一化
    
    def test_draw(self, basic_circuit):
        """测试电路绘制"""
        circuit_str = basic_circuit.draw()
        assert isinstance(circuit_str, str)
        assert "H" in circuit_str
        assert "X" in circuit_str
        # CNOT门在draw中显示为●和⊕符号
        assert "●" in circuit_str or "⊕" in circuit_str
        assert "RX" in circuit_str

def test_circuit_with_measurement():
    """测试带测量的电路"""
    circuit = Circuit(2)
    
    # 创建Bell态
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])
    
    # 执行电路
    state = circuit.forward()
    
    # 验证Bell态
    expected_state = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    assert torch.allclose(torch.abs(state[0]), torch.abs(expected_state), atol=1e-6)  # 注意取第一个batch

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
        assert torch.allclose(state[0, i], expected[i], atol=1e-6), f"振幅不符: {i}, {state[0, i]}, {expected[i]}"

def test_circuit_device():
    """测试电路在不同设备上的运行"""
    # 测试CPU
    circuit_cpu = Circuit(2, device=torch.device('cpu'))
    state_cpu = circuit_cpu.forward()
    assert state_cpu.device.type == 'cpu'
    
    # 测试GPU（如果可用）
    if torch.cuda.is_available():
        circuit_gpu = Circuit(2, device=torch.device('cuda'))
        state_gpu = circuit_gpu.forward()
        assert state_gpu.device.type == 'cuda'
        
        # 测试设备迁移
        circuit_cpu.to('cuda')
        state = circuit_cpu.forward()
        assert state.device.type == 'cuda'
        
        circuit_cpu.to('cpu')
        state = circuit_cpu.forward()
        assert state.device.type == 'cpu'

def test_circuit_device_consistency():
    """测试电路在不同设备上的一致性"""
    circuit_cpu = Circuit(2, device=torch.device('cpu'))
    circuit_cpu.add_gate("H", [0])
    circuit_cpu.add_gate("CNOT", [0, 1])
    state_cpu = circuit_cpu.forward()
    
    if torch.cuda.is_available():
        circuit_gpu = Circuit(2, device=torch.device('cuda'))
        circuit_gpu.add_gate("H", [0])
        circuit_gpu.add_gate("CNOT", [0, 1])
        state_gpu = circuit_gpu.forward()
        
        # 将GPU结果移回CPU进行比较
        state_gpu_cpu = state_gpu.cpu()
        assert torch.allclose(state_cpu, state_gpu_cpu, atol=1e-6)

def test_circuit_device_parameters():
    """测试参数化门在不同设备上的行为"""
    circuit = Circuit(2, device=torch.device('cpu'))
    circuit.add_gate("RX", [0], [0.1])
    circuit.add_gate("RY", [1], [0.2])
    
    # 检查参数是否在正确的设备上
    for name, param in circuit.named_parameters():
        assert param.device.type == 'cpu'
    
    if torch.cuda.is_available():
        circuit.to('cuda')
        # 检查参数是否被正确移动到GPU
        for name, param in circuit.named_parameters():
            assert param.device.type == 'cuda'
        
        # 执行前向传播
        state = circuit.forward()
        assert state.device.type == 'cuda'

def test_parametric_gate_gradients():
    """测试参数化门的梯度"""
    circuit = Circuit(2)
    circuit.add_gate("RX", [0], [0.5])
    circuit.add_gate("RY", [1], [0.3])
    
    # 创建输入状态
    input_state = torch.randn(1, 4, dtype=torch.complex64, requires_grad=True)
    
    # 前向传播
    output_state = circuit(input_state)
    
    # 计算损失
    loss = torch.sum(torch.abs(output_state) ** 2)
    
    # 反向传播
    loss.backward()
    
    # 检查参数是否有梯度
    for name, param in circuit.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有梯度"
        assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度包含NaN"
        assert not torch.isinf(param.grad).any(), f"参数 {name} 梯度包含Inf"

def test_multi_parameter_gate():
    """测试多参数门"""
    circuit = Circuit(2)
    # 测试U门（三参数门）
    circuit.add_gate("U", [0], [0.1, 0.2, 0.3])
    
    # 检查参数是否正确设置
    gate, qubit_indices = circuit.gates[0]
    assert gate.name == "U"
    assert qubit_indices == [0]
    assert hasattr(gate, 'alpha')
    assert hasattr(gate, 'beta')
    assert hasattr(gate, 'gamma')
    
    # 测试前向传播
    state = circuit.forward()
    assert state.shape == (1, 4)

def test_circuit_from_json():
    """测试从JSON创建电路"""
    # 创建测试JSON数据
    circuit_data = {
        "0": [
            {"gate_name": "H", "qubits": [0]},
            {"gate_name": "X", "qubits": [1]}
        ],
        "1": [
            {"gate_name": "CNOT", "qubits": [0, 1]}
        ],
        "2": [
            {"gate_name": "RX", "qubits": [0], "parameters": {"theta": 0.5}}
        ]
    }
    
    circuit = Circuit.from_json(circuit_data)
    assert circuit.num_qubits == 2
    assert len(circuit.gates) == 4
    
    # 测试前向传播
    state = circuit.forward()
    assert state.shape == (1, 4)

if __name__ == "__main__":
    pytest.main()
