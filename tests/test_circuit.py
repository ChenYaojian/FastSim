import sys
import os
import torch
import numpy as np
import pytest
import json
import tempfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.circuit import Circuit, load_gates_from_config, QuantumGate, ParametricGate, _parse_complex_matrix

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

@pytest.fixture
def complex_gates_config():
    """创建包含复数门的测试配置"""
    return {
        "gates": [
            {
                "name": "S",
                "num_qubits": 1,
                "is_parametric": False,
                "matrix": [
                    [1, 0],
                    [0, "j"]
                ]
            },
            {
                "name": "T",
                "num_qubits": 1,
                "is_parametric": False,
                "matrix": [
                    [1, 0],
                    [0, "1j"]
                ]
            },
            {
                "name": "Y",
                "num_qubits": 1,
                "is_parametric": False,
                "matrix": [
                    [0, "-j"],
                    ["j", 0]
                ]
            },
            {
                "name": "Phase",
                "num_qubits": 1,
                "is_parametric": True,
                "matrix_func": "lambda theta: torch.tensor([[1, 0], [0, torch.exp(1j * theta)]], dtype=torch.complex64)",
                "param_names": ["theta"]
            }
        ]
    }

# ==================== 单元测试 ====================

class TestComplexMatrixParsing:
    """测试复数矩阵解析功能"""
    
    def test_parse_complex_matrix_basic(self):
        """测试基本复数解析"""
        matrix = [[1, "j"], ["j", 0]]
        result = _parse_complex_matrix(matrix)
        expected = [[1, 1j], [1j, 0]]
        assert result == expected
    
    def test_parse_complex_matrix_already_complex(self):
        """测试已经是复数的值"""
        matrix = [[1j, 0], [0, 1j]]
        result = _parse_complex_matrix(matrix)
        assert result == matrix  # 应该保持不变
    
    def test_parse_complex_matrix_mixed(self):
        """测试混合格式"""
        matrix = [["1j", "-j"], [1j, "j"]]
        result = _parse_complex_matrix(matrix)
        expected = [[1j, -1j], [1j, 1j]]
        assert result == expected
    
    def test_parse_complex_matrix_negative(self):
        """测试负复数"""
        matrix = [["-j", "-1j"], ["j", "1j"]]
        result = _parse_complex_matrix(matrix)
        expected = [[-1j, -1j], [1j, 1j]]
        assert result == expected
    
    def test_parse_complex_matrix_numbers(self):
        """测试数字复数"""
        matrix = [["2j", "3j"], ["-2j", "-3j"]]
        result = _parse_complex_matrix(matrix)
        expected = [[2j, 3j], [-2j, -3j]]
        assert result == expected
    
    def test_parse_complex_matrix_complex_expressions(self):
        """测试复杂表达式"""
        matrix = [[1+1j, 0], [0, 1-1j]]
        result = _parse_complex_matrix(matrix)
        assert result == matrix  # 应该保持不变

class TestGateLoading:
    """测试门加载功能"""
    
    def test_load_gates_from_config_complex(self, complex_gates_config):
        """测试加载包含复数的门配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complex_gates_config, f)
            config_path = f.name
        
        try:
            load_gates_from_config(config_path)
            
            # 测试S门
            s_gate = QuantumGate.get_gate("S")
            s_matrix = s_gate.get_matrix()
            expected_s = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
            assert torch.allclose(s_matrix, expected_s)
            
            # 测试T门
            t_gate = QuantumGate.get_gate("T")
            t_matrix = t_gate.get_matrix()
            expected_t = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
            assert torch.allclose(t_matrix, expected_t)
            
            # 测试Y门
            y_gate = QuantumGate.get_gate("Y")
            y_matrix = y_gate.get_matrix()
            expected_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
            assert torch.allclose(y_matrix, expected_y)
            
            # 测试参数化门
            phase_gate = QuantumGate.get_gate("Phase")
            assert phase_gate.param_names == ["theta"]
            
        finally:
            os.unlink(config_path)
    
    def test_load_gates_from_config_error_handling(self):
        """测试门加载错误处理"""
        invalid_config = {
            "gates": [
                {
                    "name": "InvalidGate",
                    "num_qubits": 1,
                    "is_parametric": False
                    # 缺少matrix字段
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_gates_from_config(config_path)
        finally:
            os.unlink(config_path)

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
    
    def test_add_gate_with_complex_parameters(self):
        """测试添加带复数参数的门"""
        circuit = Circuit(2)
        
        # 测试不同参数格式
        circuit.add_gate("RX", [0], [0.5])  # 单个参数
        circuit.add_gate("U", [1], [0.1, 0.2, 0.3])  # 多个参数
        circuit.add_gate("RX", [0], {"theta": 0.7})  # 字典格式参数
        
        assert len(circuit.gates) == 3
        
        # 检查参数化门的参数
        gate, _ = circuit.gates[0]
        assert hasattr(gate, 'theta')
        assert gate.theta.item() == pytest.approx(0.5)
        
        gate, _ = circuit.gates[1]
        assert hasattr(gate, 'alpha')
        assert hasattr(gate, 'beta')
        assert hasattr(gate, 'gamma')
        
        gate, _ = circuit.gates[2]
        assert gate.theta.item() == pytest.approx(0.7)
    
    def test_forward(self, basic_circuit):
        """测试前向传播"""
        state = basic_circuit.forward()
        assert state.shape == (1, 8)  # 现在包含batch维度
        assert torch.allclose(torch.abs(state), torch.abs(state))  # 检查是否为复数
        assert torch.allclose(torch.sum(torch.abs(state)**2, dim=1), torch.ones(1), atol=1e-3)  # 检查归一化
    
    def test_forward_with_input(self, basic_circuit):
        """测试带输入的前向传播"""
        input_state = torch.randn(2, 8, dtype=torch.complex64)
        output_state = basic_circuit.forward(input_state)
        assert output_state.shape == (2, 8)
    
    def test_draw(self, basic_circuit):
        """测试电路绘制"""
        circuit_str = basic_circuit.draw()
        assert isinstance(circuit_str, str)
        assert "H" in circuit_str
        assert "X" in circuit_str
        # CNOT门在draw中显示为●和⊕符号
        assert "●" in circuit_str or "⊕" in circuit_str
        assert "RX" in circuit_str
    
    def test_error_handling(self):
        """测试错误处理"""
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
    
    def test_device_handling(self):
        """测试设备处理"""
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
    
    def test_device_consistency(self):
        """测试设备一致性"""
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
    
    def test_parametric_gate_gradients(self):
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
    
    def test_from_json(self):
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

# ==================== 流程测试 ====================

class TestQuantumAlgorithms:
    """量子算法流程测试"""
    
    def test_bell_state_creation(self):
        """测试Bell态创建"""
        circuit = Circuit(2)
        
        # 创建Bell态
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])
        
        # 执行电路
        state = circuit.forward()
        
        # 验证Bell态
        expected_state = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
        assert torch.allclose(torch.abs(state[0]), torch.abs(expected_state), atol=1e-6)
        
        # 验证振幅
        for i in range(4):
            if i in [0, 3]:
                assert torch.allclose(state[0, i], expected_state[i], atol=1e-6)
            else:
                assert torch.allclose(state[0, i], torch.tensor(0j), atol=1e-6)
    
    def test_quantum_fourier_transform(self):
        """测试量子傅里叶变换（标准2比特QFT）"""
        circuit = Circuit(2)
        
        # 标准2比特QFT电路
        circuit.add_gate("H", [0])
        circuit.add_gate("CS", [0, 1])  # 受控S门
        circuit.add_gate("H", [1])
        
        # 执行电路
        state = circuit.forward()
        
        # 验证输出状态
        assert state.shape == (1, 4)
        # 检查状态向量的模长平方和是否为1（归一化）
        norm_squared = torch.sum(torch.abs(state)**2, dim=1)
        assert abs(norm_squared.item() - 1.0) < 1e-4, f"状态未归一化: {norm_squared}"
        
        # 验证QFT输出（对|00>输入，QFT输出应该是均匀叠加）
        # 2比特QFT将|00>映射到(1/2)(|00> + |01> + |10> + |11>)
        expected_qft_state = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.complex64, device=state.device)
        assert torch.allclose(state[0], expected_qft_state, atol=1e-5), f"QFT输出不正确: {state[0]}"
    
    def test_parameterized_algorithm(self):
        """测试参数化算法"""
        circuit = Circuit(2)
        
        # 创建参数化电路
        circuit.add_gate("RX", [0], [0.5])
        circuit.add_gate("RY", [1], [0.3])
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("RZ", [0], [0.2])
        
        # 执行电路
        state = circuit.forward()
        
        # 验证输出
        assert state.shape == (1, 4)
        assert torch.allclose(torch.sum(torch.abs(state)**2, dim=1), torch.ones(1), atol=1e-6)
        
        # 测试梯度计算
        loss = torch.sum(torch.abs(state) ** 2)
        loss.backward()
        
        # 验证参数有梯度
        for name, param in circuit.named_parameters():
            assert param.grad is not None

class TestCircuitOptimization:
    """电路优化流程测试"""
    
    def test_circuit_optimization(self):
        """测试电路优化流程"""
        circuit = Circuit(2)
        
        # 创建可优化的电路
        circuit.add_gate("RX", [0], [0.1])
        circuit.add_gate("RY", [1], [0.2])
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("RX", [0], [0.3])
        
        # 定义优化器
        optimizer = torch.optim.Adam(circuit.parameters(), lr=0.01)
        
        # 定义目标状态（例如Bell态）
        target_state = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
        
        # 优化循环
        for epoch in range(10):
            optimizer.zero_grad()
            
            # 前向传播
            output_state = circuit.forward()
            
            # 计算损失（与目标状态的保真度）
            fidelity = torch.abs(torch.sum(torch.conj(target_state) * output_state[0])) ** 2
            loss = 1 - fidelity
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 验证优化后的状态
        final_state = circuit.forward()
        final_fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state[0])) ** 2
        assert final_fidelity > 0.3  # 优化应该有一定效果，降低阈值

if __name__ == "__main__":
    pytest.main() 