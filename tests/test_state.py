import sys
import os
import torch
import numpy as np
import pytest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.state import StateType, AbstractState

def test_state_creation():
    num_qubits = 2
    state = AbstractState.create_state(num_qubits)
    assert state.num_qubits == num_qubits
    assert state.state_type == StateType.STATE_VECTOR
    assert state.state_vector is not None
    assert state.state_vector.shape == (2**num_qubits,)
    assert torch.allclose(state.state_vector[0], torch.ones(1, dtype=torch.complex64))

def test_state_creation_with_bitstring():
    num_qubits = 2
    bitstring = "10"
    state = AbstractState.create_state(num_qubits, bitstring=bitstring)
    assert state.num_qubits == num_qubits
    assert state.state_type == StateType.STATE_VECTOR
    assert torch.allclose(state.state_vector[int(bitstring, 2)], torch.ones(1, dtype=torch.complex64))

class TestState:
    """测试量子态创建和使用"""
    
    def test_create_state_vector(self):
        """测试创建态向量"""
        # 创建3量子比特的态向量
        state = AbstractState.create_state(3, StateType.STATE_VECTOR)
        assert state.num_qubits == 3
        assert isinstance(state.state_vector, torch.Tensor)
        assert state.state_vector.shape == (8,)
        assert state.state_vector[0] == 1.0  # 初始化为|0⟩态
    
    def test_create_mps(self):
        """测试创建MPS态"""
        # 创建3量子比特的MPS态
        state = AbstractState.create_state(3, StateType.MPS, max_bond_dim=2)
        assert state.num_qubits == 3
        assert len(state.tensors) == 3
        assert all(tensor.shape[1] == 2 for tensor in state.tensors)  # 物理维度为2
    
    def test_create_from_bitstring(self):
        """测试从比特串创建态"""
        # 创建|101⟩态
        state = AbstractState.create_state(3, StateType.STATE_VECTOR, bitstring="101")
        assert state.num_qubits == 3
        state_vector = state.get_state_vector()
        assert state_vector[5] == 1.0  # |101⟩对应索引5
    
    def test_create_from_state_vector(self):
        """测试从态向量创建态"""
        # 创建随机态向量
        initial_state = torch.randn(8, dtype=torch.complex64)
        initial_state = initial_state / torch.norm(initial_state)
        
        # 创建态
        state = AbstractState.create_state(3, StateType.STATE_VECTOR, initial_state=initial_state)
        assert state.num_qubits == 3
        assert torch.allclose(state.get_state_vector(), initial_state)
    
    def test_invalid_state_type(self):
        """测试无效的态类型"""
        with pytest.raises(ValueError):
            AbstractState.create_state(3, "invalid_type")
    
    def test_invalid_parameters(self):
        """测试无效的参数组合"""
        with pytest.raises(ValueError):
            # 不能同时指定bitstring和initial_state
            AbstractState.create_state(3, StateType.STATE_VECTOR, 
                                     bitstring="101", 
                                     initial_state=torch.zeros(8, dtype=torch.complex64))
    
    def test_state_vector_operations(self):
        """测试态向量操作"""
        # 创建态向量
        state = AbstractState.create_state(2, StateType.STATE_VECTOR)
        
        # 测试测量
        results = state.measure(num_shots=1000)
        assert results.shape == (1000, 2)
        assert torch.all(results == 0)  # 初始态为|00⟩
        
        # 测试获取密度矩阵
        rho = state.get_density_matrix()
        assert rho.shape == (4, 4)
        assert torch.allclose(torch.trace(rho), torch.tensor(1.0, dtype=torch.complex64))
        
        # 测试计算期望值
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        expectation = state.get_expectation(Z, qubit_indices=[0])
        assert torch.allclose(expectation, torch.tensor(1.0))
        
        # 测试计算纠缠熵
        entropy = state.get_entanglement_entropy([0])
        assert torch.allclose(entropy, torch.tensor(0.0))  # 可分离态
    
    def test_mps_operations(self):
        """测试MPS操作"""
        # 创建MPS态
        state = AbstractState.create_state(3, StateType.MPS, max_bond_dim=2)
        
        # 测试获取态向量
        state_vector = state.get_state_vector()
        assert state_vector.shape == (8,)
        assert torch.allclose(state_vector[0], torch.tensor(1.0, dtype=torch.complex64))
        
        # 测试获取密度矩阵
        rho = state.get_density_matrix()
        assert rho.shape == (8, 8)
        assert torch.allclose(torch.trace(rho), torch.tensor(1.0, dtype=torch.complex64))
        
        # 测试计算期望值
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        expectation = state.get_expectation(Z, qubit_indices=[0])
        assert torch.allclose(expectation, torch.tensor(1.0))
        
        # 测试计算纠缠熵
        entropy = state.get_entanglement_entropy([0])
        assert torch.allclose(entropy, torch.tensor(0.0))  # 可分离态

if __name__ == "__main__":
    test_state_creation()
    test_state_creation_with_bitstring()
