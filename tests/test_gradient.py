import torch
import torch.nn as nn
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.circuit import Circuit, load_gates_from_config

def test_gradient_flow():
    """测试梯度是否正确传递到量子门的参数"""
    print("测试梯度传递...")
    
    # 加载门配置
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 创建简单的量子电路
    circuit = Circuit(num_qubits=2)
    
    # 添加参数化门
    circuit.add_gate("RX", [0], [0.5])
    circuit.add_gate("RY", [1], [0.3])
    circuit.add_gate("CNOT", [0, 1])
    
    # 创建输入量子态
    input_state = torch.randn(1, 4, dtype=torch.complex64, requires_grad=True)
    
    # 前向传播
    output_state = circuit(input_state)
    
    # 计算损失（使用输出态的实部）
    loss = torch.sum(output_state.real ** 2)
    
    print(f"Loss: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("\n梯度信息:")
    for name, param in circuit.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"  - 参数值: {param.item():.6f}")
            print(f"  - 梯度值: {param.grad.item():.6f}")
            print(f"  - 梯度范数: {param.grad.norm().item():.6f}")
        else:
            print(f"{name}: 无梯度")
    
    # 检查输入梯度
    if input_state.grad is not None:
        print(f"\n输入梯度范数: {input_state.grad.norm().item():.6f}")
    else:
        print("\n输入无梯度")
    
    return circuit

def test_quantum_nn_gradient():
    """测试完整的量子神经网络梯度"""
    print("\n" + "="*50)
    print("测试完整量子神经网络梯度...")
    
    # 加载门配置
    load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
    
    # 创建量子神经网络
    class SimpleQuantumNN(nn.Module):
        def __init__(self, num_qubits=2):
            super().__init__()
            self.circuit = Circuit(num_qubits)
            
            # 添加参数化门
            self.circuit.add_gate("RX", [0], [0.1])
            self.circuit.add_gate("RY", [0], [0.2])
            self.circuit.add_gate("RX", [1], [0.3])
            self.circuit.add_gate("RY", [1], [0.4])
            self.circuit.add_gate("CNOT", [0, 1])
            
            # 经典层
            self.linear = nn.Linear(8, 2)  # 2^2 * 2 (实部+虚部) = 8
    
        def forward(self, x):
            # 编码为量子态
            batch_size = x.size(0)
            quantum_state = torch.randn(batch_size, 4, dtype=torch.complex64)
            quantum_state = quantum_state / torch.norm(quantum_state, dim=1, keepdim=True)
            
            # 通过量子电路
            quantum_state = self.circuit(quantum_state)
            
            # 转换为实数
            x = torch.cat([quantum_state.real, quantum_state.imag], dim=1)
            
            # 通过经典层
            return self.linear(x)
    
    model = SimpleQuantumNN()
    
    # 创建输入和目标
    x = torch.randn(2, 10)  # batch_size=2, input_dim=10
    target = torch.randint(0, 2, (2,))
    
    # 前向传播
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    print(f"Loss: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    # 检查所有参数的梯度
    print("\n所有参数梯度信息:")
    total_params = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            print(f"{name}:")
            print(f"  - 梯度范数: {grad_norm:.6f}")
            if torch.isnan(param.grad).any():
                print(f"  - WARNING: NaN梯度!")
            if torch.isinf(param.grad).any():
                print(f"  - WARNING: Inf梯度!")
        else:
            print(f"{name}: 无梯度")
    
    print(f"\n总结: {params_with_grad}/{total_params} 个参数有梯度")
    
    return model

if __name__ == "__main__":
    # 测试简单电路梯度
    circuit = test_gradient_flow()
    
    # 测试完整神经网络梯度
    model = test_quantum_nn_gradient()
    
    print("\n梯度测试完成!") 