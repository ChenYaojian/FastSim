import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastsim.circuit import Circuit
from fastsim.circuit import load_gates_from_config

class QuantumNN(nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()
        # 创建量子电路
        load_gates_from_config(os.path.join(project_root, "configs", "gates_config.json"))
        self.circuit = Circuit(num_qubits)
        
        # 添加一些参数化门
        # 第一层：对每个量子比特应用RX和RY门
        for i in range(num_qubits):
            self.circuit.add_gate("RX", [i], [0.1])
            self.circuit.add_gate("RY", [i], [0.2])
        
        # 第二层：添加一些纠缠门
        for i in range(num_qubits-1):
            self.circuit.add_gate("CNOT", [i, i+1])
        
        # 第三层：再次应用单比特门
        for i in range(num_qubits):
            self.circuit.add_gate("RX", [i], [0.3])
            self.circuit.add_gate("RY", [i], [0.4])
        
        # 经典神经网络层
        self.linear1 = nn.Linear(2**num_qubits*2, 128)
        self.linear2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # 添加编码层，将输入数据编码为量子态
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),  # MNIST图像是28x28=784
            nn.ReLU(),
            nn.Linear(256, 2**num_qubits * 2),  # 实部和虚部
            nn.Tanh()  # 使用tanh确保值在[-1,1]范围内
        )
        
    def encode_to_quantum_state(self, x):
        """将输入数据编码为量子态
        
        Args:
            x: 输入数据，形状为 [batch_size, 784]
            
        Returns:
            torch.Tensor: 编码后的量子态，形状为 [batch_size, 2**num_qubits]
        """
        # 通过编码器
        encoded = self.encoder(x)
        # 分离实部和虚部
        real_part = encoded[:, :2**self.circuit.num_qubits]
        imag_part = encoded[:, 2**self.circuit.num_qubits:]
        # 组合成复数
        quantum_state = real_part + 1j * imag_part
        # 归一化
        norm = torch.sqrt(torch.sum(torch.abs(quantum_state)**2, dim=1, keepdim=True))
        quantum_state = quantum_state / norm
        return quantum_state
        
    def forward(self, x):
        # 将输入展平
        x = x.view(x.size(0), -1)
        
        # 编码为量子态
        quantum_state = self.encode_to_quantum_state(x)
        
        # 通过量子电路（作为神经网络层）
        quantum_state = self.circuit(quantum_state)
        
        # 将量子态转换为实数表示
        x = torch.cat([quantum_state.real, quantum_state.imag], dim=1)
        
        # 通过经典神经网络层
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 打印梯度信息
        if batch_idx % 100 == 0:
            print("\nGradient Information:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    print(f"{name}:")
                    print(f"  - Norm: {grad_norm:.6f}")
                    print(f"  - Mean: {grad_mean:.6f}")
                    print(f"  - Std: {grad_std:.6f}")
                    if torch.isnan(param.grad).any():
                        print(f"  - WARNING: NaN gradients detected!")
                    if torch.isinf(param.grad).any():
                        print(f"  - WARNING: Inf gradients detected!")
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'\nTrain Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否可用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = QuantumNN(num_qubits=4).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # 打印训练信息
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'quantum_nn_mnist.pth')
    print("Model saved to quantum_nn_mnist.pth")

if __name__ == '__main__':
    main() 