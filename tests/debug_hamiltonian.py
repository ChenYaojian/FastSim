#!/usr/bin/env python3
"""
调试哈密顿量一致性问题
"""

import torch
import numpy as np
from fastsim.hamiltonian import create_paper_4N_heisenberg_hamiltonian_operator, create_paper_4N_heisenberg_hamiltonian

def debug_hamiltonian_consistency():
    """调试哈密顿量一致性问题"""
    print("=== 调试哈密顿量一致性问题 ===")
    
    N = 1
    num_qubits = 4 * N
    print(f"N={N}, 比特数={num_qubits}")
    
    # 创建哈密顿量
    H_operator = create_paper_4N_heisenberg_hamiltonian_operator(N)
    H_dense = create_paper_4N_heisenberg_hamiltonian(N)
    
    print(f"黑盒哈密顿量类型: {type(H_operator)}")
    print(f"密集哈密顿量类型: {type(H_dense)}")
    print(f"密集哈密顿量形状: {H_dense.shape}")
    
    # 测试几个简单的态
    test_states = [
        ("|0000⟩", torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)),
        ("|0001⟩", torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)),
        ("|0010⟩", torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)),
        ("|0011⟩", torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)),
    ]
    
    for name, state in test_states:
        state = state.unsqueeze(0)  # 添加batch维度
        
        # 计算期望值
        energy_operator = H_operator @ state
        energy_dense = torch.matmul(H_dense, state.transpose(0, 1)).transpose(0, 1)
        
        # 计算能量期望值
        exp_operator = torch.sum(torch.conj(state) * energy_operator, dim=1).real
        exp_dense = torch.sum(torch.conj(state) * energy_dense, dim=1).real
        
        print(f"\n{name}:")
        print(f"  黑盒: {exp_operator.item():.6f}")
        print(f"  密集: {exp_dense.item():.6f}")
        print(f"  差异: {abs(exp_operator.item() - exp_dense.item()):.6f}")
    
    # 检查密集矩阵的一些元素
    print(f"\n密集矩阵的一些元素:")
    print(f"H[0,0] = {H_dense[0,0].item():.6f}")
    print(f"H[0,1] = {H_dense[0,1].item():.6f}")
    print(f"H[1,0] = {H_dense[1,0].item():.6f}")
    print(f"H[1,1] = {H_dense[1,1].item():.6f}")
    
    # 检查黑盒操作符对|0000⟩的作用
    state_0000 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    result_operator = H_operator @ state_0000
    print(f"\n黑盒操作符对|0000⟩的作用:")
    for i in range(min(8, len(result_operator))):
        print(f"  [{i}] = {result_operator[i].item():.6f}")

if __name__ == "__main__":
    debug_hamiltonian_consistency() 