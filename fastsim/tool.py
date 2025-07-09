import torch
import numpy as np

def hartree_fock_state(num_qubits: int, num_electrons: int = None, device=None) -> torch.Tensor:
    """
    构造Hartree-Fock初态，对于海森堡模型返回自旋对称基态|0...0>。
    Args:
        num_qubits: 量子比特数
        num_electrons: 电子数（对于海森堡模型忽略）
        device: torch设备
    Returns:
        torch.complex64, shape=[2**num_qubits]
    """
    dim = 2 ** num_qubits
    state = torch.zeros(dim, dtype=torch.complex64, device=device)
    # 对于海森堡模型，HF态为全零态|0...0>
    # 这是自旋对称的基态
    state[0] = 1.0
    return state


def get_hf_init_state(num_qubits: int, num_electrons: int = None, device=None) -> torch.Tensor:
    """
    返回Hartree-Fock初态，带batch维度，适用于VQE输入。
    """
    state = hartree_fock_state(num_qubits, num_electrons, device)
    return state.unsqueeze(0)


def hartree_fock_state_electronic(num_qubits: int, num_electrons: int, device=None) -> torch.Tensor:
    """
    构造电子系统的Hartree-Fock初态（Slater行列式）。
    假设填充最低能级的num_electrons个电子。
    Args:
        num_qubits: 量子比特数
        num_electrons: 电子数（或填充数）
        device: torch设备
    Returns:
        torch.complex64, shape=[2**num_qubits]
    """
    dim = 2 ** num_qubits
    state = torch.zeros(dim, dtype=torch.complex64, device=device)
    # HF态为前num_electrons个比特为1，其余为0
    # 例如4比特2电子: |1100> = 0b1100 = 12
    hf_index = 0
    for i in range(num_electrons):
        hf_index |= (1 << (num_qubits - 1 - i))
    state[hf_index] = 1.0
    return state


def get_hf_init_state_electronic(num_qubits: int, num_electrons: int, device=None) -> torch.Tensor:
    """
    返回电子系统的Hartree-Fock初态，带batch维度，适用于VQE输入。
    """
    state = hartree_fock_state_electronic(num_qubits, num_electrons, device)
    return state.unsqueeze(0) 