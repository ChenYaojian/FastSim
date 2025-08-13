import torch
import time
import numpy as np
from fastsim.hamiltonian import create_hamiltonian
from fastsim.vqe import VQE, build_pqc_adaptive
from fastsim.circuit import load_gates_from_config

def test_16_qubit_lightweight():
    """轻量级16比特系统测试"""
    print("=== 16比特系统轻量级测试 (N=4) ===")
    
    device = torch.device('cpu')
    N = 4
    num_qubits = 4 * N
    
    # 1. 哈密顿量结构分析
    print("\n1. 哈密顿量结构分析")
    H = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=False, device=device)
    
    positive_terms = 0
    negative_terms = 0
    for term in H.terms:
        if hasattr(term, 'coefficient'):
            coef = term.coefficient
            if coef > 0:
                positive_terms += 1
            else:
                negative_terms += 1
    
    print(f"总项数: {len(H.terms)}")
    print(f"正系数项数: {positive_terms}")
    print(f"负系数项数: {negative_terms}")
    
    # 2. 创建直积态初态（使用更小的系统）
    print("\n2. 创建直积态初态")
    H_4qubit = create_hamiltonian('paper_4n_heisenberg', N=1, use_decomposed=False, device=device)
    H_4qubit_matrix = H_4qubit.get_matrix()
    eigenvalues_4, eigenvectors_4 = torch.linalg.eigh(H_4qubit_matrix)
    ground_state_4 = eigenvectors_4[:, 0]
    
    print(f"4比特子系统基态能量: {eigenvalues_4[0].item():.6f}")
    
    # 创建直积态
    product_state = ground_state_4
    for i in range(1, N):
        product_state = torch.kron(product_state, ground_state_4)
    product_state = product_state.unsqueeze(0)
    
    print(f"直积态形状: {product_state.shape}")
    print(f"直积态范数: {torch.norm(product_state):.6f}")
    
    # 3. 计算初始能量
    initial_energy = H.expectation(product_state)
    print(f"直积态初始能量: {initial_energy.item():.6f}")
    print(f"理论直积态能量: {N * eigenvalues_4[0].item():.6f}")
    
    # 4. PQC结构分析
    print("\n3. PQC结构分析")
    pqc = build_pqc_adaptive(num_qubits, device=device)
    print(f"参数数量: {pqc.parameter_count}")
    print(f"门数量: {len(pqc.gates)}")
    
    # 5. VQE测试（减少迭代次数）
    print("\n4. VQE优化测试")
    H_vqe = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=True, device=device)
    vqe = VQE(pqc, H_vqe, optimizer_kwargs={'lr': 0.01}, store_best_state=True)
    
    start_time = time.time()
    result = vqe.optimize(
        num_iterations=300,  # 减少迭代次数
        input_state=product_state,
        convergence_threshold=1e-6,
        patience=50  # 减少耐心值
    )
    vqe_time = time.time() - start_time
    
    print(f"优化时间: {vqe_time:.4f}秒")
    print(f"总迭代次数: {result['iterations']}")
    print(f"最终能量: {result['final_energy']:.6f}")
    print(f"最优能量: {result['best_energy']:.6f}")
    print(f"能量改善: {initial_energy.item() - result['best_energy']:.6f}")
    
    return {
        'hamiltonian_terms': len(H.terms),
        'positive_terms': positive_terms,
        'negative_terms': negative_terms,
        'initial_energy': initial_energy.item(),
        'final_energy': result['best_energy'],
        'energy_improvement': initial_energy.item() - result['best_energy'],
        'parameter_count': pqc.parameter_count,
        'iterations': result['iterations'],
        'time': vqe_time
    }

def main():
    """主函数"""
    print("=== 16比特系统轻量级测试 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试16比特系统
    result = test_16_qubit_lightweight()
    
    # 结果总结
    print("\n=== 结果总结 ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return result

if __name__ == "__main__":
    result = main() 