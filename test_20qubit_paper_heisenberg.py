import torch
import time
import numpy as np
from fastsim.vqe import VQE, build_pqc_adaptive, create_paper_4N_heisenberg_hamiltonian
from fastsim.hamiltonian import create_hamiltonian
from fastsim.tool import get_hf_init_state
from fastsim.circuit import load_gates_from_config

def create_4qubit_subsystem_ground_state(device=None):
    """
    创建4比特子系统的基态
    通过对角化获得精确的基态
    """
    # 创建4比特的paper海森堡哈密顿量
    H_4qubit = create_hamiltonian('paper_4n_heisenberg', N=1, use_decomposed=False, device=device)
    
    # 构建密集矩阵
    H_matrix = H_4qubit.get_matrix()
    
    # 对角化获得基态
    eigenvalues, eigenvectors = torch.linalg.eigh(H_matrix)
    
    # 获取基态（最小特征值对应的特征向量）
    ground_state = eigenvectors[:, 0]  # 基态向量
    
    # 确保是2D张量 [batch_size, dim]
    if ground_state.dim() == 1:
        ground_state = ground_state.unsqueeze(0)
    
    return ground_state

def create_20qubit_product_state(device=None):
    """
    创建20比特的直积态：5个4比特子系统基态的直积
    """
    # 创建5个4比特子系统的基态
    subsystem_states = []
    for i in range(5):
        subsystem_state = create_4qubit_subsystem_ground_state(device)
        subsystem_states.append(subsystem_state)
    
    # 计算直积态
    # 每个子系统是4比特，总共有5个，所以是20比特
    total_state = subsystem_states[0]
    for i in range(1, 5):
        # 计算直积：将当前状态与下一个子系统状态做张量积
        # 重塑为矩阵形式进行张量积
        current_shape = total_state.shape
        next_shape = subsystem_states[i].shape
        
        # 重塑为2D矩阵进行张量积
        if total_state.dim() == 1:
            total_state = total_state.unsqueeze(0)  # [dim] -> [1, dim]
        if subsystem_states[i].dim() == 1:
            subsystem_states[i] = subsystem_states[i].unsqueeze(0)  # [dim] -> [1, dim]
        
        # 计算张量积
        # 对于状态向量，我们需要计算 |ψ⟩ ⊗ |φ⟩
        # 如果 |ψ⟩ = [a0, a1, ..., a15], |φ⟩ = [b0, b1, ..., b15]
        # 那么 |ψ⟩ ⊗ |φ⟩ = [a0*b0, a0*b1, ..., a15*b15]
        
        # 重塑为矩阵形式
        current_matrix = total_state.view(-1, 1)  # [dim, 1]
        next_matrix = subsystem_states[i].view(1, -1)  # [1, dim]
        
        # 计算外积
        product = current_matrix * next_matrix  # [dim, dim]
        
        # 重塑为向量形式
        total_state = product.view(-1)  # [dim*dim]
        
        # 如果还有更多子系统，保持为向量形式
        if i < 4:
            total_state = total_state.unsqueeze(0)  # [1, dim*dim]
    
    # 确保最终状态是2D张量 [batch_size, dim]
    if total_state.dim() == 1:
        total_state = total_state.unsqueeze(0)
    
    return total_state

def test_20qubit_paper_heisenberg():
    """测试20比特paper海森堡模型的VQE优化"""
    print("=== 20比特Paper海森堡模型VQE测试 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 设置设备
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 创建哈密顿量
    print("\n1. 创建哈密顿量...")
    start_time = time.time()
    
    # 使用新的hamiltonian模块创建paper 4*N海森堡模型
    N = 5  # 20比特 = 4 * 5
    H = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=True, device=device)
    
    hamiltonian_time = time.time() - start_time
    print(f"哈密顿量创建时间: {hamiltonian_time:.4f}秒")
    print(f"哈密顿量项数: {len(H.terms)}")
    
    # 创建初始化态
    print("\n2. 创建初始化态...")
    start_time = time.time()
    
    init_state = create_20qubit_product_state(device)
    
    init_time = time.time() - start_time
    print(f"初始化态创建时间: {init_time:.4f}秒")
    print(f"初始化态形状: {init_state.shape}")
    print(f"初始化态范数: {torch.norm(init_state):.6f}")
    
    # 计算初始能量
    print("\n3. 计算初始能量...")
    start_time = time.time()
    
    initial_energy = H.expectation(init_state)
    
    energy_time = time.time() - start_time
    print(f"初始能量计算时间: {energy_time:.4f}秒")
    print(f"初始能量: {initial_energy.item():.6f}")
    
    # 创建PQC
    print("\n4. 创建参数化量子电路...")
    start_time = time.time()
    
    # 对于20比特系统，使用更深的电路
    pqc = build_pqc_adaptive(20, device=device)
    
    pqc_time = time.time() - start_time
    print(f"PQC创建时间: {pqc_time:.4f}秒")
    print(f"参数数量: {pqc.parameter_count}")
    
    # 创建VQE
    print("\n5. 创建VQE...")
    vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.001}, store_best_state=True)
    
    # 测试单步优化
    print("\n6. 测试单步优化...")
    start_time = time.time()
    
    energy = vqe.optimize_step(init_state)
    
    step_time = time.time() - start_time
    print(f"单步优化时间: {step_time:.4f}秒")
    print(f"单步优化后能量: {energy:.6f}")
    print(f"能量改善: {initial_energy.item() - energy:.6f}")
    
    # 进行完整优化
    print("\n7. 开始完整优化...")
    start_time = time.time()
    
    # 对于大系统，使用更多的迭代次数
    result = vqe.optimize(
        num_iterations=2000,  # 增加迭代次数
        input_state=init_state,
        convergence_threshold=1e-8,
        patience=200
    )
    
    total_time = time.time() - start_time
    print(f"完整优化时间: {total_time:.4f}秒")
    print(f"最终能量: {result['final_energy']:.6f}")
    print(f"最优能量: {result['best_energy']:.6f}")
    print(f"总迭代次数: {result['iterations']}")
    print(f"总能量改善: {initial_energy.item() - result['best_energy']:.6f}")
    
    # 保存结果
    print("\n8. 保存结果...")
    vqe.save_best_state("20qubit_paper_heisenberg")
    
    print("\n=== 测试完成 ===")
    print(f"初始能量: {initial_energy.item():.6f}")
    print(f"最终能量: {result['best_energy']:.6f}")
    print(f"能量改善: {initial_energy.item() - result['best_energy']:.6f}")
    print(f"改善百分比: {(initial_energy.item() - result['best_energy']) / abs(initial_energy.item()) * 100:.2f}%")
    
    return {
        'initial_energy': initial_energy.item(),
        'final_energy': result['best_energy'],
        'energy_improvement': initial_energy.item() - result['best_energy'],
        'iterations': result['iterations'],
        'total_time': total_time,
        'hamiltonian_terms': len(H.terms),
        'parameters': pqc.parameter_count
    }

if __name__ == "__main__":
    result = test_20qubit_paper_heisenberg()
    print("\n=== 结果总结 ===")
    for key, value in result.items():
        print(f"{key}: {value}") 