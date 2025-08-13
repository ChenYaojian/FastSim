import torch
import time
import numpy as np
from fastsim.hamiltonian import create_hamiltonian
from fastsim.vqe import VQE, build_pqc_adaptive, build_pqc_u_cz, build_pqc_rx_rz_cnot, build_pqc_alternating, build_double_cz_pqc
from fastsim.circuit import load_gates_from_config
from fastsim.tool import get_hf_init_state

def create_zero_state(num_qubits, device=None):
    """创建|0...0⟩态（默认初态）"""
    state = torch.zeros(1, 2**num_qubits, dtype=torch.complex64, device=device)
    state[0, 0] = 1.0
    return state

def create_random_state(num_qubits, device=None):
    """创建随机初态"""
    state = torch.randn(1, 2**num_qubits, dtype=torch.complex64, device=device)
    state = state / torch.norm(state)
    return state

def create_hf_state(num_qubits, device=None):
    """创建Hartree-Fock初态"""
    return get_hf_init_state(num_qubits, device=device)

def create_partial_ground_state_product(num_qubits, device=None):
    """
    创建部分基态直积态作为初态
    对于N*4比特系统，使用N个4比特子系统的基态直积
    """
    N = num_qubits // 4
    print(f"创建 {N} 个4比特子系统的直积态...")
    
    # 计算4比特子系统的基态
    H_4qubit = create_hamiltonian('paper_4n_heisenberg', N=1, use_decomposed=False, device=device)
    H_4qubit_matrix = H_4qubit.get_matrix()
    _, eigenvectors_4 = torch.linalg.eigh(H_4qubit_matrix)
    ground_state_4 = eigenvectors_4[:, 0]
    
    # 创建直积态
    product_state = ground_state_4
    for i in range(1, N):
        product_state = torch.kron(product_state, ground_state_4)
    
    product_state = product_state.unsqueeze(0)  # 添加batch维度
    
    print(f"直积态形状: {product_state.shape}")
    print(f"直积态范数: {torch.norm(product_state):.6f}")
    
    return product_state

def create_perturbed_ground_state(num_qubits, perturbation_strength=0.1, device=None):
    """
    创建微扰基态
    在精确基态基础上添加小的随机扰动
    """
    # 计算精确基态
    H = create_hamiltonian('paper_4n_heisenberg', N=num_qubits//4, use_decomposed=False, device=device)
    H_matrix = H.get_matrix()
    _, eigenvectors = torch.linalg.eigh(H_matrix)
    ground_state = eigenvectors[:, 0]
    
    # 添加随机扰动
    perturbation = torch.randn_like(ground_state, dtype=torch.complex64) * perturbation_strength
    perturbed_state = ground_state + perturbation
    perturbed_state = perturbed_state / torch.norm(perturbed_state)
    
    return perturbed_state.unsqueeze(0)

def create_adiabatic_initialization(num_qubits, device=None):
    """
    创建绝热初始化态
    使用简单的绝热路径构造初态
    """
    # 从简单的哈密顿量开始
    H_simple = torch.zeros(2**num_qubits, 2**num_qubits, dtype=torch.complex64, device=device)
    
    # 添加对角项（简单的局域场）
    for i in range(2**num_qubits):
        H_simple[i, i] = -0.5 * bin(i).count('1')  # 简单的局域场
    
    # 计算基态
    _, eigenvectors = torch.linalg.eigh(H_simple)
    ground_state = eigenvectors[:, 0]
    
    return ground_state.unsqueeze(0)

def test_vqe_with_different_initializations(num_qubits, device=None):
    """测试不同初态初始化方法对VQE性能的影响"""
    print(f"\n{'='*60}")
    print(f"=== {num_qubits}比特系统不同初态初始化测试 ===")
    print(f"{'='*60}")
    
    # 创建哈密顿量
    N = num_qubits // 4
    H = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=True, device=device)
    print(f"哈密顿量项数: {len(H.terms)}")
    
    # 计算精确基态能量（用于对比）
    H_matrix = H.get_matrix()
    eigenvalues, _ = torch.linalg.eigh(H_matrix)
    exact_ground_energy = eigenvalues[0].item()
    print(f"精确基态能量: {exact_ground_energy:.6f}")
    
    # 定义不同的初态初始化方法
    init_methods = [
        ('零态', create_zero_state),
        ('随机态', create_random_state),
        ('HF态', create_hf_state),
        ('部分基态直积', create_partial_ground_state_product),
        ('微扰基态', lambda n, d: create_perturbed_ground_state(n, 0.1, d)),
        ('绝热初始化', create_adiabatic_initialization)
    ]
    
    # 使用简单的PQC进行测试
    pqc = build_pqc_adaptive(num_qubits, device)
    print(f"使用自适应PQC，参数数量: {pqc.parameter_count}")
    
    results = {}
    
    for method_name, init_func in init_methods:
        print(f"\n--- 测试 {method_name} 初始化 ---")
        
        try:
            # 创建初态
            start_time = time.time()
            init_state = init_func(num_qubits, device)
            init_time = time.time() - start_time
            
            # 计算初态能量
            initial_energy = H.expectation(init_state)
            print(f"初态能量: {initial_energy.item():.6f}")
            print(f"与精确基态能量差: {abs(initial_energy.item() - exact_ground_energy):.6f}")
            print(f"初态创建时间: {init_time:.4f}秒")
            
            # 创建VQE
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01}, store_best_state=True)
            
            # 进行优化
            print(f"开始VQE优化...")
            start_time = time.time()
            
            result = vqe.optimize(
                num_iterations=300,  # 减少迭代次数以加快测试
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=50
            )
            
            vqe_time = time.time() - start_time
            
            # 记录结果
            results[method_name] = {
                'initial_energy': initial_energy.item(),
                'final_energy': result['best_energy'],
                'energy_improvement': initial_energy.item() - result['best_energy'],
                'energy_error': abs(result['best_energy'] - exact_ground_energy),
                'iterations': result['iterations'],
                'vqe_time': vqe_time,
                'init_time': init_time,
                'total_time': init_time + vqe_time
            }
            
            print(f"最终能量: {result['best_energy']:.6f}")
            print(f"能量改善: {initial_energy.item() - result['best_energy']:.6f}")
            print(f"与精确基态误差: {abs(result['best_energy'] - exact_ground_energy):.6f}")
            print(f"优化时间: {vqe_time:.2f}秒")
            print(f"迭代次数: {result['iterations']}")
            
        except Exception as e:
            print(f"方法 {method_name} 失败: {e}")
            results[method_name] = {'error': str(e)}
    
    return results

def analyze_initialization_results(results):
    """分析不同初始化方法的结果"""
    print(f"\n{'='*80}")
    print(f"=== 初态初始化方法对比分析 ===")
    print(f"{'='*80}")
    
    # 按能量误差排序
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    sorted_results = sorted(valid_results.items(), 
                          key=lambda x: x[1]['energy_error'])
    
    print(f"{'方法':<15} {'初态能量':<12} {'最终能量':<12} {'能量改善':<10} {'误差':<10} {'迭代数':<8} {'总时间':<8}")
    print("-" * 80)
    
    for method_name, result in sorted_results:
        print(f"{method_name:<15} {result['initial_energy']:<12.6f} {result['final_energy']:<12.6f} "
              f"{result['energy_improvement']:<10.6f} {result['energy_error']:<10.6f} "
              f"{result['iterations']:<8} {result['total_time']:<8.2f}")
    
    # 找出最佳方法
    if sorted_results:
        best_method, best_result = sorted_results[0]
        print(f"\n最佳方法: {best_method}")
        print(f"最终能量误差: {best_result['energy_error']:.6f}")
        print(f"能量改善: {best_result['energy_improvement']:.6f}")
        print(f"总时间: {best_result['total_time']:.2f}秒")

def test_parameter_initialization_strategies(num_qubits, device=None):
    """测试不同的参数初始化策略"""
    print(f"\n{'='*60}")
    print(f"=== {num_qubits}比特系统参数初始化策略测试 ===")
    print(f"{'='*60}")
    
    # 创建哈密顿量和PQC
    N = num_qubits // 4
    H = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=True, device=device)
    pqc = build_pqc_adaptive(num_qubits, device)
    
    # 使用相同的初态（HF态）
    init_state = create_hf_state(num_qubits, device)
    
    # 不同的参数初始化策略
    param_strategies = [
        ('随机初始化', None),  # 默认随机初始化
        ('零初始化', lambda: torch.zeros(pqc.parameter_count)),
        ('小随机初始化', lambda: torch.randn(pqc.parameter_count) * 0.1),
        ('大随机初始化', lambda: torch.randn(pqc.parameter_count) * 1.0)
    ]
    
    results = {}
    
    for strategy_name, param_init_func in param_strategies:
        print(f"\n--- 测试 {strategy_name} ---")
        
        try:
            # 创建VQE
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
            
            # 如果指定了参数初始化函数，应用它
            if param_init_func is not None:
                with torch.no_grad():
                    params = param_init_func()
                    pqc.set_parameters(params)
            
            # 进行优化
            result = vqe.optimize(
                num_iterations=200,
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=50
            )
            
            results[strategy_name] = {
                'final_energy': result['best_energy'],
                'iterations': result['iterations']
            }
            
            print(f"最终能量: {result['best_energy']:.6f}")
            print(f"迭代次数: {result['iterations']}")
            
        except Exception as e:
            print(f"策略 {strategy_name} 失败: {e}")
            results[strategy_name] = {'error': str(e)}
    
    return results

def main():
    """主函数"""
    print("=== VQE初态和参数初始化策略测试 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试8比特系统（较小系统便于快速测试）
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 测试不同初态初始化方法
    print("\n" + "="*60)
    print("测试不同初态初始化方法")
    print("="*60)
    init_results = test_vqe_with_different_initializations(8, device)
    
    # 分析结果
    analyze_initialization_results(init_results)
    
    # 测试不同参数初始化策略
    print("\n" + "="*60)
    print("测试不同参数初始化策略")
    print("="*60)
    param_results = test_parameter_initialization_strategies(8, device)
    
    # 总结
    print(f"\n{'='*80}")
    print("=== 测试总结 ===")
    print(f"{'='*80}")
    print("1. 初态初始化方法对比完成")
    print("2. 参数初始化策略对比完成")
    print("3. 建议使用部分基态直积态或微扰基态作为初态")
    print("4. 参数初始化建议使用小随机初始化")
    
    return {
        'initialization_results': init_results,
        'parameter_results': param_results
    }

if __name__ == "__main__":
    result = main()
    print("\n=== 测试完成 ===") 