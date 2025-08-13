import torch
import time
import numpy as np
from fastsim.hamiltonian import create_hamiltonian
from fastsim.vqe import VQE, build_pqc_adaptive, build_pqc_u_cz, build_pqc_rx_rz_cnot, build_pqc_alternating, build_double_cz_pqc
from fastsim.circuit import load_gates_from_config

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

def analyze_pqc_structure(pqc, num_qubits):
    """分析PQC电路结构"""
    print(f"\n=== PQC电路结构分析 ===")
    print(f"量子比特数: {num_qubits}")
    print(f"参数数量: {pqc.parameter_count}")
    print(f"门数量: {len(pqc.gates)}")
    
    # 统计不同类型的门
    gate_types = {}
    parametric_gates = 0
    entanglement_gates = 0
    
    for gate, _ in pqc.gates:
        gate_name = gate.__class__.__name__
        gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
        
        if hasattr(gate, 'param_names'):
            parametric_gates += 1
        
        # 检查是否是纠缠门
        if hasattr(gate, 'qubit_indices') and len(gate.qubit_indices) > 1:
            entanglement_gates += 1
    
    print(f"参数化门数量: {parametric_gates}")
    print(f"纠缠门数量: {entanglement_gates}")
    print(f"门类型分布: {gate_types}")
    
    # 计算电路深度（估算）
    estimated_depth = len(pqc.gates) // num_qubits if num_qubits > 0 else 0
    print(f"估算电路深度: {estimated_depth}")
    
    return {
        'parameter_count': pqc.parameter_count,
        'gate_count': len(pqc.gates),
        'parametric_gates': parametric_gates,
        'entanglement_gates': entanglement_gates,
        'estimated_depth': estimated_depth
    }

def test_vqe_convergence(num_qubits, device=None):
    """测试指定比特数的VQE收敛性"""
    print(f"\n{'='*50}")
    print(f"=== {num_qubits}比特系统VQE测试 ===")
    print(f"{'='*50}")
    
    # 创建哈密顿量
    N = num_qubits // 4
    H = create_hamiltonian('paper_4n_heisenberg', N=N, use_decomposed=True, device=device)
    print(f"哈密顿量项数: {len(H.terms)}")
    
    # 创建初态
    init_state = create_partial_ground_state_product(num_qubits, device)
    
    # 计算初始能量
    initial_energy = H.expectation(init_state)
    print(f"初态能量: {initial_energy.item():.6f}")
    
    # 测试不同的PQC结构
    pqc_configs = [
        ('adaptive', build_pqc_adaptive),
        ('u_cz_2', lambda n, d: build_pqc_u_cz(n, num_layers=num_qubits*2//3, device=d)),
        ('u_cz_3', lambda n, d: build_double_cz_pqc(n, num_layers=num_qubits*2//3, device=d)),
        ('rx_rz_2', lambda n, d: build_pqc_rx_rz_cnot(n, num_layers=num_qubits*2//3, device=d)),
        ('rx_rz_3', lambda n, d: build_pqc_rx_rz_cnot(n, num_layers=num_qubits*2//3, device=d)),
        ('alternating_2', lambda n, d: build_pqc_alternating(n, num_layers=num_qubits*2//3, device=d)),
        ('alternating_3', lambda n, d: build_pqc_alternating(n, num_layers=num_qubits*2//3, device=d))
    ]
    
    results = {}
    
    for config_name, pqc_builder in pqc_configs:
        print(f"\n--- 测试 {config_name} 配置 ---")
        
        try:
            # 创建PQC
            pqc = pqc_builder(num_qubits, device)
            
            # 分析电路结构
            structure_info = analyze_pqc_structure(pqc, num_qubits)
            
            # 创建VQE
            vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01}, store_best_state=True)
            
            # 进行优化
            print(f"开始VQE优化...")
            start_time = time.time()
            
            result = vqe.optimize(
                num_iterations=500,  # 减少迭代次数以加快测试
                input_state=init_state,
                convergence_threshold=1e-6,
                patience=50
            )
            
            vqe_time = time.time() - start_time
            
            # 记录结果
            results[config_name] = {
                'initial_energy': initial_energy.item(),
                'final_energy': result['best_energy'],
                'energy_improvement': initial_energy.item() - result['best_energy'],
                'iterations': result['iterations'],
                'vqe_time': vqe_time,
                'structure_info': structure_info
            }
            
            print(f"最终能量: {result['best_energy']:.6f}")
            print(f"能量改善: {initial_energy.item() - result['best_energy']:.6f}")
            print(f"优化时间: {vqe_time:.2f}秒")
            print(f"迭代次数: {result['iterations']}")
            
        except Exception as e:
            print(f"配置 {config_name} 失败: {e}")
            results[config_name] = {'error': str(e)}
    
    return results

def test_12_qubit():
    """测试12比特系统"""
    device = torch.device('cpu')  # 使用CPU避免内存问题
    print(f"使用设备: {device}")
    
    return test_vqe_convergence(12, device)

def test_16_qubit():
    """测试16比特系统"""
    device = torch.device('cpu')  # 使用CPU避免内存问题
    print(f"使用设备: {device}")
    
    return test_vqe_convergence(16, device)

def analyze_results(results_12, results_16):
    """分析测试结果"""
    print(f"\n{'='*60}")
    print(f"=== 结果分析 ===")
    print(f"{'='*60}")
    
    for num_qubits, results in [('12比特', results_12), ('16比特', results_16)]:
        print(f"\n{num_qubits}系统结果:")
        print("-" * 40)
        
        # 按能量改善排序
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]['energy_improvement'], 
                              reverse=True)
        
        for i, (config_name, result) in enumerate(sorted_results):
            print(f"{i+1}. {config_name}:")
            print(f"   最终能量: {result['final_energy']:.6f}")
            print(f"   能量改善: {result['energy_improvement']:.6f}")
            print(f"   参数数量: {result['structure_info']['parameter_count']}")
            print(f"   门数量: {result['structure_info']['gate_count']}")
            print(f"   估算深度: {result['structure_info']['estimated_depth']}")
            print(f"   优化时间: {result['vqe_time']:.2f}秒")
            print()

def main():
    """主函数"""
    print("=== 12比特和16比特VQE收敛性测试 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试12比特系统
    print("\n" + "="*60)
    print("开始12比特系统测试")
    print("="*60)
    results_12 = test_12_qubit()
    
    # 测试16比特系统
    print("\n" + "="*60)
    print("开始16比特系统测试")
    print("="*60)
    results_16 = test_16_qubit()
    
    # 分析结果
    analyze_results(results_12, results_16)
    
    return {
        '12_qubit': results_12,
        '16_qubit': results_16
    }

if __name__ == "__main__":
    result = main()
    print("\n=== 测试完成 ===") 