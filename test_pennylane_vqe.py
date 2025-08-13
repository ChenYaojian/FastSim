import torch
import time
import numpy as np
import pennylane as qml
from fastsim.hamiltonian import create_hamiltonian
from fastsim.circuit import load_gates_from_config

def create_heisenberg_hamiltonian_pennylane(num_qubits, device=None):
    """
    使用PennyLane创建Heisenberg哈密顿量
    对应paper_4n_heisenberg模型
    """
    N = num_qubits // 4
    
    # 创建4比特Heisenberg哈密顿量的Pauli项
    def create_4qubit_heisenberg():
        # 4比特Heisenberg模型的项
        terms = []
        coeffs = []
        
        # 最近邻相互作用
        for i in range(4):
            j = (i + 1) % 4
            # XX项
            pauli = ['I'] * 4
            pauli[i] = 'X'
            pauli[j] = 'X'
            terms.append(''.join(pauli))
            coeffs.append(1.0)
            
            # YY项
            pauli = ['I'] * 4
            pauli[i] = 'Y'
            pauli[j] = 'Y'
            terms.append(''.join(pauli))
            coeffs.append(1.0)
            
            # ZZ项
            pauli = ['I'] * 4
            pauli[i] = 'Z'
            pauli[j] = 'Z'
            terms.append(''.join(pauli))
            coeffs.append(1.0)
        
        return terms, coeffs
    
    # 获取4比特项
    terms_4, coeffs_4 = create_4qubit_heisenberg()
    
    # 扩展到N*4比特系统
    all_terms = []
    all_coeffs = []
    
    for n in range(N):
        for term, coeff in zip(terms_4, coeffs_4):
            # 在4比特块内应用项
            full_term = ['I'] * num_qubits
            for i, pauli in enumerate(term):
                if pauli != 'I':
                    full_term[n * 4 + i] = pauli
            all_terms.append(''.join(full_term))
            all_coeffs.append(coeff)
    
    return all_terms, all_coeffs

def build_pqc_adaptive_pennylane(num_qubits, device=None):
    """使用PennyLane构建自适应PQC"""
    
    def circuit(params):
        # 参数化单比特门
        param_idx = 0
        for i in range(num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # 纠缠层
        for i in range(0, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        
        for i in range(1, num_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    
    # 设置参数数量
    circuit.num_params = 2 * num_qubits
    
    return circuit

def build_pqc_u_cz_pennylane(num_qubits, num_layers=1, device=None):
    """使用PennyLane构建U+CZ PQC"""
    
    def circuit(params):
        param_idx = 0
        
        for layer in range(num_layers):
            # 单比特U门
            for i in range(num_qubits):
                qml.U3(params[param_idx], params[param_idx + 1], params[param_idx + 2], wires=i)
                param_idx += 3
            
            # CZ纠缠
            for i in range(0, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            
            for i in range(1, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
    
    # 设置参数数量
    circuit.num_params = 3 * num_qubits * num_layers
    
    return circuit

def build_pqc_rx_rz_cnot_pennylane(num_qubits, num_layers=1, device=None):
    """使用PennyLane构建RX+RZ+CNOT PQC"""
    
    def circuit(params):
        param_idx = 0
        
        for layer in range(num_layers):
            # RX门
            for i in range(num_qubits):
                qml.RX(params[param_idx], wires=i)
                param_idx += 1
            
            # RZ门
            for i in range(num_qubits):
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1
            
            # CNOT纠缠
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            
            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    
    # 设置参数数量
    circuit.num_params = 2 * num_qubits * num_layers
    
    return circuit

def build_pqc_alternating_pennylane(num_qubits, num_layers=1, device=None):
    """使用PennyLane构建交替PQC"""
    
    def circuit(params):
        param_idx = 0
        
        for layer in range(num_layers):
            # 偶数比特RX
            for i in range(0, num_qubits, 2):
                qml.RX(params[param_idx], wires=i)
                param_idx += 1
            
            # 奇数比特RZ
            for i in range(1, num_qubits, 2):
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1
            
            # CNOT纠缠
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            
            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    
    # 设置参数数量
    circuit.num_params = num_qubits * num_layers
    
    return circuit

def build_double_cz_pqc_pennylane(num_qubits, num_layers=1, device=None):
    """使用PennyLane构建双CZ PQC"""
    
    def circuit(params):
        param_idx = 0
        
        for layer in range(num_layers):
            # 单比特U门
            for i in range(num_qubits):
                qml.U3(params[param_idx], params[param_idx + 1], params[param_idx + 2], wires=i)
                param_idx += 3
            
            # 双CZ纠缠
            for i in range(0, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
                if i + 2 < num_qubits:
                    qml.CZ(wires=[i + 1, i + 2])
            
            for i in range(1, num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
                if i + 2 < num_qubits:
                    qml.CZ(wires=[i + 1, i + 2])
    
    # 设置参数数量
    circuit.num_params = 3 * num_qubits * num_layers
    
    return circuit

def create_hamiltonian_pennylane(num_qubits):
    """创建PennyLane哈密顿量"""
    # 简化的Heisenberg哈密顿量，只考虑最近邻相互作用
    coeffs = []
    observables = []
    
    for i in range(num_qubits - 1):
        # XX项
        coeffs.append(1.0)
        observables.append(qml.PauliX(i) @ qml.PauliX(i + 1))
        
        # YY项
        coeffs.append(1.0)
        observables.append(qml.PauliY(i) @ qml.PauliY(i + 1))
        
        # ZZ项
        coeffs.append(1.0)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    
    return qml.Hamiltonian(coeffs, observables)

def analyze_pqc_structure_pennylane(circuit, num_qubits):
    """分析PennyLane PQC电路结构"""
    print(f"\n=== PennyLane PQC电路结构分析 ===")
    print(f"量子比特数: {num_qubits}")
    print(f"参数数量: {circuit.num_params}")
    
    # 估算门数量（基于参数数量）
    estimated_gates = circuit.num_params + num_qubits  # 参数化门 + 纠缠门
    
    return {
        'parameter_count': circuit.num_params,
        'gate_count': estimated_gates,
        'parametric_gates': circuit.num_params,
        'entanglement_gates': num_qubits,  # 估算
        'estimated_depth': estimated_gates // num_qubits if num_qubits > 0 else 0
    }

def vqe_optimize_pennylane(circuit, hamiltonian, num_qubits, num_iterations=500, lr=0.01, convergence_threshold=1e-6, patience=50):
    """使用PennyLane进行VQE优化"""
    
    @qml.qnode(qml.device("default.qubit", wires=num_qubits))
    def cost_fn(params):
        circuit(params)
        return qml.expval(hamiltonian)
    
    # 初始化参数
    num_params = circuit.num_params
    params = np.random.random(num_params) * 2 * np.pi
    
    # 优化器
    opt = qml.AdamOptimizer(stepsize=lr)
    
    best_energy = float('inf')
    best_params = params.copy()
    patience_counter = 0
    energies = []
    
    print(f"开始PennyLane VQE优化，参数数量: {num_params}")
    
    for iteration in range(num_iterations):
        params, energy = opt.step_and_cost(cost_fn, params)
        energies.append(energy)
        
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if iteration % 50 == 0:
            print(f"迭代 {iteration}: 能量 = {energy:.6f}")
        
        # 收敛检查
        if patience_counter >= patience:
            print(f"在迭代 {iteration} 收敛")
            break
    
    return {
        'best_energy': best_energy,
        'best_params': best_params,
        'iterations': len(energies),
        'energies': energies
    }

def test_vqe_convergence_pennylane(num_qubits):
    """测试指定比特数的PennyLane VQE收敛性"""
    print(f"\n{'='*50}")
    print(f"=== {num_qubits}比特系统PennyLane VQE测试 ===")
    print(f"{'='*50}")
    
    # 创建哈密顿量
    H_pennylane = create_hamiltonian_pennylane(num_qubits)
    print(f"PennyLane哈密顿量项数: {len(H_pennylane.coeffs)}")
    
    # 计算基态能量（用于对比）
    H_fastsim = create_hamiltonian('paper_4n_heisenberg', N=num_qubits//4, use_decomposed=True)
    H_matrix = H_fastsim.get_matrix()
    eigenvalues, _ = torch.linalg.eigh(H_matrix)
    ground_energy = eigenvalues[0].item()
    print(f"FastSV基态能量: {ground_energy:.6f}")
    
    # 测试不同的PQC结构
    pqc_configs = [
        ('adaptive', build_pqc_adaptive_pennylane),
        ('u_cz_2', lambda n, d: build_pqc_u_cz_pennylane(n, num_layers=n*2//3)),
        ('u_cz_3', lambda n, d: build_double_cz_pqc_pennylane(n, num_layers=n*2//3)),
        ('rx_rz_2', lambda n, d: build_pqc_rx_rz_cnot_pennylane(n, num_layers=n*2//3)),
        ('rx_rz_3', lambda n, d: build_pqc_rx_rz_cnot_pennylane(n, num_layers=n*2//3)),
        ('alternating_2', lambda n, d: build_pqc_alternating_pennylane(n, num_layers=n*2//3)),
        ('alternating_3', lambda n, d: build_pqc_alternating_pennylane(n, num_layers=n*2//3))
    ]
    
    results = {}
    
    for config_name, pqc_builder in pqc_configs:
        print(f"\n--- 测试 {config_name} 配置 ---")
        
        try:
            # 创建PQC
            pqc = pqc_builder(num_qubits, None)
            
            # 分析电路结构
            structure_info = analyze_pqc_structure_pennylane(pqc, num_qubits)
            
            # 进行优化
            print(f"开始PennyLane VQE优化...")
            start_time = time.time()
            
            result = vqe_optimize_pennylane(
                pqc, 
                H_pennylane,
                num_qubits,
                num_iterations=500,
                lr=0.01,
                convergence_threshold=1e-6,
                patience=50
            )
            
            vqe_time = time.time() - start_time
            
            # 记录结果
            results[config_name] = {
                'ground_energy': ground_energy,
                'final_energy': result['best_energy'],
                'energy_error': abs(result['best_energy'] - ground_energy),
                'iterations': result['iterations'],
                'vqe_time': vqe_time,
                'structure_info': structure_info
            }
            
            print(f"最终能量: {result['best_energy']:.6f}")
            print(f"与基态能量误差: {abs(result['best_energy'] - ground_energy):.6f}")
            print(f"优化时间: {vqe_time:.2f}秒")
            print(f"迭代次数: {result['iterations']}")
            
        except Exception as e:
            print(f"配置 {config_name} 失败: {e}")
            results[config_name] = {'error': str(e)}
    
    return results

def test_12_qubit_pennylane():
    """测试12比特系统"""
    print(f"使用PennyLane测试12比特系统")
    return test_vqe_convergence_pennylane(12)

def test_16_qubit_pennylane():
    """测试16比特系统"""
    print(f"使用PennyLane测试16比特系统")
    return test_vqe_convergence_pennylane(16)

def compare_results(fastsim_results, pennylane_results):
    """对比FastSV和PennyLane的结果"""
    print(f"\n{'='*80}")
    print(f"=== FastSV vs PennyLane 结果对比 ===")
    print(f"{'='*80}")
    
    for num_qubits, (fastsim_key, pennylane_key) in [('12比特', ('12_qubit', '12_qubit')), ('16比特', ('16_qubit', '16_qubit'))]:
        print(f"\n{num_qubits}系统对比:")
        print("-" * 60)
        
        fastsim_data = fastsim_results.get(fastsim_key, {})
        pennylane_data = pennylane_results.get(pennylane_key, {})
        
        # 找出共同的配置
        common_configs = set(fastsim_data.keys()) & set(pennylane_data.keys())
        common_configs = {k for k in common_configs if 'error' not in fastsim_data.get(k, {}) and 'error' not in pennylane_data.get(k, {})}
        
        if not common_configs:
            print("没有可对比的配置")
            continue
        
        print(f"{'配置':<15} {'FastSV能量':<12} {'PennyLane能量':<15} {'误差':<10} {'时间比':<10}")
        print("-" * 80)
        
        for config in sorted(common_configs):
            fastsim_result = fastsim_data[config]
            pennylane_result = pennylane_data[config]
            
            fastsim_energy = fastsim_result.get('final_energy', 0)
            pennylane_energy = pennylane_result.get('final_energy', 0)
            energy_diff = abs(fastsim_energy - pennylane_energy)
            
            fastsim_time = fastsim_result.get('vqe_time', 1)
            pennylane_time = pennylane_result.get('vqe_time', 1)
            time_ratio = fastsim_time / pennylane_time if pennylane_time > 0 else 0
            
            print(f"{config:<15} {fastsim_energy:<12.6f} {pennylane_energy:<15.6f} {energy_diff:<10.6f} {time_ratio:<10.2f}")

def main():
    """主函数"""
    print("=== PennyLane VQE实现测试 ===")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    
    # 测试12比特系统
    print("\n" + "="*60)
    print("开始12比特系统PennyLane测试")
    print("="*60)
    results_12_pennylane = test_12_qubit_pennylane()
    
    # 测试16比特系统
    print("\n" + "="*60)
    print("开始16比特系统PennyLane测试")
    print("="*60)
    results_16_pennylane = test_16_qubit_pennylane()
    
    # 运行FastSV测试以进行对比
    print("\n" + "="*60)
    print("运行FastSV测试进行对比")
    print("="*60)
    
    # 导入并运行FastSV测试
    from test_12_16_qubit_vqe import test_12_qubit, test_16_qubit
    
    results_12_fastsim = test_12_qubit()
    results_16_fastsim = test_16_qubit()
    
    # 对比结果
    fastsim_results = {
        '12_qubit': results_12_fastsim,
        '16_qubit': results_16_fastsim
    }
    
    pennylane_results = {
        '12_qubit': results_12_pennylane,
        '16_qubit': results_16_pennylane
    }
    
    compare_results(fastsim_results, pennylane_results)
    
    return {
        'fastsim': fastsim_results,
        'pennylane': pennylane_results
    }

if __name__ == "__main__":
    result = main()
    print("\n=== PennyLane测试完成 ===") 