#!/usr/bin/env python3
"""
测试HI电路的性能
测试arXiv:2007.10917v2论文中提出的HI电路结构
"""

import sys
import os
import torch
import time
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from fastsim.vqe import (
    build_pqc_hi_paper, 
    test_hi_circuit_performance, 
    compare_circuit_architectures,
    create_paper_4N_heisenberg_hamiltonian,
    get_hf_init_state,
    VQE
)
from fastsim.circuit import load_gates_from_config


def main():
    """主函数"""
    print("=" * 80)
    print("HI电路性能测试")
    print("测试arXiv:2007.10917v2论文中的HI电路结构")
    print("=" * 80)
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    print("✓ 门配置已加载")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 测试1: HI电路在不同cycle数下的性能
    print("\n" + "=" * 80)
    print("测试1: HI电路在不同cycle数下的性能")
    print("=" * 80)
    
    try:
        hi_results = test_hi_circuit_performance()
        
        # 保存结果
        results_file = "hi_circuit_results.json"
        with open(results_file, 'w') as f:
            json.dump(hi_results, f, indent=2, default=str)
        print(f"\n✓ 结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"❌ HI电路性能测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 比较不同电路架构
    print("\n" + "=" * 80)
    print("测试2: 比较不同电路架构的性能")
    print("=" * 80)
    
    try:
        comparison_results = compare_circuit_architectures()
        
        # 保存比较结果
        comparison_file = "circuit_comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\n✓ 比较结果已保存到: {comparison_file}")
        
    except Exception as e:
        print(f"❌ 电路架构比较失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


def test_single_hi_circuit():
    """测试单个HI电路的基本功能"""
    print("测试单个HI电路的基本功能...")
    
    # 加载门配置
    load_gates_from_config("configs/gates_config.json")
    print("✓ 门配置已加载")
    
    # 创建4比特HI电路
    pqc = build_pqc_hi_paper(4, num_cycles=2)
    print(f"✓ HI电路创建成功，参数数量: {pqc.parameter_count}")
    
    # 创建哈密顿量
    H = create_paper_4N_heisenberg_hamiltonian(1)  # N=1, 4比特
    print(f"✓ 哈密顿量创建成功，维度: {H.shape}")
    
    # 创建VQE
    vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})
    print("✓ VQE创建成功")
    
    # 测试前向传播
    hf_state = get_hf_init_state(4)
    energy = vqe.forward(hf_state)
    print(f"✓ 前向传播成功，初始能量: {energy.item():.6f}")
    
    # 测试一步优化
    energy_after_step = vqe.optimize_step(hf_state)
    print(f"✓ 优化步骤成功，能量: {energy_after_step:.6f}")
    
    print("✓ 所有基本功能测试通过!")


if __name__ == "__main__":
    # 首先测试基本功能
    print("测试基本功能...")
    test_single_hi_circuit()
    
    # 然后运行完整测试
    main() 