import torch
import json
import argparse
from typing import Union, Dict
from circuit import Circuit, load_gates_from_config
from state import StateVector, StateType

def execute_circuit(circuit_path: str, 
                   gates_config_path: str,
                   output_path: str,
                   state_type: Union[str, StateType] = StateType.STATE_VECTOR,
                   dtype: torch.dtype = torch.complex64) -> None:
    """执行量子电路并将结果保存到文件
    
    Args:
        circuit_path: 电路JSON文件路径
        gates_config_path: 门定义配置文件路径
        output_path: 输出文件路径
        state_type: 量子态类型，默认为STATE_VECTOR
        dtype: 数据类型，默认为torch.complex64
    """
    # 加载门定义
    load_gates_from_config(gates_config_path)
    
    # 从JSON加载电路
    circuit = Circuit.from_json(circuit_path)
    
    # 创建初始态（|0⟩态）
    state = StateVector.create_state(
        num_qubits=circuit.num_qubits,
        state_type=state_type,
        dtype=dtype
    )
    
    # 执行电路
    final_state = circuit.forward(state.get_state_vector())
    
    # 更新状态
    state.initialize(final_state)
    
    # 保存结果
    state.tofile(output_path)
    
    # 打印一些基本信息
    print(f"电路执行完成:")
    print(f"- 量子比特数: {circuit.num_qubits}")
    print(f"- 门操作数: {len(circuit.gates)}")
    print(f"- 输出文件: {output_path}")
    print(f"- 状态类型: {state_type}")
    print(f"- 数据类型: {dtype}")

def main():
    parser = argparse.ArgumentParser(description="执行量子电路并保存结果")
    parser.add_argument("--circuit_path", help="电路JSON文件路径")
    parser.add_argument("--gates_config_path", help="门定义配置文件路径")
    parser.add_argument("--output_path", help="输出文件路径")
    parser.add_argument("--state-type", default="state_vector", 
                      choices=["state_vector", "mps"],
                      help="量子态类型 (默认: state_vector)")
    parser.add_argument("--dtype", default="complex64",
                      choices=["complex64", "complex128"],
                      help="数据类型 (默认: complex64)")
    
    args = parser.parse_args()
    
    # 转换数据类型
    dtype_map = {
        "complex64": torch.complex64,
        "complex128": torch.complex128
    }
    dtype = dtype_map[args.dtype]
    
    # 执行电路
    execute_circuit(
        circuit_path=args.circuit_path,
        gates_config_path=args.gates_config_path,
        output_path=args.output_path,
        state_type=args.state_type,
        dtype=dtype
    )

if __name__ == "__main__":
    main() 