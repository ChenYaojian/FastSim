#!/bin/bash

#SBATCH --job-name=quantum_circuit    # 作业名称
#SBATCH --partition=debug            # 使用debug分区
#SBATCH --nodes=1                    # 使用1个节点
#SBATCH --ntasks-per-node=1          # 每个节点1个任务
#SBATCH --cpus-per-task=1            # 每个任务1个CPU
#SBATCH --time=00:30:00              # 最大运行时间30分钟
#SBATCH --output=logs/circuit_%j.log      # 标准输出文件
#SBATCH --error=logs/circuit_%j.err       # 标准错误文件

# 加载必要的环境模块（如果需要）
# module load python/3.8

# 设置工作目录
cd ${SLURM_SUBMIT_DIR}

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# 运行量子电路
python fastsim/execute_circuit.py \
    --circuit_path "data/sim_cir_input_4layers.json" \
    --gates_config_path "configs/gates_config.json" \
    --output_path "data/sim_cir_output_4layers.bin" \
    --state-type "state_vector" \
    --dtype "complex64" 