# 量子电路执行脚本

这个目录包含了用于在SLURM集群上运行量子电路的提交脚本。

## 文件说明

- `submit_circuit.sh`: SLURM提交脚本
- `circuit.json`: 电路定义文件
- `gates_config.json`: 门定义配置文件
- `output.bin`: 输出文件（由脚本生成）
- `circuit_*.log`: 标准输出日志
- `circuit_*.err`: 错误日志

## 使用方法

1. 准备输入文件：
   - 将电路定义保存为`circuit.json`
   - 将门定义保存为`gates_config.json`

2. 提交作业：
```bash
sbatch submit_circuit.sh
```

3. 查看作业状态：
```bash
squeue -u $USER
```

4. 查看输出：
```bash
# 查看标准输出
cat logs/circuit_<job_id>.log

# 查看错误信息
cat logs/circuit_<job_id>.err
```

## 配置说明

当前配置：
- 使用debug分区
- 单节点单进程运行
- 最大运行时间30分钟
- 使用state_vector表示
- 使用complex64数据类型

如需修改配置，请编辑`submit_circuit.sh`中的相应参数。 