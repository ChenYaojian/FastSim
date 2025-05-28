# FastSim - 快速量子态向量模拟器

FastSim是一个高效的量子电路模拟器，支持量子态向量计算、VQE算法和各种量子算法。该项目采用模块化设计，提供了完整的量子计算模拟框架。

## 📋 目录

- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [使用示例](#使用示例)
- [环境要求](#环境要求)
- [开发指南](#开发指南)

## 🚀 安装指南

### 快速安装

```bash
git clone https://gitlab.quantumsc.online/qlc/fastsim/
cd FastSim
pip install -e .
```

### 安装可选依赖

```bash
# 安装开发依赖
pip install -e .[dev]
# 安装文档依赖
pip install -e .[docs]
# 安装示例依赖
pip install -e .[examples]
```

## 🎯 快速开始

### 基本使用

```python
import fastsim
from fastsim.circuit import Circuit, load_gates_from_config
from fastsim.state import StateVector

# 加载门配置
load_gates_from_config('configs/gates_config.json')

# 创建2量子比特电路
circuit = Circuit(2)
circuit.add_gate('H', [0])  # Hadamard门
circuit.add_gate('CNOT', [0, 1])  # CNOT门

# 执行电路
state = StateVector(2)
result_state = circuit(state.get_state_vector().unsqueeze(0))[0]
print(result_state)
```

### VQE算法使用

```python
from fastsim.vqe import VQE, PQC
from fastsim.hamiltonian import create_hamiltonian
from fastsim.circuit import load_gates_from_config

# 加载门配置
load_gates_from_config('configs/gates_config.json')

# 创建哈密顿量
hamiltonian = create_hamiltonian('heisenberg', num_qubits=4)

# 创建参数化量子电路
pqc = PQC(4)
pqc.add_parametric_gate('RX', [0], [0.5])
pqc.add_parametric_gate('RY', [1], [0.3])
pqc.add_parametric_gate('CNOT', [0, 1])

# 创建VQE实例
vqe = VQE(pqc, hamiltonian)

# 运行VQE
result = vqe.optimize(num_iterations=100, convergence_threshold=1e-6)
print(f"基态能量: {result['final_energy']}")
```

## 🔧 核心模块

### 量子电路模块

量子电路模块提供了完整的量子电路构建和执行功能。

#### 已支持功能：
- **基本量子门**: H, X, Y, Z, CNOT, CZ, RX, RY, RZ等
- **自定义量子门注册**: 支持注册自定义量子门
- **量子电路构建**: 支持添加、删除、修改门
- **量子电路可视化**: 基本电路可视化功能
- **参数化量子门**: 支持参数化门和优化

#### 待支持功能：
- 常见量子算法对应的电路（QFT, Grover, QAOA等）
- 更高级的电路优化
- 电路分解和编译

### 量子态模块

量子态模块提供了量子态的表示和操作功能。

#### 功能特性：
- **量子态表示**: 支持复数向量表示
- **态向量操作**: 支持态的加法、乘法、归一化
- **测量操作**: 支持投影测量和POVM测量
- **态演化**: 支持量子态的演化计算

### 哈密顿量模块

哈密顿量模块是FastSim的核心模块之一，提供了统一的哈密顿量处理框架。

#### 🎯 主要特性

1. **统一接口**: 所有哈密顿量都继承自`Hamiltonian`基类
2. **多种表示形式**: 支持密集矩阵、稀疏矩阵、分解形式
3. **字符串构建**: 支持从字符串表达式构建哈密顿量
4. **内存效率**: 分解形式显著减少内存占用
5. **数学正确性**: 保持厄米性质和物理一致性

#### 🚀 支持的哈密顿量类型

1. **HeisenbergHamiltonian**: 海森堡模型
2. **IsingHamiltonian**: 一维横场Ising模型
3. **HubbardHamiltonian**: Hubbard模型
4. **Quasi1DAFMHamiltonian**: 准一维反铁磁模型
5. **Paper4NHeisenbergHamiltonian**: arXiv:2007.10917v2结构的4*N海森堡模型

#### 📝 使用方式

##### 预定义类型：
```python
from fastsim.hamiltonian import create_hamiltonian

# 海森堡哈密顿量
H = create_hamiltonian('heisenberg', num_qubits=4, J=1.0, h=0.5)

# Ising哈密顿量
H = create_hamiltonian('ising', num_qubits=4, J=1.0, h=0.5)

# Paper 4N Heisenberg哈密顿量
H = create_hamiltonian('paper_4n_heisenberg', N=2)
```

##### 字符串表达式：
```python
# Ising字符串
H = create_hamiltonian("-1.0*ZZ[0,1] -0.5*X[0] -0.5*X[1]", num_qubits=2)

# Heisenberg字符串
H = create_hamiltonian("1.0*XX[0,1] + 1.0*YY[0,1] + 1.0*ZZ[0,1] + 0.5*Z[0] + 0.5*Z[1]", num_qubits=2)

# 自动推断量子比特数
H = create_hamiltonian("-1.0*ZZ[0,1] -0.5*X[0] -0.5*X[1]")  # 自动推断为2量子比特
```

#### ⚡ 性能优势

1. **内存效率**:
   - 分解形式：只存储泡利算符，不存储大矩阵
   - 压缩表示：避免存储完整的张量积矩阵
   - 对于大系统，内存占用显著减少

2. **计算效率**:
   - 只对相关量子比特进行矩阵乘法
   - 利用期望值的分配律，可以并行计算
   - 支持批处理操作

3. **可扩展性**:
   - 易于添加新的哈密顿量模型
   - 支持自定义字符串格式
   - 完全兼容PyTorch框架

### VQE算法模块

VQE（变分量子本征求解器）模块提供了完整的量子-经典混合算法实现。

#### 功能特性：
- **参数化量子电路**: 支持各种PQC结构
- **优化算法**: 支持多种优化器（Adam, SGD等）
- **随机重启**: 支持随机重启优化策略
- **状态保存**: 支持最佳状态的保存和加载
- **收敛监控**: 实时监控优化过程

#### 支持的PQC结构：
- **自适应PQC**: 根据系统大小自动调整
- **U+CZ结构**: 通用门+CZ纠缠层
- **RX+RZ+CNOT结构**: 旋转门+CNOT纠缠层
- **交替纠缠结构**: 交替的纠缠模式

### 采样模块

采样模块提供了量子态的采样和测量功能。

#### 功能特性：
- **量子态采样**: 支持对量子态进行采样
- **测量操作**: 支持各种测量方式
- **期望值计算**: 支持可观测量期望值的计算
- **统计分析**: 提供采样结果的统计分析

## 📊 使用示例

### 完整的VQE工作流

```python
import torch
from fastsim.vqe import VQE, build_pqc_adaptive
from fastsim.hamiltonian import create_hamiltonian
from fastsim.circuit import load_gates_from_config

# 加载门配置
load_gates_from_config('configs/gates_config.json')

# 创建哈密顿量
H = create_hamiltonian('heisenberg', num_qubits=4, J=1.0, h=0.5)

# 创建PQC
pqc = build_pqc_adaptive(4)

# 创建VQE实例
vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})

# 创建初始态
init_state = torch.zeros(16, dtype=torch.complex64)
init_state[0] = 1.0
init_state = init_state.unsqueeze(0)

# 运行VQE优化
result = vqe.optimize(
    num_iterations=1000,
    input_state=init_state,
    convergence_threshold=1e-6,
    patience=100
)

print(f"优化完成!")
print(f"最终能量: {result['final_energy']:.6f}")
print(f"最佳能量: {result['best_energy']:.6f}")
print(f"迭代次数: {result['iterations']}")
```

### 字符串构建哈密顿量

```python
from fastsim.hamiltonian import create_hamiltonian

# 使用字符串构建Ising哈密顿量
ising_string = "-1.0*ZZ[0,1] -0.5*X[0] -0.5*X[1]"
H_ising = create_hamiltonian(ising_string, num_qubits=2)

# 使用字符串构建Heisenberg哈密顿量
heisenberg_string = "1.0*XX[0,1] + 1.0*YY[0,1] + 1.0*ZZ[0,1] + 0.5*Z[0] + 0.5*Z[1]"
H_heisenberg = create_hamiltonian(heisenberg_string, num_qubits=2)

# 验证一致性
matrix_ising = H_ising.get_matrix()
matrix_heisenberg = H_heisenberg.get_matrix()
print(f"Ising矩阵形状: {matrix_ising.shape}")
print(f"Heisenberg矩阵形状: {matrix_heisenberg.shape}")
```

### 随机重启优化

```python
from fastsim.vqe import VQE, build_pqc_adaptive
from fastsim.hamiltonian import create_hamiltonian

# 创建哈密顿量和PQC
H = create_hamiltonian('paper_4n_heisenberg', N=1)
pqc = build_pqc_adaptive(4)

# 创建VQE实例
vqe = VQE(pqc, H, optimizer_kwargs={'lr': 0.01})

# 运行随机重启优化
result = vqe.optimize_with_random_restarts(
    num_epochs=5,
    iterations_per_epoch=200,
    convergence_threshold=1e-6,
    patience=100,
    random_scale=0.1,
    use_seed=True
)

print(f"随机重启优化完成!")
print(f"最终能量: {result['final_energy']:.6f}")
print(f"最佳能量: {result['best_energy']:.6f}")
print(f"总迭代次数: {result['total_iterations']}")
```

## 🔧 环境要求

- Python >= 3.10
- NumPy >= 1.26.4
- PyTorch >= 2.1.0
- SciPy >= 1.11.0
- Matplotlib >= 3.7.0

## 🛠️ 开发指南

### 开发环境设置

```bash
# 创建虚拟环境
python -m venv fastsim_env
source fastsim_env/bin/activate  # Linux/Mac
# 或
fastsim_env\Scripts\activate  # Windows

# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest

# 代码格式化
black fastsim/ tests/
```

### 故障排除

#### 常见问题

1. **PyTorch安装失败**
   ```bash
   # 使用conda安装PyTorch
   conda install pytorch torchvision torchaudio -c pytorch
   ```

2. **CUDA支持**
   ```bash
   # 安装CUDA版本的PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **权限问题**
   ```bash
   # 使用用户安装
   pip install --user -e .
   ```

### 代码结构

```
fastsim/
├── __init__.py          # 包初始化
├── circuit.py           # 量子电路模块
├── state.py             # 量子态模块
├── hamiltonian.py       # 哈密顿量模块
├── vqe.py              # VQE算法模块
├── sampling.py          # 采样模块
└── tool.py              # 工具函数

tests/                   # 测试文件
├── test_circuit.py
├── test_vqe.py
├── test_hamiltonian.py
└── ...

configs/                 # 配置文件
├── gates_config.json    # 门定义配置
└── ...

run/                     # 运行脚本
├── standard_vqe_sampling.py
└── ...
```

## 📄 许可证

本项目采用MIT许可证。

## 🤝 贡献者

- Yaojian Chen (yj-chen21@mails.tsinghua.edu.cn)

## 📞 联系方式

如有问题或建议，请联系：
- 邮箱：yj-chen21@mails.tsinghua.edu.cn
- 项目地址：https://gitlab.quantumsc.online/qlc/fastsim/

---

**FastSim** - 让量子计算更简单、更高效！ 
