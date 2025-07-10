# FastSim 安装指南

## 快速安装

git clone https://gitlab.quantumsc.online/qlc/fastsim/

### 从本地安装（开发模式）

```bash
cd FastSim

# 安装开发模式
pip install -e .
```

### 从本地安装（生产模式）

```bash
# 构建并安装
pip install .
```

### 安装可选依赖

```bash
# 安装开发依赖
pip install -e .[dev]

# 安装文档依赖
pip install -e .[docs]

# 安装示例依赖
pip install -e .[examples]

# 安装所有可选依赖
pip install -e .[dev,docs,examples]
```

## 使用示例

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
from fastsim.hamiltonian import create_heisenberg_hamiltonian
from fastsim.circuit import load_gates_from_config

# 加载门配置
load_gates_from_config('configs/gates_config.json')

# 创建哈密顿量
hamiltonian = create_heisenberg_hamiltonian(4)

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
```

## 环境要求

- Python >= 3.10
- NumPy >= 1.26.4
- PyTorch >= 2.1.0
- SciPy >= 1.11.0
- Matplotlib >= 3.7.0

## 故障排除

### 常见问题

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

## 开发环境设置

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

## 卸载

```bash
pip uninstall fastsim
``` 