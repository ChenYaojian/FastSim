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

# 创建量子电路
from fastsim.circuit import QuantumCircuit
from fastsim.state import QuantumState

# 创建2量子比特电路
circuit = QuantumCircuit(2)
circuit.h(0)  # Hadamard门
circuit.cx(0, 1)  # CNOT门

# 执行电路
state = QuantumState(2)
result_state = circuit.execute(state)
print(result_state)
```

### VQE算法使用

```python
from fastsim.vqe import VQE
from fastsim.hamiltonian import create_heisenberg_hamiltonian

# 创建哈密顿量
hamiltonian = create_heisenberg_hamiltonian(4)

# 创建VQE实例
vqe = VQE(hamiltonian, num_qubits=4)

# 运行VQE
energy, params = vqe.optimize()
print(f"基态能量: {energy}")
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
black src/ tests/
```

## 卸载

```bash
pip uninstall fastsim
``` 