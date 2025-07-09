"""
FastSim - 快速量子态向量模拟器

一个高效的量子电路模拟器，支持量子态向量计算和VQE算法。

主要模块:
- circuit: 量子电路和量子门
- state: 量子态向量操作
- hamiltonian: 哈密顿量构建
- vqe: 变分量子本征求解器
- sampling: 量子态采样
- tool: 工具函数
"""

__version__ = "0.1.0"
__author__ = "Yaojian Chen"
__email__ = "yj-chen21@mails.tsinghua.edu.cn"

# 导入主要模块
from . import circuit
from . import state
from . import hamiltonian
from . import vqe
from . import sampling
from . import tool

# 导出主要类和函数
__all__ = [
    "circuit",
    "state", 
    "hamiltonian",
    "vqe",
    "sampling",
    "tool",
]
