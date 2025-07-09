#!/usr/bin/env python3
"""
FastSim - 快速量子态向量模拟器
一个高效的量子电路模拟器，支持量子态向量计算和VQE算法
"""

from setuptools import setup, find_packages
import os

# 读取README文件作为长描述
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="fastsim",
    version="0.1.0",
    author="Yaojian Chen",
    author_email="yj-chen21@mails.tsinghua.edu.cn",
    description="快速量子态向量模拟器 - 高效的量子电路模拟和VQE算法实现",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitlab.quantumsc.online/qlc/fastsim/",
    project_urls={
        "Bug Tracker": "https://gitlab.quantumsc.online/qlc/fastsim/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="quantum computing, quantum simulation, quantum circuits, VQE, quantum algorithms",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.4",
        "torch>=2.1.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fastsim=fastsim.tool:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    zip_safe=False,
) 