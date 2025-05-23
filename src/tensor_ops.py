import torch
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

class TensorOps:
    """张量操作类
    
    提供基本的张量操作，包括：
    1. 张量收缩(contract)
    2. 张量重塑(reshape)
    3. 张量转置(permute)
    4. 张量切片(slice)
    5. 张量连接(concatenate)
    6. 张量分割(split)
    
    所有操作都支持CPU和GPU，并且可以方便地替换底层实现。
    """
    
    @staticmethod
    def contract(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                indices1: List[int], indices2: List[int]) -> torch.Tensor:
        """张量收缩操作
        
        通过矩阵乘法实现张量收缩，支持任意维度的张量。
        
        Args:
            tensor1: 第一个张量
            tensor2: 第二个张量
            indices1: 第一个张量中要收缩的维度索引
            indices2: 第二个张量中要收缩的维度索引
            
        Returns:
            torch.Tensor: 收缩后的张量
            
        Raises:
            ValueError: 如果收缩维度的大小不匹配
        """
        # 检查收缩维度的大小是否匹配
        for i, j in zip(indices1, indices2):
            if tensor1.shape[i] != tensor2.shape[j]:
                raise ValueError(f"Contracting dimensions must have the same size: "
                               f"tensor1[{i}]={tensor1.shape[i]}, tensor2[{j}]={tensor2.shape[j]}")
        
        # 获取张量的维度
        dim1 = len(tensor1.shape)
        dim2 = len(tensor2.shape)
        
        # 计算非收缩维度
        non_contract1 = [i for i in range(dim1) if i not in indices1]
        non_contract2 = [i for i in range(dim2) if i not in indices2]
        
        # 重塑张量为矩阵形式
        shape1 = [tensor1.shape[i] for i in non_contract1]
        shape2 = [tensor2.shape[i] for i in non_contract2]
        contract_size = tensor1.shape[indices1[0]]
        
        # 转置张量，使收缩维度在最后
        perm1 = non_contract1 + indices1
        perm2 = non_contract2 + indices2
        tensor1_perm = tensor1.permute(perm1)
        tensor2_perm = tensor2.permute(perm2)
        
        # 重塑为矩阵
        tensor1_mat = tensor1_perm.reshape(-1, contract_size)
        tensor2_mat = tensor2_perm.reshape(contract_size, -1)
        
        # 执行矩阵乘法
        result_mat = torch.matmul(tensor1_mat, tensor2_mat)
        
        # 重塑回张量形式
        result_shape = shape1 + shape2
        result = result_mat.reshape(result_shape)
        
        return result
    
    @staticmethod
    def reshape(tensor: torch.Tensor, new_shape: Tuple[int, ...]) -> torch.Tensor:
        """重塑张量形状
        
        Args:
            tensor: 输入张量
            new_shape: 新的形状
            
        Returns:
            torch.Tensor: 重塑后的张量
            
        Raises:
            ValueError: 如果新形状的元素总数与原始张量不匹配
        """
        if np.prod(tensor.shape) != np.prod(new_shape):
            raise ValueError(f"Total size of new shape must match original tensor: "
                           f"original={np.prod(tensor.shape)}, new={np.prod(new_shape)}")
        return tensor.reshape(new_shape)
    
    @staticmethod
    def permute(tensor: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """转置张量维度
        
        Args:
            tensor: 输入张量
            dims: 新的维度顺序
            
        Returns:
            torch.Tensor: 转置后的张量
            
        Raises:
            ValueError: 如果维度顺序无效
        """
        if len(dims) != len(tensor.shape):
            raise ValueError(f"Number of dimensions must match tensor rank: "
                           f"got {len(dims)}, expected {len(tensor.shape)}")
        if set(dims) != set(range(len(dims))):
            raise ValueError("Invalid dimension order")
        return tensor.permute(dims)
    
    @staticmethod
    def slice(tensor: torch.Tensor, indices: Dict[int, Union[int, slice]]) -> torch.Tensor:
        """张量切片操作
        
        Args:
            tensor: 输入张量
            indices: 维度索引字典，键为维度，值为索引或切片
            
        Returns:
            torch.Tensor: 切片后的张量
        """
        # 构建切片索引
        slice_indices = [slice(None)] * len(tensor.shape)
        for dim, idx in indices.items():
            slice_indices[dim] = idx
        return tensor[tuple(slice_indices)]
    
    @staticmethod
    def concatenate(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
        """连接张量
        
        Args:
            tensors: 要连接的张量列表
            dim: 连接的维度
            
        Returns:
            torch.Tensor: 连接后的张量
            
        Raises:
            ValueError: 如果张量形状不兼容
        """
        # 检查张量形状是否兼容
        shapes = [t.shape for t in tensors]
        for i in range(len(shapes)):
            for j in range(len(shapes[i])):
                if j != dim and shapes[i][j] != shapes[0][j]:
                    raise ValueError(f"Incompatible shapes for concatenation: {shapes}")
        return torch.cat(tensors, dim=dim)
    
    @staticmethod
    def split(tensor: torch.Tensor, sizes: Union[int, List[int]], dim: int) -> List[torch.Tensor]:
        """分割张量
        
        Args:
            tensor: 输入张量
            sizes: 分割大小，可以是整数（均匀分割）或列表（指定每个部分的大小）
            dim: 分割的维度
            
        Returns:
            List[torch.Tensor]: 分割后的张量列表
            
        Raises:
            ValueError: 如果分割大小无效
        """
        if isinstance(sizes, int):
            # 均匀分割
            size = tensor.shape[dim] // sizes
            if size * sizes != tensor.shape[dim]:
                raise ValueError(f"Tensor size {tensor.shape[dim]} is not divisible by {sizes}")
            sizes = [size] * sizes
        elif sum(sizes) != tensor.shape[dim]:
            raise ValueError(f"Sum of split sizes {sum(sizes)} does not match tensor size {tensor.shape[dim]}")
        
        return torch.split(tensor, sizes, dim=dim)


class MatrixOps:
    """矩阵操作类
    
    提供基本的矩阵操作，包括：
    1. 矩阵乘法(matmul)
    2. 奇异值分解(svd)
    3. 特征值分解(eig)
    4. 矩阵求逆(inverse)
    5. 矩阵转置(transpose)
    6. 矩阵范数(norm)
    
    所有操作都支持CPU和GPU，并且可以方便地替换底层实现。
    """
    
    @staticmethod
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            torch.Tensor: 矩阵乘积
            
        Raises:
            ValueError: 如果矩阵维度不兼容
        """
        if a.shape[-1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix dimensions: {a.shape} and {b.shape}")
        return torch.matmul(a, b)
    
    @staticmethod
    def svd(matrix: torch.Tensor, full_matrices: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """奇异值分解
        
        Args:
            matrix: 输入矩阵
            full_matrices: 是否返回完整的U和V矩阵
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (U, S, V)元组
        """
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    
    @staticmethod
    def eig(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """特征值分解
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特征值, 特征向量)元组
        """
        return torch.linalg.eig(matrix)
    
    @staticmethod
    def inverse(matrix: torch.Tensor) -> torch.Tensor:
        """矩阵求逆
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            torch.Tensor: 逆矩阵
            
        Raises:
            ValueError: 如果矩阵不可逆
        """
        try:
            return torch.linalg.inv(matrix)
        except RuntimeError as e:
            raise ValueError(f"Matrix is not invertible: {e}")
    
    @staticmethod
    def transpose(matrix: torch.Tensor) -> torch.Tensor:
        """矩阵转置
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            torch.Tensor: 转置矩阵
        """
        return matrix.transpose(-2, -1)
    
    @staticmethod
    def norm(matrix: torch.Tensor, ord: Optional[Union[int, float, str]] = None) -> torch.Tensor:
        """计算矩阵范数
        
        Args:
            matrix: 输入矩阵
            ord: 范数类型，可以是：
                - None: Frobenius范数
                - 'nuc': 核范数
                - 'fro': Frobenius范数
                - int/float: p-范数
                
        Returns:
            torch.Tensor: 矩阵范数
        """
        return torch.linalg.norm(matrix, ord=ord)
    
    @staticmethod
    def qr(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """QR分解
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Q, R)元组
        """
        return torch.linalg.qr(matrix)
    
    @staticmethod
    def cholesky(matrix: torch.Tensor) -> torch.Tensor:
        """Cholesky分解
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            torch.Tensor: 下三角矩阵L，满足A = LL^T
            
        Raises:
            ValueError: 如果矩阵不是正定的
        """
        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError as e:
            raise ValueError(f"Matrix is not positive definite: {e}")
    
    @staticmethod
    def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """求解线性方程组 Ax = b
        
        Args:
            a: 系数矩阵A
            b: 右侧向量或矩阵b
            
        Returns:
            torch.Tensor: 解x
            
        Raises:
            ValueError: 如果矩阵A是奇异的
        """
        try:
            return torch.linalg.solve(a, b)
        except RuntimeError as e:
            raise ValueError(f"Matrix is singular: {e}")
    
    @staticmethod
    def lstsq(a: torch.Tensor, b: torch.Tensor, rcond: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """最小二乘解
        
        Args:
            a: 系数矩阵A
            b: 右侧向量或矩阵b
            rcond: 截断参数
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            (解x, 残差, 秩, 奇异值)元组
        """
        return torch.linalg.lstsq(a, b, rcond=rcond)
