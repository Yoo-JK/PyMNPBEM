# point.py - 점들의 집합 클래스
"""
Point class for collection of points.
Converted from MATLAB MNPBEM @point class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Callable


class Point:
    """점 클래스 - 점들의 집합을 관리"""
    
    def __init__(self, pos: np.ndarray, vec: Optional[np.ndarray] = None):
        """
        Point 초기화
        
        Parameters:
        -----------
        pos : array, shape (n_points, 3)
            점들의 좌표
        vec : array, shape (n_points, 3) or None
            위치에서의 기저 벡터 {x, y, z}
            None으로 설정하면 기본값 사용
        """
        self.pos = np.array(pos)
        
        # 기저 벡터
        if vec is not None and len(vec) > 0:
            self.vec = [np.array(vec)]
        else:
            n = self.pos.shape[0]
            # 기본 직교 기저 벡터들
            self.vec = [
                np.tile([1, 0, 0], (n, 1)),  # x 방향
                np.tile([0, 1, 0], (n, 1)),  # y 방향
                np.tile([0, 0, 1], (n, 1))   # z 방향
            ]
    
    def __str__(self):
        """문자열 표현"""
        return f"Point:\n  pos: {self.pos.shape}\n  vec: {len(self.vec)} basis vectors"
    
    def select(self, method: str, *args) -> Union['Point', Tuple['Point', 'Point']]:
        """점들 선택"""
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        z = self.pos[:, 2]
        
        # 점들 선택
        if method in ['ind', 'index']:
            indices = args[0]
        elif method in ['carfun', 'cartfun']:
            indices = np.where(args[0](x, y, z))[0]
        elif method == 'polfun':
            phi = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2)
            indices = np.where(args[0](phi, r, z))[0]
        elif method == 'sphfun':
            r = np.sqrt(x**2 + y**2 + z**2)
            phi = np.arctan2(y, x)
            theta = np.arccos(z / (r + 1e-10))
            indices = np.where(args[0](phi, np.pi/2 - theta, r))[0]
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # 선택된 점들을 가진 객체
        obj1 = self._compress(indices)
        
        # 보완 집합이 요청된 경우
        if len(args) > 1 or method != 'index':
            complement_indices = np.setdiff1d(np.arange(self.pos.shape[0]), indices)
            obj2 = self._compress(complement_indices)
            return obj1, obj2
        
        return obj1
    
    def _compress(self, indices: np.ndarray) -> 'Point':
        """점들 압축 및 선택된 점들만 유지"""
        new_point = Point(self.pos[indices])
        
        # 기저 벡터들도 선택
        new_point.vec = [
            self.vec[0][indices],
            self.vec[1][indices], 
            self.vec[2][indices]
        ]
        
        return new_point
    
    def __add__(self, other):
        """점들을 수직으로 연결"""
        return self.vertcat(other)
    
    def vertcat(self, *others):
        """점들을 수직으로 연결"""
        new_pos = self.pos.copy()
        new_vec = [vec.copy() for vec in self.vec]
        
        for other in others:
            new_pos = np.vstack([new_pos, other.pos])
            for dim in range(3):
                new_vec[dim] = np.vstack([new_vec[dim], other.vec[dim]])
        
        result = Point(new_pos)
        result.vec = new_vec
        return result
    
    @property
    def nvec(self) -> np.ndarray:
        """법선 벡터 (호환성을 위해 z-성분)"""
        return self.vec[2]
    
    @property
    def tvec(self) -> List[np.ndarray]:
        """접선 벡터 (x, y-성분)"""
        return self.vec[:2]
    
    @property
    def tvec1(self) -> np.ndarray:
        """첫 번째 접선 벡터 (x-성분)"""
        return self.vec[0]
    
    @property
    def tvec2(self) -> np.ndarray:
        """두 번째 접선 벡터 (y-성분)"""
        return self.vec[1]
    
    @property
    def size(self) -> int:
        """점들의 수"""
        return self.pos.shape[0]
    
    @property
    def n(self) -> int:
        """점들의 수 (size와 동일)"""
        return self.size
    
    def __len__(self):
        """점들의 수"""
        return self.size
    
    def __getitem__(self, key):
        """인덱싱 지원"""
        if isinstance(key, (int, slice, np.ndarray)):
            return self._compress(key)
        else:
            raise TypeError("Index must be int, slice, or numpy array")
    
    def copy(self):
        """Point 객체 복사"""
        new_point = Point(self.pos.copy())
        new_point.vec = [vec.copy() for vec in self.vec]
        return new_point