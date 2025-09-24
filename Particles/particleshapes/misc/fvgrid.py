# fvgrid.py - 2D 격자를 면-꼭짓점 구조로 변환
"""
Convert 2D grid to face-vertex structure.
Converted from MATLAB MNPBEM misc/fvgrid.m
"""

import numpy as np
from typing import Tuple, Optional, Union
from ..particle import Particle


def fvgrid(x: Union[np.ndarray, list], y: Union[np.ndarray, list], 
          triangles: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 격자를 면-꼭짓점 구조로 변환
    
    Parameters:
    -----------
    x : array_like
        격자의 x-좌표
    y : array_like  
        격자의 y-좌표
    triangles : str, optional
        'triangles'로 설정하면 사각형 대신 삼각형 사용
        
    Returns:
    --------
    verts : ndarray
        삼각화된 격자의 꼭짓점들 (PARTICLE에서 사용)
    faces : ndarray
        삼각화된 격자의 면들
    """
    x = np.array(x)
    y = np.array(y)
    
    # 1D 배열인 경우 meshgrid 생성
    if x.ndim == 1 or y.ndim == 1:
        x, y = np.meshgrid(x, y)
    
    # 격자 만들기
    use_triangles = triangles is not None and triangles == 'triangles'
    
    if use_triangles:
        faces, verts = _surf2patch_triangles(x, y, np.zeros_like(x))
    else:
        faces, verts = _surf2patch_quads(x, y, np.zeros_like(x))
    
    # 면 방향 뒤집기 (MATLAB의 fliplr 효과)
    faces = np.fliplr(faces)
    
    # 입자 생성 (norm='off' 효과로 보조 정보 계산하지 않음)
    p = Particle(verts, faces, norm='off')
    
    # 중점 추가
    p = p.midpoints('flat')
    
    # 면과 꼭짓점 추출
    verts = p.verts2 if p.verts2 is not None else p.verts
    faces = p.faces2 if p.faces2 is not None else p.faces
    
    return verts, faces


def _surf2patch_triangles(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """표면을 삼각형 패치로 변환 (MATLAB surf2patch와 유사)"""
    m, n = x.shape
    
    # 꼭짓점 생성
    verts = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # 삼각형 면 생성
    faces = []
    for i in range(m - 1):
        for j in range(n - 1):
            # 각 사각형을 두 개의 삼각형으로 분할
            v1 = i * n + j
            v2 = i * n + (j + 1)
            v3 = (i + 1) * n + j
            v4 = (i + 1) * n + (j + 1)
            
            # 첫 번째 삼각형: [v1, v2, v3]
            faces.append([v1, v2, v3])
            # 두 번째 삼각형: [v2, v4, v3]  
            faces.append([v2, v4, v3])
    
    return np.array(faces), verts


def _surf2patch_quads(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """표면을 사각형 패치로 변환 (MATLAB surf2patch와 유사)"""
    m, n = x.shape
    
    # 꼭짓점 생성
    verts = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # 사각형 면 생성
    faces = []
    for i in range(m - 1):
        for j in range(n - 1):
            # 사각형 면의 꼭짓점 인덱스 (시계반대방향)
            v1 = i * n + j
            v2 = i * n + (j + 1)
            v3 = (i + 1) * n + (j + 1)
            v4 = (i + 1) * n + j
            
            faces.append([v1, v2, v3, v4])
    
    return np.array(faces), verts