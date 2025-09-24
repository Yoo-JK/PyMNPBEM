# trirod.py - 막대 모양 입자의 면과 꼭짓점
"""
TRIROD - Faces and vertices for rod-shaped particle.
Converted from MATLAB MNPBEM trirod.m
"""

import numpy as np
from typing import List, Union, Optional
from ..particle import Particle
from .fvgrid import fvgrid
from .trispheresegment import trispheresegment


def trirod(diameter: float, height: float, n: Optional[List[int]] = None, 
          triangles: Optional[str] = None, **kwargs) -> Particle:
    """
    막대 모양 입자의 면과 꼭짓점 생성
    
    Parameters:
    -----------
    diameter : float
        막대의 지름
    height : float
        막대의 전체 높이(길이)
    n : list of int, optional
        이산화 점의 수 [nphi, ntheta, nz] (기본값: [15, 20, 20])
    triangles : str, optional
        'triangles'로 설정하면 사각형 대신 삼각형 사용
    **kwargs :
        PARTICLE에 전달할 추가 인수들
        
    Returns:
    --------
    p : Particle
        삼각화된 막대의 면과 꼭짓점
    """
    # 이산화 점의 수 추출
    if n is None:
        n = [15, 20, 20]
    
    assert len(n) == 3, "n must have exactly 3 elements [nphi, ntheta, nz]"
    
    # 각도들
    phi = np.linspace(0, 2 * np.pi, n[0])
    theta = np.linspace(0, 0.5 * np.pi, n[1])
    
    # 원통의 z-값들
    z = 0.5 * np.linspace(-1, 1, n[2]) * (height - diameter)
    
    # 위쪽 캡
    cap1 = trispheresegment(phi, theta, diameter, **kwargs)
    cap1 = cap1.shift([0, 0, 0.5 * (height - diameter)])
    
    # 아래쪽 캡 (위쪽 캡을 z축 기준으로 뒤집기)
    cap2 = cap1.flip(2)  # z축 (인덱스 2) 기준으로 뒤집기
    
    # 원통 이산화를 위한 격자
    verts, faces = fvgrid(phi, z, triangles)
    
    # 원통 좌표
    phi_cyl = verts[:, 0]
    z_cyl = verts[:, 1]
    
    # 원통 만들기
    x = 0.5 * diameter * np.cos(phi_cyl)
    y = 0.5 * diameter * np.sin(phi_cyl)
    
    # varargin에서 TRIANGLES 인수 제거 (이미 fvgrid에서 처리됨)
    cylinder_kwargs = kwargs.copy()
    
    # 원통 입자
    cyl = Particle(np.column_stack([x, y, z_cyl]), faces, **cylinder_kwargs)
    
    # 입자 조합
    p = cap1.vertcat(cap2, cyl)
    p = p.clean()
    
    return p