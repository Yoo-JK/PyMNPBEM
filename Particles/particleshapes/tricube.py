# tricube.py - 둥근 모서리를 가진 정육면체 입자
"""
TRICUBE - Cube particle with rounded edges.
Converted from MATLAB MNPBEM tricube.m
"""

import numpy as np
from typing import Union, Optional
from ..particle import Particle
from ..getbemoptions import getbemoptions
from .fvgrid import fvgrid


def tricube(n: int, length: Optional[Union[float, list]] = None, **kwargs) -> Particle:
    """
    둥근 모서리를 가진 정육면체 입자 생성
    
    Parameters:
    -----------
    n : int
        격자 크기
    length : float or array-like, optional
        정육면체 모서리의 길이 (기본값: 1)
        단일 값 또는 [x, y, z] 길이 배열
    **kwargs :
        e : float, 모서리의 둥글림 매개변수 (기본값: 0.25)
        
    Returns:
    --------
    p : Particle
        둥근 모서리를 가진 정육면체 입자
    """
    # 길이 처리
    if length is None:
        length = 1
    
    # 길이를 배열로 만들기
    if np.isscalar(length):
        length = np.array([length, length, length])
    else:
        length = np.array(length)
        if len(length) != 3:
            length = np.tile(length[0], 3)
    
    # 옵션 추출
    op = getbemoptions(**kwargs)
    
    # 둥글림 매개변수
    e = op.get('e', 0.25)
    
    # 정육면체의 한 면을 이산화
    x, y, faces = _square(n, e)
    
    # z-값
    z = 0.5 * np.ones_like(x)
    
    # 정육면체 면들 조합
    particles = [
        Particle(np.column_stack([x, y, z]), faces),      # 앞면
        Particle(np.column_stack([y, x, -z]), faces),     # 뒷면
        Particle(np.column_stack([y, z, x]), faces),      # 오른쪽면
        Particle(np.column_stack([x, -z, y]), faces),     # 왼쪽면
        Particle(np.column_stack([z, x, y]), faces),      # 위쪽면
        Particle(np.column_stack([-z, y, x]), faces)     # 아래쪽면
    ]
    
    # 입자들을 수직으로 연결
    p = particles[0]
    for particle in particles[1:]:
        p = p.vertcat(particle)
    
    # 중복 꼭짓점 제거
    p = p.clean()
    
    # 구면 좌표에서의 꼭짓점 위치
    x_pos = p.verts2[:, 0] if p.verts2 is not None else p.verts[:, 0]
    y_pos = p.verts2[:, 1] if p.verts2 is not None else p.verts[:, 1]
    z_pos = p.verts2[:, 2] if p.verts2 is not None else p.verts[:, 2]
    
    # 직교좌표를 구면좌표로 변환
    r = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
    phi = np.arctan2(y_pos, x_pos)
    theta = np.arccos(z_pos / (r + 1e-10))  # 0으로 나누기 방지
    
    # 부호 있는 사인과 코사인
    def signed_sin(x):
        return np.sign(np.sin(x)) * np.abs(np.sin(x))**e
    
    def signed_cos(x):
        return np.sign(np.cos(x)) * np.abs(np.cos(x))**e
    
    # 모서리 둥글림을 위한 초구면 사용
    x_new = 0.5 * signed_cos(theta) * signed_cos(phi)
    y_new = 0.5 * signed_cos(theta) * signed_sin(phi)
    z_new = 0.5 * signed_sin(theta)
    
    # 입자 객체 만들기
    new_verts = np.column_stack([x_new, y_new, z_new])
    faces_to_use = p.faces2 if p.faces2 is not None else p.faces
    
    p_new = Particle(new_verts, faces_to_use, **kwargs)
    
    # 스케일링
    p_new = p_new.scale(length)
    
    return p_new


def _square(n: int, e: float) -> tuple:
    """
    사각형 삼각화
    
    Parameters:
    -----------
    n : int
        격자 크기
    e : float
        둥글림 매개변수
        
    Returns:
    --------
    x, y : ndarray
        변환된 좌표
    faces : ndarray
        면 배열
    """
    # 격자 생성
    u = np.linspace(-0.5**e, 0.5**e, n)
    
    # 격자
    verts, faces = fvgrid(u, u)
    
    # 격자에 대한 간격
    x = np.sign(verts[:, 0]) * (np.abs(verts[:, 0]))**(1/e)
    y = np.sign(verts[:, 1]) * (np.abs(verts[:, 1]))**(1/e)
    
    return x, y, faces