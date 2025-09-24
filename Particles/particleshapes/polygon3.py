# polygon3.py - 3D 다각형 압출 클래스
"""
Polygon3 class for 3D polygon extrusion of particles.
Converted from MATLAB MNPBEM @polygon3 class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
from mpl_toolkits.mplot3d import Axes3D

from ..getbemoptions import getbemoptions
from ..particle import Particle
from ..polygon import Polygon
from .edgeprofile import EdgeProfile
from ..misc import round_array


class Polygon3:
    """3D 다각형 클래스 - 입자 압출을 위한 3D 다각형"""
    
    def __init__(self, poly: Polygon, z: float, **kwargs):
        """
        Polygon3 초기화
        
        Parameters:
        -----------
        poly : Polygon
            2D 다각형
        z : float
            다각형의 z-값
        **kwargs :
            edge : EdgeProfile, 모서리 프로파일
            refun : callable, 플레이트 이산화를 위한 세분화 함수
        """
        self.poly = poly
        self.z = z
        self.edge = None
        self._refun = None
        
        self._init(poly, z, **kwargs)
    
    def _init(self, poly: Polygon, z: float, **kwargs):
        """polygon3 객체 초기화"""
        # 옵션 추출
        op = getbemoptions(**kwargs)
        
        # 입력 추출
        if 'edge' in op:
            self.edge = op['edge']
        if 'refun' in op:
            self._refun = op['refun']
        
        # 모서리 프로파일의 기본값
        if self.edge is None:
            self.edge = EdgeProfile()
    
    def __str__(self):
        """문자열 표현"""
        return f"Polygon3:\n  poly: {self.poly}\n  z: {self.z}\n  edge: {self.edge}"
    
    @property
    def refun(self):
        """세분화 함수"""
        return self._refun
    
    @refun.setter
    def refun(self, value):
        """세분화 함수 설정"""
        self._refun = value
    
    def flip(self, axis: int = 0):
        """주어진 축을 따라 다각형 뒤집기"""
        self.poly = self.poly.flip(axis)
        return self
    
    def shift(self, vec: np.ndarray):
        """주어진 벡터로 다각형 이동"""
        vec = np.array(vec)
        
        # 2D 부분으로 다각형 이동
        self.poly = self.poly.shift(vec[:2])
        
        # z-값 이동
        if len(vec) > 2:
            self.z = self.z + vec[2]
        
        return self
    
    def shiftbnd(self, dist: float):
        """법선 방향으로 다각형 경계 이동"""
        self.poly = self.poly.shiftbnd(dist)[0]
        
        # 모서리 프로파일 지우기
        self.edge = EdgeProfile()
        
        return self
    
    def set_properties(self, **kwargs):
        """POLYGON3 객체의 속성 설정"""
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except:
                # POLYGON의 값 설정 시도
                try:
                    setattr(self.poly, key, value)
                except AttributeError:
                    print(f"Warning: Could not set property '{key}'")
        
        return self
    
    def plot(self, line_spec: str = 'b-'):
        """다각형 플롯"""
        pos = self.poly.pos
        
        # plot3 호출
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        z_values = np.full(pos.shape[0], self.z)
        ax.plot(pos[:, 0], pos[:, 1], z_values, line_spec)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
    
    def plate(self, **kwargs) -> Tuple[Particle, 'Polygon3']:
        """다각형들로부터 플레이트 만들기"""
        # 단일 객체인 경우를 위한 래퍼
        if not isinstance(self, list):
            obj_list = [self]
        else:
            obj_list = self
        
        # z-값들 가져오기
        z_values = [obj.z for obj in obj_list]
        
        # 모든 z-값이 동일한지 확인
        z_rounded = round_array(np.array(z_values), 8)
        z_unique = np.unique(z_rounded)
        assert len(z_unique) == 1, "All z-values must be identical"
        
        # 옵션 구조체
        op = getbemoptions(**kwargs)
        
        # 모서리 프로파일 오버라이드
        if 'edge' in op:
            for obj in obj_list:
                obj.edge = op['edge']
        
        # 다각형들의 세분화 함수
        fun_list = [obj.refun for obj in obj_list]
        
        # PLATE에 전달된 세분화 함수 추가
        if 'refun' in op:
            fun_list.append(op['refun'])
        
        # Mesh2d에 전달된 옵션들
        if 'hdata' not in op:
            op['hdata'] = {}
        if 'options' not in op:
            op['options'] = {'output': False}
        
        # 세분화 함수
        if any(f is not None for f in fun_list):
            op['hdata']['fun'] = lambda x, y, _: self._refun_wrapper(obj_list, x, y, fun_list)
        
        # 대칭성
        if 'sym' not in op:
            op['sym'] = None
        
        # 다각형들 가져오기
        poly_list = [obj.poly for obj in obj_list]
        
        # 다각형들 대칭화
        # 실제 구현에서는 symmetry와 close 함수가 필요
        poly1 = poly_list  # 간단화된 버전
        
        # 플레이트 삼각화 (간단화된 버전)
        print("Warning: Plate triangulation is simplified")
        
        # 기본 삼각화
        from scipy.spatial import Delaunay
        
        # 모든 다각형 위치 결합
        all_pos = np.vstack([poly.pos for poly in poly1])
        
        # Delaunay 삼각화
        tri = Delaunay(all_pos)
        
        # 3D 꼭짓점 생성
        verts_3d = np.column_stack([all_pos, np.full(len(all_pos), obj_list[0].z)])
        
        # 입자 만들기
        p = Particle(verts_3d, tri.simplices)
        
        # 중점 추가
        p = p.midpoints('flat')
        
        # 플레이트 법선 방향
        direction = op.get('dir', 1)
        
        # 법선 벡터가 올바른 방향을 가리키는지 확인
        if hasattr(p, 'nvec') and p.nvec is not None:
            if np.sign(np.sum(p.nvec[:, 2])) != direction:
                p = p.flipfaces()
        
        return p, obj_list[0]
    
    def _refun_wrapper(self, obj_list: List['Polygon3'], x: np.ndarray, y: np.ndarray, fun_list: List[Callable]) -> np.ndarray:
        """Mesh2d 도구상자를 위한 세분화 함수"""
        # 위치
        pos = np.column_stack([x.flatten(), y.flatten(), np.full(len(x.flatten()), obj_list[0].z)])
        
        # 거리 배열
        d_list = []
        for obj in obj_list:
            d, _ = obj.poly.dist(pos[:, :2])
            d_list.append(d)
        
        # 비어있지 않은 세분화 함수에 대한 인덱스
        valid_fun_indices = [i for i, fun in enumerate(fun_list[:len(obj_list)]) if fun is not None]
        
        # 이산화 매개변수
        if valid_fun_indices:
            h_values = []
            for i in valid_fun_indices:
                h_values.append(fun_list[i](pos, d_list[i]))
            h = np.min(h_values, axis=0)
        else:
            h = np.full(len(pos), np.inf)
        
        # PLATE에 전달된 세분화 함수
        if len(fun_list) > len(obj_list) and fun_list[-1] is not None:
            h_extra = fun_list[-1](pos, d_list)
            h = np.minimum(h, h_extra)
        
        return h
    
    def hribbon(self, d: np.ndarray, **kwargs) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        """수평 리본 만들기"""
        # 옵션 가져오기
        op = getbemoptions(**kwargs)
        
        # 방향과 대칭성의 기본값
        direction = op.get('dir', 1)
        sym = op.get('sym', None)
        
        # 리본 이산화
        p, inner, outer = self._hribbon_single(d, direction, sym, **kwargs)
        
        return p, inner, outer
    
    def _hribbon_single(self, d: np.ndarray, direction: int, sym: Optional[str], **kwargs) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        """단일 다각형을 위한 수평 리본"""
        # 부드러운 다각형
        poly = self.poly.midpoints()
        
        # 대칭성 처리
        if sym:
            # 실제 구현에서는 symmetry 함수 필요
            print("Warning: Symmetry handling is simplified")
        
        # 다각형의 위치와 법선 벡터
        pos = poly.pos
        vec = poly.norm_vectors()
        
        # 윤곽 닫기?
        if sym is None or not np.all(np.abs(np.prod(pos[[0, -1]], axis=1)) < 1e-6):
            pos = np.vstack([pos, pos[0:1]])
            vec = np.vstack([vec, vec[0:1]])
        
        # 중점을 위한 모서리의 z-값들 확장
        d_extended = np.zeros(2 * len(d))
        d_extended[::2] = d
        d_extended[1::2] = 0.5 * (d[:-1] + d[1:]) if len(d) > 1 else d[0]
        
        # 평면 리본 삼각화 (간단화된 버전)
        print("Warning: Ribbon triangulation is simplified")
        
        # 기본 격자 생성
        n_pos = len(pos)
        n_d = len(d_extended)
        
        # 면 꼭짓점
        x = np.zeros((n_pos, n_d))
        y = np.zeros((n_pos, n_d))
        
        # 변위에 대해 반복
        for i, d_val in enumerate(d_extended):
            # 경계 이동 (간단화된 버전)
            dist_vals = np.full(n_pos, d_val)
            
            # 리본 꼭짓점 변위
            x[:, i] = pos[:, 0] + dist_vals * vec[:, 0]
            y[:, i] = pos[:, 1] + dist_vals * vec[:, 1]
        
        # 꼭짓점 조립 (간단화)
        verts_2d = np.column_stack([x.flatten(), y.flatten()])
        verts_3d = np.column_stack([verts_2d, np.full(len(verts_2d), self.z)])
        
        # 기본 면 생성 (간단화)
        faces = []
        for i in range(n_pos - 1):
            for j in range(n_d - 1):
                idx = i * n_d + j
                faces.extend([
                    [idx, idx + 1, idx + n_d],
                    [idx + 1, idx + n_d + 1, idx + n_d]
                ])
        
        if faces:
            faces = np.array(faces)
        else:
            faces = np.array([]).reshape(0, 3)
        
        # 입자로 저장
        p = Particle(verts_3d, faces, **kwargs)
        
        # 법선 벡터가 올바른 방향을 가리키는지 확인
        if hasattr(p, 'nvec') and p.nvec is not None and len(p.nvec) > 0:
            if np.sign(np.sum(p.nvec[:, 2])) != direction:
                p = p.flipfaces()
        
        # 위/아래 리본 경계를 위한 다각형
        inner = Polygon3(self.poly.shiftbnd(np.min(d))[0], self.z)
        outer = Polygon3(self.poly.shiftbnd(np.max(d))[0], self.z)
        
        return p, inner, outer
    
    def vribbon(self, z: Optional[np.ndarray] = None, **kwargs) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        """수직 리본 압출"""
        # 옵션 가져오기
        op = getbemoptions(**kwargs)
        
        # 모서리 함수들
        if 'edge' in op:
            self.edge = op['edge']
        
        # 대칭성
        sym = op.get('sym', None)
        
        # 모서리 프로파일의 리본
        def shift_func(z_val):
            return self.edge.hshift(z_val)
        
        # 리본 이산화
        if z is not None:
            p, upper, lower = self._vribbon_single(z, shift_func, sym, **kwargs)
        else:
            p, upper, lower = self._vribbon_single(self.edge.z, shift_func, sym, **kwargs)
        
        return p, upper, lower
    
    def _vribbon_single(self, z: np.ndarray, shift_func: Callable, sym: Optional[str], **kwargs) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        """단일 다각형을 위한 수직 리본 압출"""
        # 부드러운 다각형
        poly = self.poly.midpoints()
        
        # 대칭성 처리
        if sym:
            print("Warning: Symmetry handling is simplified")
        
        # 다각형의 위치와 법선 벡터
        pos = poly.pos
        vec = poly.norm_vectors()
        
        # 윤곽 닫기?
        if sym is None or not np.all(np.abs(np.prod(pos[[0, -1]], axis=1)) < 1e-6):
            pos = np.vstack([pos, pos[0:1]])
            vec = np.vstack([vec, vec[0:1]])
        
        # 중점을 위한 모서리의 z-값들 확장
        z_extended = np.zeros(2 * len(z))
        z_extended[::2] = z
        z_extended[1::2] = 0.5 * (z[:-1] + z[1:]) if len(z) > 1 else z[0]
        
        # 평면 리본 삼각화 (간단화된 버전)
        n_pos = len(pos)
        n_z = len(z_extended)
        
        # 리본 위치 변환
        verts = []
        for i in range(n_pos):
            for j, z_val in enumerate(z_extended):
                verts.append([pos[i, 0], pos[i, 1], z_val])
        
        verts = np.array(verts)
        
        # 시프트 함수 적용
        shifts = shift_func(verts[:, 2])
        if np.isscalar(shifts):
            shifts = np.full(len(verts), shifts)
        
        # 꼭짓점 변위
        for i in range(n_pos):
            for j in range(n_z):
                idx = i * n_z + j
                pos_idx = i % len(vec)
                verts[idx, 0] += shifts[idx] * vec[pos_idx, 0]
                verts[idx, 1] += shifts[idx] * vec[pos_idx, 1]
        
        # 기본 면 생성
        faces = []
        for i in range(n_pos - 1):
            for j in range(n_z - 1):
                idx = i * n_z + j
                faces.extend([
                    [idx, idx + 1, idx + n_z],
                    [idx + 1, idx + n_z + 1, idx + n_z]
                ])
        
        if faces:
            faces = np.array(faces)
        else:
            faces = np.array([]).reshape(0, 3)
        
        # 입자로 저장
        p = Particle(verts, faces, **kwargs)
        
        # 첫 번째 다각형 점에 가장 가까운 점 찾기
        if len(p.pos) > 0:
            distances = ((p.pos[:, 0] - pos[0, 0])**2 + (p.pos[:, 1] - pos[0, 1])**2)
            ind = np.argmin(distances)
            
            # 법선 벡터가 올바른 방향을 가리키는지 확인
            if hasattr(p, 'nvec') and p.nvec is not None and len(p.nvec) > ind:
                vec_3d = np.array([vec[0, 0], vec[0, 1], 0])
                if np.dot(vec_3d, p.nvec[ind]) < 0:
                    p = p.flipfaces()
        
        # 위/아래 리본 경계를 위한 다각형
        max_z, min_z = np.max(z), np.min(z)
        max_shift = shift_func(max_z) if callable(shift_func) else 0
        min_shift = shift_func(min_z) if callable(shift_func) else 0
        
        upper = Polygon3(self.poly.shiftbnd(max_shift)[0], max_z)
        lower = Polygon3(self.poly.shiftbnd(min_shift)[0], min_z)
        
        return p, upper, lower