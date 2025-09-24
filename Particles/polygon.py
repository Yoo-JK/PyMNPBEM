# polygon.py - 2D 다각형 클래스
"""
Polygon class for 2D polygons for use with Mesh2d.
Converted from MATLAB MNPBEM @polygon class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Any, Dict
from scipy.interpolate import splprep, splev
from matplotlib.path import Path


class Polygon:
    """다각형 클래스 - Mesh2d와 함께 사용하기 위한 2D 다각형"""
    
    def __init__(self, *args, **kwargs):
        """
        Polygon 초기화
        
        Usage:
        ------
        polygon(n, **kwargs) : n개 꼭짓점으로 다각형 초기화
        polygon('n', n, **kwargs) : n개 꼭짓점으로 다각형 초기화  
        polygon('pos', pos, **kwargs) : 주어진 위치로 다각형 초기화
        
        Parameters:
        -----------
        **kwargs :
            dir : 다각형 방향 (시계방향 또는 반시계방향)
            sym : 대칭 키워드 [], 'x', 'y', 'xy'
            size : [size[0], size[1]]로 다각형 스케일링
        """
        self.pos, self.dir, self.sym = self._init(*args, **kwargs)
        
        if self.sym:
            self._apply_symmetry(self.sym)
    
    def _init(self, *args, **kwargs) -> Tuple[np.ndarray, int, Optional[str]]:
        """다각형 초기화"""
        pos = None
        n = None
        
        # 위치 인수 처리
        if len(args) > 0:
            if isinstance(args[0], str) and args[0] == 'n':
                n = args[1]
            elif isinstance(args[0], str) and args[0] == 'pos':
                pos = np.array(args[1])
            elif isinstance(args[0], (int, float)):
                n = int(args[0])
            elif isinstance(args[0], (list, np.ndarray)):
                pos = np.array(args[0])
        
        # 키워드 인수 처리
        if 'n' in kwargs:
            n = kwargs['n']
        if 'pos' in kwargs:
            pos = np.array(kwargs['pos'])
        
        direction = kwargs.get('dir', 1)
        sym = kwargs.get('sym', None)
        size = kwargs.get('size', None)
        
        # 위치 초기화
        if pos is None and n is not None:
            phi = np.arange(n) / n * 2 * np.pi + np.pi / n
            pos = np.column_stack([np.cos(phi), np.sin(phi)])
        elif pos is None:
            raise ValueError("Either 'n' or 'pos' must be provided")
        
        # 크기 조정
        if size is not None:
            size = np.array(size)
            if len(size) == 1:
                size = np.array([size[0], size[0]])
            
            x_range = pos[:, 0].max() - pos[:, 0].min()
            y_range = pos[:, 1].max() - pos[:, 1].min()
            
            if x_range > 0:
                pos[:, 0] = size[0] / x_range * pos[:, 0]
            if y_range > 0:
                pos[:, 1] = size[1] / y_range * pos[:, 1]
        
        return pos, direction, sym
    
    def __str__(self):
        """문자열 표현"""
        return f"Polygon:\n  pos: {self.pos.shape}\n  dir: {self.dir}\n  sym: {self.sym}"
    
    def close(self):
        """xy-대칭의 경우 다각형 닫기"""
        if not self.sym or self.sym != 'xy':
            return self
        
        # 원점이 이미 리스트의 일부가 아닌 경우 원점 (0,0) 추가
        self._sort()
        pos = self.pos
        
        origin_at_end = np.allclose(pos[-1], [0, 0], atol=1e-6)
        origin_at_start = np.allclose(pos[0], [0, 0], atol=1e-6)
        crosses_axes = (np.abs(np.prod(pos[0])) < 1e-6 and 
                       np.abs(np.prod(pos[-1])) < 1e-6)
        
        if not origin_at_end and crosses_axes and not origin_at_start:
            self.pos = np.vstack([pos, [0, 0]])
        
        return self
    
    def dist(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """점 위치들의 다각형으로부터의 거리 찾기"""
        pos = np.array(pos)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        
        dmin = np.full(pos.shape[0], 1e10)
        imin = np.zeros(pos.shape[0], dtype=int)
        
        # 다각형의 위치
        xa = self.pos[:, 0]
        xb = np.roll(self.pos[:, 0], -1)
        ya = self.pos[:, 1] 
        yb = np.roll(self.pos[:, 1], -1)
        
        # 각 점에 대해 반복
        for j in range(pos.shape[0]):
            x, y = pos[j]
            
            # 가장 가까운 선분 위의 점까지의 거리 계산
            dx = xb - xa
            dy = yb - ya
            
            # 매개변수 t (0 <= t <= 1)
            t = ((x - xa) * dx + (y - ya) * dy) / (dx**2 + dy**2 + 1e-10)
            t = np.clip(t, 0, 1)
            
            # 가장 가까운 점들
            closest_x = xa + t * dx
            closest_y = ya + t * dy
            
            # 거리들
            distances = np.sqrt((closest_x - x)**2 + (closest_y - y)**2)
            
            # 최소 거리와 인덱스
            min_idx = np.argmin(distances)
            dmin[j] = distances[min_idx]
            imin[j] = min_idx
        
        return dmin, imin
    
    def flip(self, axis: int = 0):
        """주어진 축을 따라 다각형 뒤집기"""
        self.pos[:, axis] = -self.pos[:, axis]
        return self
    
    def interp1(self, pos: np.ndarray) -> Tuple['Polygon', List[int]]:
        """보간을 이용하여 주어진 위치들을 통과하는 새 다각형 만들기"""
        pos = np.array(pos)
        
        # 다각형 위에 있는 점들 찾기
        d, inst = self.dist(pos)
        valid_mask = np.abs(d) < 1e-6
        
        ipos = np.where(valid_mask)[0]
        inst = inst[valid_mask]
        
        if len(ipos) == 0:
            return self, []
        
        # 가장 가까운 이웃까지의 거리
        distances = np.linalg.norm(self.pos[inst] - pos[ipos], axis=1)
        
        # 거리에 따라 정렬
        sort_idx = np.argsort(distances)
        ipos = ipos[sort_idx]
        inst = inst[sort_idx]
        
        # 가장 가까운 이웃의 인덱스에 따라 정렬
        sort_idx = np.argsort(inst)
        ipos = ipos[sort_idx]
        
        # 보간된 다각형
        new_polygon = Polygon(pos=pos[ipos])
        new_polygon.dir = self.dir
        new_polygon.sym = self.sym
        
        return new_polygon, ipos.tolist()
    
    def midpoints(self, key: str = 'add'):
        """홀수 다각형 위치를 위한 중점 추가"""
        if key == 'same':
            # 위치가 이미 중점을 포함하는 경우, 매끄럽게만 함
            pos = np.vstack([self.pos[::2], self.pos[0:1]])
        else:
            # 다각형 위치에 첫 번째 점 추가하여 닫기
            pos = np.vstack([self.pos, self.pos[0:1]])
        
        n = pos.shape[0] - 1
        
        # 다각형 조각의 길이
        lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        
        # 다각형의 호장 길이
        x = np.concatenate([[0], np.cumsum(lengths)])
        
        # 중점에서의 보간
        xi = 0.5 * (x[:-1] + x[1:])
        
        # 스플라인 보간으로 위치 보간
        if len(pos) > 3:  # 스플라인을 위해 최소 4개 점 필요
            tck_x, u = splprep([pos[:, 0]], s=0, per=True)
            tck_y, u = splprep([pos[:, 1]], s=0, per=True)
            
            # 매개변수 재매핑
            u_interp = xi / x[-1]
            
            pos_interp_x = splev(u_interp, tck_x[0])
            pos_interp_y = splev(u_interp, tck_y[0])
            pos_interp = np.column_stack([pos_interp_x, pos_interp_y])
        else:
            # 선형 보간
            pos_interp = np.zeros((len(xi), 2))
            for i, xi_val in enumerate(xi):
                idx = np.searchsorted(x[1:], xi_val)
                if idx < len(lengths):
                    t = (xi_val - x[idx]) / lengths[idx]
                    pos_interp[i] = pos[idx] + t * (pos[idx + 1] - pos[idx])
        
        # 새 위치 배열 할당
        new_pos = np.zeros((2 * n, 2))
        
        # 다각형 위치 설정
        new_pos[::2] = pos[:-1]      # 원래 점들 
        new_pos[1::2] = pos_interp   # 중점들
        
        self.pos = new_pos
        return self
    
    def norm_vectors(self) -> np.ndarray:
        """다각형 위치에서의 법선 벡터"""
        pos = self.pos
        
        # 단위 벡터 함수
        def unit_vector(v):
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 0 벡터 방지
            return v / norms
        
        # 법선 벡터
        vec = np.roll(pos, -1, axis=0) - pos
        nvec = np.column_stack([-vec[:, 1], vec[:, 0]])
        nvec = unit_vector(nvec)
        
        # 다각형 위치에 보간
        nvec = (nvec + np.roll(nvec, 1, axis=0)) / 2
        nvec = unit_vector(nvec)
        
        # 법선 벡터의 방향 확인
        posp = pos + 1e-6 * nvec
        
        # 점이 다각형 내부에 있는지 확인
        path = Path(pos)
        inside = path.contains_points(posp)
        
        if self.dir == 1:
            nvec[inside] = -nvec[inside]
        elif self.dir == -1:
            nvec[~inside] = -nvec[~inside]
        
        # 대칭점에서의 법선 벡터
        if self.sym:
            if self.sym in ['x', 'xy']:
                x_sym_mask = np.abs(pos[:, 0]) < 1e-10
                nvec[x_sym_mask, 0] = 0
            if self.sym in ['y', 'xy']:
                y_sym_mask = np.abs(pos[:, 1]) < 1e-10
                nvec[y_sym_mask, 1] = 0
            nvec = unit_vector(nvec)
        
        return nvec
    
    def plot(self, line_spec: str = 'b-', nvec: bool = False, scale: float = 1.0):
        """다각형 플롯"""
        pos = self.pos
        
        # 다각형 플롯 (닫기 위해 첫 번째 점 추가)
        closed_pos = np.vstack([pos, pos[0:1]])
        plt.plot(closed_pos[:, 0], closed_pos[:, 1], line_spec)
        
        # 법선 벡터 플롯
        if nvec:
            nvec_arr = self.norm_vectors()
            plt.quiver(pos[:, 0], pos[:, 1], 
                      nvec_arr[:, 0], nvec_arr[:, 1], 
                      scale=scale, scale_units='xy', angles='xy')
        
        plt.axis('equal')
        plt.grid(True)
    
    def polymesh2d(self, **kwargs):
        """mesh2d 함수를 다각형과 함께 호출 (간단화된 버전)"""
        # 실제 구현에서는 mesh2d 라이브러리가 필요
        print("Warning: polymesh2d requires external mesh2d library")
        
        # 기본 삼각화를 위한 간단한 구현
        from scipy.spatial import Delaunay
        
        pos, cnet = self.union()
        
        # Delaunay 삼각화
        tri = Delaunay(pos)
        
        return pos, tri.simplices
    
    def rot(self, angle: float):
        """주어진 각도로 다각형 회전"""
        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle)
        
        # 회전 행렬
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 위치 회전
        self.pos = self.pos @ rotation_matrix.T
        return self
    
    def round_edges(self, rad: Optional[float] = None, nrad: int = 5, 
                   edge: Optional[List[int]] = None):
        """다각형의 모서리 둥글게 하기"""
        pos = self.pos
        
        if rad is None:
            rad = 0.1 * np.max(np.abs(pos))
        
        if edge is None:
            edge = list(range(pos.shape[0]))
        
        # 간단화된 구현 - 실제로는 더 복잡한 기하학적 계산 필요
        print("Warning: Edge rounding is simplified")
        return self
    
    def scale(self, scale_factor: Union[float, List[float]]):
        """다각형 스케일링"""
        if isinstance(scale_factor, (int, float)):
            scale_factor = [scale_factor, scale_factor]
        
        scale_factor = np.array(scale_factor)
        self.pos = self.pos * scale_factor
        return self
    
    def shift(self, vec: np.ndarray):
        """주어진 벡터로 다각형 이동"""
        vec = np.array(vec)
        self.pos = self.pos + vec
        return self
    
    def shiftbnd(self, dist: float) -> Tuple['Polygon', np.ndarray]:
        """법선 방향으로 다각형 경계 이동"""
        # 법선 벡터 계산
        nvec = self.norm_vectors()
        
        # 변위 벡터
        displacement = np.sign(dist) * nvec
        
        # 간단화된 변위 (실제로는 교차점 계산 필요)
        actual_displacement = np.full(len(self.pos), abs(dist))
        
        # 꼭짓점 변위
        self.pos = self.pos + (actual_displacement.reshape(-1, 1) * displacement)
        
        return self, actual_displacement
    
    def _sort(self):
        """내부 사용을 위한 다각형의 정상 순서"""
        if not self.sym:
            return self
        
        # x 및/또는 y축에 있는 위치 찾기
        indices = []
        
        if self.sym in ['x', 'xy']:
            x_axis_mask = np.abs(self.pos[:, 0]) < 1e-6
            indices.extend(np.where(x_axis_mask)[0])
        
        if self.sym in ['y', 'xy']:
            y_axis_mask = np.abs(self.pos[:, 1]) < 1e-6
            indices.extend(np.where(y_axis_mask)[0])
        
        indices = np.unique(indices)
        
        # 첫 번째/마지막 점이 x/y축에 있도록 위치 이동
        if len(indices) >= 2 and indices[0] != 0:
            shift_amount = len(self.pos) - indices[-1]
            self.pos = np.roll(self.pos, shift_amount, axis=0)
        
        return self
    
    def _apply_symmetry(self, sym: str):
        """주어진 대칭에 대해 다각형 변환"""
        if not sym:
            return
        
        # 대칭을 위해 위치 축소
        pos = np.vstack([self.pos, self.pos[0:1]])  # 다각형 닫기
        
        # 위치 반올림
        pos[np.abs(pos[:, 0]) < 1e-8, 0] = 0
        pos[np.abs(pos[:, 1]) < 1e-8, 1] = 0
        
        # 대칭 영역 내부의 위치 찾기
        inside_mask = self._inside_symmetry_region(pos, sym)
        
        first = np.argmax(inside_mask)
        
        sympos = []
        
        if first == 0:
            sympos.append(pos[first])
        else:
            # 교차점 계산 (간단화)
            intersect_pos = self._intersect_symmetry(pos[first-1], pos[first], sym)
            sympos.append(intersect_pos)
        
        # 나머지 점들 처리
        for i in range(first + 1, len(pos)):
            if inside_mask[i-1] and inside_mask[i]:
                sympos.append(pos[i])
            elif inside_mask[i-1] != inside_mask[i]:
                intersect_pos = self._intersect_symmetry(pos[i-1], pos[i], sym)
                if len(sympos) == 0 or not np.allclose(intersect_pos, sympos[-1]):
                    sympos.append(intersect_pos)
                
                if sym == 'xy' and not inside_mask[i]:
                    sympos.append([0, 0])
                
                if inside_mask[i] and not np.allclose(intersect_pos, pos[i]):
                    sympos.append(pos[i])
        
        # 처음과 마지막 위치가 다른지 확인
        sympos = np.array(sympos)
        if len(sympos) > 1 and np.allclose(sympos[0], sympos[-1]):
            sympos = sympos[:-1]
        
        # 위치와 대칭 키워드 설정
        self.pos = sympos
        self.sym = sym
        
        # 다각형 정렬
        self._sort()
        
        # xy-대칭의 경우 원점 제거
        if sym == 'xy' and np.allclose(self.pos[-1], [0, 0]):
            if not np.any(np.all(self.pos[:-1] == 0, axis=1)):
                self.pos = self.pos[:-1]
    
    def _inside_symmetry_region(self, pos: np.ndarray, sym: str) -> np.ndarray:
        """대칭 영역 내부에 있는지 결정"""
        if sym == 'x':
            return pos[:, 0] >= 0
        elif sym == 'y':
            return pos[:, 1] >= 0
        elif sym == 'xy':
            return (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
        else:
            return np.ones(len(pos), dtype=bool)
    
    def _intersect_symmetry(self, posa: np.ndarray, posb: np.ndarray, sym: str) -> np.ndarray:
        """다각형 점들 사이의 연결 교차"""
        xa, ya = posa
        xb, yb = posb
        
        if sym == 'x':
            x = 0
            y = ya - xa * (yb - ya) / (xb - xa + 1e-10)
            return np.array([x, y])
        elif sym == 'y':
            x = xa - ya * (xb - xa) / (yb - ya + 1e-10)
            y = 0
            return np.array([x, y])
        elif sym == 'xy':
            if xa * xb <= 0:
                return self._intersect_symmetry(posa, posb, 'x')
            elif ya * yb <= 0:
                return self._intersect_symmetry(posa, posb, 'y')
        
        return posa  # 기본값
    
    def union(self) -> Tuple[np.ndarray, np.ndarray]:
        """mesh2d에서 사용하기 위해 다각형의 위치와 연결 결합"""
        pos = self.pos
        n = len(pos)
        
        # 연결 네트워크 (각 점을 다음 점과 연결)
        connections = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
        
        return pos, connections