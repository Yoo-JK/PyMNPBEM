# particle.py - 이산화된 입자 표면 클래스
"""
Particle class for discretized particle surfaces.
Converted from MATLAB MNPBEM @particle class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
from scipy.sparse import sparse, csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .quadface import quadface
from .getbemoptions import getbemoptions
from .misc import round_array
from .shape import TriShape, QuadShape


class Particle:
    """입자 클래스 - 이산화된 입자 표면의 면과 꼭짓점 관리"""
    
    def __init__(self, verts: Optional[np.ndarray] = None, faces: Optional[np.ndarray] = None, **kwargs):
        """
        Particle 초기화
        
        Parameters:
        -----------
        verts : array, shape (n_vertices, 3)
            경계 요소의 꼭짓점들
        faces : array, shape (n_faces, 3 or 4)
            경계 요소의 면들 (삼각형 또는 사각형)
        **kwargs :
            interp : 'flat' 또는 'curv' 입자 경계
            norm : 'off'로 설정 시 보조 정보 생성 안함
        """
        # 기본 속성 초기화
        self.verts = None
        self.faces = None
        self.pos = None          # 면의 중심점
        self.vec = None          # 중심점에서의 접선 및 법선 벡터
        self.area = None         # 면의 넓이
        self.quad = None         # 경계 요소 적분을 위한 구적법 규칙
        
        # 곡면 입자 경계를 위한 추가 속성
        self.verts2 = None       # 곡면 입자 경계를 위한 추가 꼭짓점
        self.faces2 = None       # 곡면 입자 경계를 위한 추가 면
        self.interp = 'flat'     # 'flat' 또는 'curv' 입자 경계
        
        # 초기화
        if verts is not None:
            self._init(verts, faces, **kwargs)
    
    def _init(self, verts: np.ndarray, faces: np.ndarray, **kwargs):
        """이산화된 입자 표면 초기화"""
        if verts is None or len(verts) == 0:
            return
        
        # 삼각형 요소만 있는 경우
        if faces.shape[1] == 3:
            self.verts = verts
            self.faces = np.column_stack([faces, np.full(faces.shape[0], np.nan)])
        # 삼각형 및/또는 사각형 요소
        elif faces.shape[1] == 4:
            self.verts = verts
            self.faces = faces
        # 곡면 입자 보간을 위한 중간점들
        else:
            # 곡면 입자 보간을 위한 전체 면
            self.verts2 = verts
            self.faces2 = faces
            
            # 꼭짓점 인덱스
            faces_flat = faces[:, :4].flatten()
            ind = ~np.isnan(faces_flat)
            
            # 고유 꼭짓점 인덱스
            unique_indices, inv_indices = np.unique(faces_flat[ind].astype(int), return_inverse=True)
            
            # 고유 꼭짓점
            self.verts = verts[unique_indices]
            
            # 고유 꼭짓점을 위한 면 리스트
            faces_new = np.full((faces.shape[0], 4), np.nan)
            faces_new.flat[ind] = inv_indices
            self.faces = faces_new
        
        # 면 적분
        self.quad = quadface(**kwargs)
        
        # 옵션 가져오기
        op = getbemoptions(['particle'], **kwargs)
        
        # 평면 또는 곡면 입자 경계
        if 'interp' in op:
            self.interp = op['interp']
        
        # 이산화된 입자 표면을 위한 보조 정보
        if 'norm' not in op or op['norm'] != 'off':
            self.norm(**kwargs)
    
    def __str__(self):
        """문자열 표현"""
        return f"Particle:\n  verts: {self.verts.shape if self.verts is not None else None}\n  faces: {self.faces.shape if self.faces is not None else None}\n  interp: {self.interp}"
    
    def index34(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """삼각형 및 사각형 경계 요소에 대한 인덱스"""
        if ind is None:
            ind3 = np.where(np.isnan(self.faces[:, 3]))[0]
            ind4 = np.where(~np.isnan(self.faces[:, 3]))[0]
        else:
            ind3 = np.where(np.isnan(self.faces[ind, 3]))[0]
            ind4 = np.where(~np.isnan(self.faces[ind, 3]))[0]
        
        return ind3, ind4
    
    def norm(self, **kwargs):
        """이산화된 입자 표면을 위한 보조 정보"""
        if self.interp == 'flat':
            self._norm_flat()
        elif self.interp == 'curv':
            self._norm_curv()
        
        return self
    
    def _norm_flat(self):
        """평면 경계 요소를 위한 보조 정보"""
        n = self.faces.shape[0]
        ind3, ind4 = self.index34()
        
        # 배열 할당
        self.pos = np.zeros((n, 3))
        
        # 삼각형 요소의 중심점 위치
        if len(ind3) > 0:
            self.pos[ind3] = (self.verts[self.faces[ind3, 0].astype(int)] + 
                             self.verts[self.faces[ind3, 1].astype(int)] + 
                             self.verts[self.faces[ind3, 2].astype(int)]) / 3
        
        # 사각형 요소의 중심점 위치  
        if len(ind4) > 0:
            self.pos[ind4] = (self.verts[self.faces[ind4, 0].astype(int)] + 
                             self.verts[self.faces[ind4, 1].astype(int)] + 
                             self.verts[self.faces[ind4, 2].astype(int)] + 
                             self.verts[self.faces[ind4, 3].astype(int)]) / 4
        
        # 면 요소를 삼각형으로 분할
        faces_tri, ind4_split = self.totriangles()
        
        # 꼭짓점
        v1 = self.verts[faces_tri[:, 0].astype(int)]
        v2 = self.verts[faces_tri[:, 1].astype(int)]
        v3 = self.verts[faces_tri[:, 2].astype(int)]
        
        # 삼각형 벡터
        vec1 = v1 - v2
        vec2 = v3 - v2
        
        # 법선 벡터
        nvec = np.cross(vec1, vec2)
        
        # 삼각형 요소의 넓이
        area = 0.5 * np.linalg.norm(nvec, axis=1)
        
        # 벡터 정규화
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1).reshape(-1, 1)
        nvec_norm = nvec / np.linalg.norm(nvec, axis=1).reshape(-1, 1)
        
        # 직교 기저 만들기
        vec2_norm = np.cross(nvec_norm, vec1_norm)
        
        if len(ind4_split) == 0:
            # 사각형 요소가 없는 경우
            self.area = area
            self.vec = [vec1_norm, vec2_norm, nvec_norm]
        else:
            # 넓이 누적
            self.area = np.bincount(np.arange(n).tolist() + ind4_split[:, 0].tolist(), 
                                  weights=area.tolist(), minlength=n)
            
            # 더 큰 넓이의 인덱스
            larger_area_mask = area[ind4_split[:, 0]] < area[ind4_split[:, 1]]
            
            # 더 큰 요소에서 벡터 선택
            for i, mask in enumerate(larger_area_mask):
                if mask:
                    idx = ind4_split[i, 0]
                    vec1_norm[idx] = vec1_norm[ind4_split[i, 1]]
                    vec2_norm[idx] = vec2_norm[ind4_split[i, 1]]
                    nvec_norm[idx] = nvec_norm[ind4_split[i, 1]]
            
            # 기저 저장
            self.vec = [vec1_norm[:n], vec2_norm[:n], nvec_norm[:n]]
    
    def _norm_curv(self):
        """곡면 경계 요소를 위한 보조 정보 (간단화된 버전)"""
        # 곡면 처리는 복잡하므로 기본 구현만 제공
        n = self.faces.shape[0]
        
        # 적분 가중치를 이용한 넓이 계산
        _, w = self.quad_curv()
        self.area = np.array([np.sum(w[w.indices == i]) for i in range(n)])
        
        # 기본 중심점과 벡터 (단순화된 계산)
        self.pos = np.zeros((n, 3))
        self.vec = [np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))]
        
        # 실제 곡면 계산은 더 복잡한 형상 함수가 필요
        print("Warning: Curved surface calculation is simplified")
    
    def totriangles(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """사각형 면 요소를 삼각형으로 분할"""
        if self.interp == 'flat':
            return self._totriangles_flat(ind)
        else:
            return self._totriangles_curv(ind)
    
    def _totriangles_flat(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """평면 사각형 면 요소를 삼각형으로 분할"""
        if ind is None:
            ind = np.arange(self.faces.shape[0])
        
        _, ind4 = self.index34(ind)
        
        # 면들
        faces = self.faces[ind, :3].astype(int)
        
        if len(ind4) > 0:
            # 사각형 요소 분할
            quad_faces = self.faces[ind[ind4]]
            additional_faces = np.column_stack([
                quad_faces[:, 2], quad_faces[:, 3], quad_faces[:, 0]
            ]).astype(int)
            faces = np.vstack([faces, additional_faces])
            
            # 인덱스 업데이트
            ind4_new = np.column_stack([
                ind4, 
                len(ind) + np.arange(len(ind4))
            ])
        else:
            ind4_new = np.array([]).reshape(0, 2)
        
        return faces, ind4_new
    
    def _totriangles_curv(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """곡면 사각형 면 요소를 삼각형으로 분할 (간단화된 버전)"""
        # 곡면의 경우 더 복잡한 처리 필요
        return self._totriangles_flat(ind)
    
    def quad_flat(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, csr_matrix, np.ndarray]:
        """평면 경계 요소 적분을 위한 구적점과 가중치"""
        if ind is None:
            ind = np.arange(self.faces.shape[0])
        
        # 삼각형으로 분해
        faces, ind4 = self.totriangles(ind)
        
        # 삼각형 인덱스
        ind3 = np.arange(len(ind))
        if len(ind4) > 0:
            ind3 = np.concatenate([ind3, ind4[:, 0]])
        
        # 삼각형 요소의 법선 벡터
        v1 = self.verts[faces[:, 0]]
        v2 = self.verts[faces[:, 1]] 
        v3 = self.verts[faces[:, 2]]
        
        nvec = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(nvec, axis=1)
        
        # 적분점과 가중치
        x = self.quad.x.flatten()
        y = self.quad.y.flatten()
        w = self.quad.w.flatten()
        
        m = len(w)
        n = m * len(ind3)
        
        # 적분점과 가중치 배열 할당
        pos = np.zeros((n, 3))
        weight = np.zeros(n)
        row = np.zeros(n, dtype=int)
        col = np.arange(n)
        
        # 삼각형 형상 요소
        tri = np.column_stack([x, y, 1 - x - y])
        
        # 삼각형 요소에 대해 반복
        offset = 0
        for i, face_idx in enumerate(ind3):
            it = slice(offset, offset + m)
            
            # 적분점 보간
            vertices = self.verts[faces[i, :3]]
            pos[it] = tri @ vertices
            
            # 적분 가중치
            weight[it] = w * area[i]
            row[it] = face_idx
            
            offset += m
        
        # 가중치 행렬
        weight_matrix = csr_matrix((weight, (row, col)))
        
        # 적분점에 대한 면 인덱스
        iface = row
        
        return pos, weight_matrix, iface
    
    def quad_curv(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, csr_matrix, np.ndarray]:
        """곡면 경계 요소 적분을 위한 구적점과 가중치 (간단화된 버전)"""
        # 곡면의 경우 더 복잡한 형상 함수 필요
        return self.quad_flat(ind)
    
    def quad(self, ind: Optional[np.ndarray] = None) -> Tuple[np.ndarray, csr_matrix, np.ndarray]:
        """경계 요소 적분을 위한 구적점과 가중치"""
        if self.interp == 'flat':
            return self.quad_flat(ind)
        else:
            return self.quad_curv(ind)
    
    def clean(self, cutoff: float = 1e-10):
        """중복 꼭짓점과 너무 작은 넓이의 요소 제거"""
        # 반올림 오류 방지
        verts = round_array(self.verts, 8)
        
        # 고유 꼭짓점
        verts_unique, _, ind = np.unique(verts, axis=0, return_inverse=True)
        
        # 중복 꼭짓점 제거
        if verts_unique.shape[0] != self.verts.shape[0]:
            faces = self.faces.copy()
            ind3, ind4 = self.index34()
            
            # 삼각형 및 사각형 경계 요소 업데이트
            if len(ind3) > 0:
                faces[ind3, :3] = ind[faces[ind3, :3].astype(int)]
            if len(ind4) > 0:
                faces[ind4, :4] = ind[faces[ind4, :4].astype(int)]
            
            # 객체 업데이트
            self.verts = verts_unique
            self.faces = faces
        
        # 중복 꼭짓점을 가진 사각형 제거
        _, ind4 = self.index34()
        for i in ind4:
            face_vertices = self.faces[i, :4]
            face_vertices = face_vertices[~np.isnan(face_vertices)].astype(int)
            unique_verts, order = np.unique(face_vertices, return_index=True)
            
            # 중복 꼭짓점을 가진 요소 제거
            if len(unique_verts) == 3:
                new_face = np.full(4, np.nan)
                new_face[:3] = face_vertices[np.sort(order)]
                self.faces[i] = new_face
        
        # 충분히 큰 넓이를 가진 요소들만 유지
        if self.area is not None:
            valid_indices = np.where(self.area > cutoff * np.mean(self.area))[0]
            return self.select('index', valid_indices)
        
        return self.norm()
    
    def select(self, method: str, *args) -> 'Particle':
        """이산화된 입자 표면의 일부 선택"""
        x = self.pos[:, 0] if self.pos is not None else np.zeros(self.faces.shape[0])
        y = self.pos[:, 1] if self.pos is not None else np.zeros(self.faces.shape[0])
        z = self.pos[:, 2] if self.pos is not None else np.zeros(self.faces.shape[0])
        
        # 면 선택
        if method in ['ind', 'index']:
            indices = args[0]
        elif method in ['carfun', 'cartfun']:
            indices = np.where(args[0](x, y, z))[0]
        elif method == 'polfun':
            phi, r = np.arctan2(y, x), np.sqrt(x**2 + y**2)
            indices = np.where(args[0](phi, r, z))[0]
        elif method == 'sphfun':
            r = np.sqrt(x**2 + y**2 + z**2)
            phi = np.arctan2(y, x)
            theta = np.arccos(z / (r + 1e-10))
            indices = np.where(args[0](phi, np.pi/2 - theta, r))[0]
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # 선택된 면을 가진 입자
        return self._compress(indices)
    
    def _compress(self, indices: np.ndarray) -> 'Particle':
        """입자 압축 및 사용되지 않는 꼭짓점 제거"""
        faces = self.faces[indices]
        
        # 고유 꼭짓점 찾기
        face_vertices = faces.flatten()
        unique_vertices = np.unique(face_vertices[~np.isnan(face_vertices)]).astype(int)
        
        # 고유 꼭짓점 테이블 만들기
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        
        # 면 인덱스 업데이트
        new_faces = faces.copy()
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                if not np.isnan(faces[i, j]):
                    new_faces[i, j] = vertex_map[int(faces[i, j])]
        
        # 새 입자 생성
        new_particle = Particle()
        new_particle.verts = self.verts[unique_vertices]
        new_particle.faces = new_faces
        new_particle.interp = self.interp
        
        # 곡면 경계를 위한 추가 처리
        if self.verts2 is not None and self.faces2 is not None:
            faces2 = self.faces2[indices]
            face_vertices2 = faces2.flatten()
            unique_vertices2 = np.unique(face_vertices2[~np.isnan(face_vertices2)]).astype(int)
            
            vertex_map2 = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices2)}
            
            new_faces2 = faces2.copy()
            for i in range(faces2.shape[0]):
                for j in range(faces2.shape[1]):
                    if not np.isnan(faces2[i, j]):
                        new_faces2[i, j] = vertex_map2[int(faces2[i, j])]
            
            new_particle.verts2 = self.verts2[unique_vertices2]
            new_particle.faces2 = new_faces2
        
        # 보조 정보
        return new_particle.norm()
    
    def flip(self, direction: int = 1, **kwargs):
        """주어진 방향을 따라 이산화된 입자 표면 뒤집기"""
        # 꼭짓점 뒤집기
        self.verts[:, direction-1] = -self.verts[:, direction-1]
        if self.verts2 is not None:
            self.verts2[:, direction-1] = -self.verts2[:, direction-1]
        
        # 면 뒤집기
        return self.flipfaces(**kwargs)
    
    def flipfaces(self, **kwargs):
        """표면 요소의 방향 뒤집기"""
        ind3, ind4 = self.index34()
        
        # 면 뒤집기
        if len(ind3) > 0:
            self.faces[ind3, :3] = np.fliplr(self.faces[ind3, :3])
        if len(ind4) > 0:
            self.faces[ind4, :4] = np.fliplr(self.faces[ind4, :4])
        
        if self.verts2 is not None:
            if len(ind3) > 0:
                # 삼각형의 곡면 면 뒤집기
                old_order = self.faces2[ind3][:, [0, 1, 2, 4, 5, 6]]
                new_order = self.faces2[ind3][:, [2, 1, 0, 5, 4, 6]]
                self.faces2[ind3][:, [0, 1, 2, 4, 5, 6]] = new_order
            
            if len(ind4) > 0:
                # 사각형의 곡면 면 뒤집기
                old_order = self.faces2[ind4][:, [0, 1, 2, 3, 4, 5, 6, 7]]
                new_order = self.faces2[ind4][:, [3, 2, 1, 0, 6, 5, 4, 7]]
                self.faces2[ind4][:, [0, 1, 2, 3, 4, 5, 6, 7]] = new_order
        
        # 보조 정보
        return self.norm(**kwargs)
    
    def shift(self, vec: np.ndarray, **kwargs):
        """이산화된 입자 표면 이동"""
        self.verts = self.verts + vec
        if self.verts2 is not None:
            self.verts2 = self.verts2 + vec
        
        # 이산화된 입자 경계를 위한 보조 정보
        return self.norm(**kwargs)
    
    def scale(self, scale_factor: Union[float, np.ndarray], **kwargs):
        """이산화된 입자 표면의 좌표 스케일링"""
        self.verts = self.verts * scale_factor
        if self.verts2 is not None:
            self.verts2 = self.verts2 * scale_factor
        
        # 보조 정보
        return self.norm(**kwargs)
    
    def rot(self, angle: float, axis: Optional[np.ndarray] = None, **kwargs):
        """이산화된 입자 표면 회전"""
        if axis is None:
            axis = np.array([0, 0, 1])
        
        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle)
        
        # 축을 단위 벡터로 만들기
        axis = axis / np.linalg.norm(axis)
        
        # 로드리게스 회전 공식
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 회전 행렬
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # 꼭짓점 회전
        self.verts = self.verts @ R.T
        if self.verts2 is not None:
            self.verts2 = self.verts2 @ R.T
        
        # 보조 정보
        return self.norm(**kwargs)
    
    def flat(self):
        """평면 입자 경계 만들기"""
        self.interp = 'flat'
        return self.norm()
    
    def curved(self, **kwargs):
        """곡면 입자 경계 만들기"""
        if self.verts2 is None or kwargs:
            self.midpoints(**kwargs)
        
        self.interp = 'curv'
        return self.norm()
    
    def midpoints(self, key: str = 'flat'):
        """곡면 입자 경계를 위한 중점 추가 (간단화된 버전)"""
        self.interp = key
        
        if key == 'flat':
            # 평면 경계 요소를 위한 중점 추가
            edges, edge_faces = self.edges()
            n = self.verts.shape[0]
            
            # 꼭짓점 리스트에 중점 추가
            midpoints = 0.5 * (self.verts[edges[:, 0]] + self.verts[edges[:, 1]])
            self.verts2 = np.vstack([self.verts, midpoints])
            
            # 삼각형과 사각형 인덱스
            ind3, ind4 = self.index34()
            
            # 배열 할당
            self.faces2 = np.column_stack([self.faces, np.full((self.faces.shape[0], 5), np.nan)])
            
            # 삼각형을 위한 면 리스트 확장
            if len(ind3) > 0:
                self.faces2[ind3, 4:7] = n + edge_faces[ind3, :3]
            
            # 사각형을 위한 면 리스트 확장
            if len(ind4) > 0:
                self.faces2[ind4, 4:8] = n + edge_faces[ind4, :4]
                
                # 면 리스트에 중심점 추가
                centroids_start = self.verts2.shape[0]
                self.faces2[ind4, 8] = centroids_start + np.arange(len(ind4))
                
                # 꼭짓점 리스트에 중심점 추가
                centroids = 0.25 * (self.verts[self.faces[ind4, 0].astype(int)] + 
                                   self.verts[self.faces[ind4, 1].astype(int)] + 
                                   self.verts[self.faces[ind4, 2].astype(int)] + 
                                   self.verts[self.faces[ind4, 3].astype(int)])
                self.verts2 = np.vstack([self.verts2, centroids])
        
        elif key == 'curv':
            # 곡률을 이용한 중점 추가는 복잡하므로 간단화
            print("Warning: Curved midpoint calculation is simplified")
            self.midpoints('flat')  # 일단 평면 방식으로 대체
        
        # 이산화된 입자 표면을 위한 보조 정보
        if self.interp == 'curv':
            self.norm()
    
    def edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """입자의 고유한 모서리 찾기"""
        faces = self.faces
        ind3, ind4 = self.index34()
        
        # 모서리 리스트
        edge_list = []
        
        # 삼각형 모서리
        if len(ind3) > 0:
            tri_edges = np.array([
                [faces[ind3, 0], faces[ind3, 1]],
                [faces[ind3, 1], faces[ind3, 2]], 
                [faces[ind3, 2], faces[ind3, 0]]
            ]).reshape(-1, 2)
            edge_list.append(tri_edges)
        
        # 사각형 모서리
        if len(ind4) > 0:
            quad_edges = np.array([
                [faces[ind4, 0], faces[ind4, 1]],
                [faces[ind4, 1], faces[ind4, 2]],
                [faces[ind4, 2], faces[ind4, 3]], 
                [faces[ind4, 3], faces[ind4, 0]]
            ]).reshape(-1, 2)
            edge_list.append(quad_edges)
        
        if edge_list:
            net = np.vstack(edge_list).astype(int)
        else:
            net = np.array([]).reshape(0, 2)
        
        # 고유한 모서리
        net_sorted = np.sort(net, axis=1)
        unique_edges, inverse_indices = np.unique(net_sorted, axis=0, return_inverse=True)
        
        # 면 리스트의 모서리
        edge_faces = np.full((faces.shape[0], 4), -1)
        
        # 삼각형 면의 모서리 인덱스
        if len(ind3) > 0:
            edge_faces[ind3, :3] = inverse_indices[:3*len(ind3)].reshape(-1, 3)
        
        # 사각형 면의 모서리 인덱스  
        if len(ind4) > 0:
            start_idx = 3 * len(ind3)
            edge_faces[ind4, :4] = inverse_indices[start_idx:start_idx+4*len(ind4)].reshape(-1, 4)
        
        return unique_edges, edge_faces
    
    def border(self) -> np.ndarray:
        """입자의 경계 (단일 모서리) 찾기"""
        faces = self.faces
        ind3, ind4 = self.index34()
        
        # 모서리 리스트
        edge_list = []
        
        # 삼각형 모서리
        if len(ind3) > 0:
            for i in ind3:
                edges = [[faces[i, 0], faces[i, 1]], 
                        [faces[i, 1], faces[i, 2]], 
                        [faces[i, 2], faces[i, 0]]]
                edge_list.extend(edges)
        
        # 사각형 모서리
        if len(ind4) > 0:
            for i in ind4:
                edges = [[faces[i, 0], faces[i, 1]], 
                        [faces[i, 1], faces[i, 2]], 
                        [faces[i, 2], faces[i, 3]], 
                        [faces[i, 3], faces[i, 0]]]
                edge_list.extend(edges)
        
        if not edge_list:
            return np.array([]).reshape(0, 2)
        
        net = np.array(edge_list).astype(int)
        
        # 고유한 모서리
        net_sorted = np.sort(net, axis=1)
        unique_edges, inverse_indices = np.unique(net_sorted, axis=0, return_inverse=True)
        
        # 단일 모서리 결정 (한 번만 나타나는 모서리)
        edge_counts = np.bincount(inverse_indices)
        single_edge_mask = edge_counts == 1
        single_edge_indices = np.where(single_edge_mask)[0]
        
        return net[np.isin(inverse_indices, single_edge_indices)]
    
    def vertices(self, ind: int, close: bool = False) -> np.ndarray:
        """인덱싱된 면의 꼭짓점"""
        face_indices = self.faces[ind]
        face_indices = face_indices[~np.isnan(face_indices)].astype(int)
        
        if close:
            face_indices = np.append(face_indices, face_indices[0])
        
        return self.verts[face_indices]
    
    def interp_values(self, v: np.ndarray, key: str = 'area') -> Tuple[np.ndarray, csr_matrix]:
        """면에서 꼭짓점으로 (또는 그 반대로) 값 보간"""
        ind3, ind4 = self.index34()
        n_faces, n_verts = self.faces.shape[0], self.verts.shape[0]
        
        # 삼각형과 사각형을 위한 면 요소
        faces3 = self.faces[ind3, :3].astype(int)
        faces4 = self.faces[ind4, :4].astype(int)
        
        # v의 주요 차원
        n = v.shape[0]
        
        # 보간 행렬
        if n == n_faces:
            # 면에서 꼭짓점으로 보간
            if key == 'area':
                # 경계 요소의 넓이로 가중된 꼭짓점에서의 평균
                if self.area is not None:
                    row_ind = np.concatenate([faces3.flatten(), faces4.flatten()])
                    col_ind = np.concatenate([
                        np.repeat(ind3, 3), 
                        np.repeat(ind4, 4)
                    ])
                    data = np.concatenate([
                        np.repeat(self.area[ind3], 3),
                        np.repeat(self.area[ind4], 4)
                    ])
                    mat = csr_matrix((data, (row_ind, col_ind)), shape=(n_verts, n_faces))
                    
                    # 변환 행렬 정규화
                    row_sums = np.array(mat.sum(axis=1)).flatten()
                    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
                    mat = csr_matrix((mat.data / row_sums[mat.indices], mat.indices, mat.indptr))
                else:
                    # 넓이 정보가 없는 경우 균등 가중치
                    row_ind = np.concatenate([faces3.flatten(), faces4.flatten()])
                    col_ind = np.concatenate([
                        np.repeat(ind3, 3), 
                        np.repeat(ind4, 4)
                    ])
                    data = np.concatenate([
                        np.full(len(faces3.flatten()), 1/3),
                        np.full(len(faces4.flatten()), 1/4)
                    ])
                    mat = csr_matrix((data, (row_ind, col_ind)), shape=(n_verts, n_faces))
            else:
                # 의사역행렬 방법
                row_ind = np.concatenate([
                    np.repeat(ind3, 3), 
                    np.repeat(ind4, 4)
                ])
                col_ind = np.concatenate([faces3.flatten(), faces4.flatten()])
                data = np.concatenate([
                    np.full(len(faces3.flatten()), 1/3),
                    np.full(len(faces4.flatten()), 1/4)
                ])
                con = csr_matrix((data, (row_ind, col_ind)), shape=(n_faces, n_verts))
                mat = np.linalg.pinv(con.toarray())
                mat = csr_matrix(mat)
        else:
            # 꼭짓점에서 면으로 보간
            row_ind = np.concatenate([
                np.repeat(ind3, 3), 
                np.repeat(ind4, 4)
            ])
            col_ind = np.concatenate([faces3.flatten(), faces4.flatten()])
            data = np.concatenate([
                np.full(len(faces3.flatten()), 1/3),
                np.full(len(faces4.flatten()), 1/4)
            ])
            mat = csr_matrix((data, (row_ind, col_ind)), shape=(n_faces, n_verts))
        
        # 값 보간
        if v.ndim == 1:
            v_interp = mat @ v
        else:
            v_interp = mat @ v
        
        return v_interp, mat
    
    def plot(self, val: Optional[np.ndarray] = None, **kwargs):
        """이산화된 입자 표면이나 표면에 주어진 값들을 플롯"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 삼각형과 사각형 분리
        ind3, ind4 = self.index34()
        
        # 값이 주어진 경우 색상으로 표시
        if val is not None:
            if len(val) == len(self.faces):
                # 면 값을 꼭짓점 값으로 보간
                val_verts, _ = self.interp_values(val)
            else:
                val_verts = val
            
            # 삼각형 플롯
            if len(ind3) > 0:
                for i in ind3:
                    face_verts = self.faces[i, :3].astype(int)
                    triangle = self.verts[face_verts]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                  color=plt.cm.viridis(val_verts[face_verts].mean()))
            
            # 사각형 플롯 (두 개의 삼각형으로 분할)
            if len(ind4) > 0:
                for i in ind4:
                    face_verts = self.faces[i, :4].astype(int)
                    quad = self.verts[face_verts]
                    # 첫 번째 삼각형
                    ax.plot_trisurf(quad[[0,1,2], 0], quad[[0,1,2], 1], quad[[0,1,2], 2],
                                  color=plt.cm.viridis(val_verts[face_verts[:3]].mean()))
                    # 두 번째 삼각형
                    ax.plot_trisurf(quad[[2,3,0], 0], quad[[2,3,0], 1], quad[[2,3,0], 2],
                                  color=plt.cm.viridis(val_verts[face_verts[[2,3,0]]].mean()))
        else:
            # 값이 없는 경우 기본 색상으로 표시
            if len(ind3) > 0:
                triangles = self.faces[ind3, :3].astype(int)
                ax.plot_trisurf(self.verts[:, 0], self.verts[:, 1], self.verts[:, 2], 
                              triangles=triangles, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        
        # 동일한 스케일 설정
        max_range = np.array([self.verts[:,0].max()-self.verts[:,0].min(),
                             self.verts[:,1].max()-self.verts[:,1].min(),
                             self.verts[:,2].max()-self.verts[:,2].min()]).max() / 2.0
        mid_x = (self.verts[:,0].max()+self.verts[:,0].min()) * 0.5
        mid_y = (self.verts[:,1].max()+self.verts[:,1].min()) * 0.5
        mid_z = (self.verts[:,2].max()+self.verts[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    
    def __add__(self, other):
        """입자들을 수직으로 연결"""
        return self.vertcat(other)
    
    def vertcat(self, *others):
        """입자들을 수직으로 연결"""
        new_particle = Particle()
        new_particle.verts = self.verts.copy()
        new_particle.faces = self.faces.copy()
        new_particle.interp = self.interp
        
        if self.verts2 is not None:
            new_particle.verts2 = self.verts2.copy()
            new_particle.faces2 = self.faces2.copy()
        
        for other in others:
            # 면-꼭짓점 리스트 확장
            n_verts = new_particle.verts.shape[0]
            new_particle.faces = np.vstack([
                new_particle.faces, 
                other.faces + n_verts
            ])
            new_particle.verts = np.vstack([new_particle.verts, other.verts])
            
            # 곡면 입자 경계를 위한 면-꼭짓점 리스트 확장
            if new_particle.verts2 is not None and other.verts2 is not None:
                n_verts2 = new_particle.verts2.shape[0]
                new_particle.faces2 = np.vstack([
                    new_particle.faces2, 
                    other.faces2 + n_verts2
                ])
                new_particle.verts2 = np.vstack([new_particle.verts2, other.verts2])
        
        # 입자에 대한 보조 정보
        return new_particle.norm()
    
    @property
    def nvec(self) -> np.ndarray:
        """표면 요소의 법선 벡터"""
        return self.vec[2] if self.vec else None
    
    @property
    def tvec(self) -> List[np.ndarray]:
        """표면 요소의 접선 벡터"""
        return self.vec[:2] if self.vec else None
    
    @property
    def tvec1(self) -> np.ndarray:
        """첫 번째 접선 벡터"""
        return self.vec[0] if self.vec else None
    
    @property
    def tvec2(self) -> np.ndarray:
        """두 번째 접선 벡터"""
        return self.vec[1] if self.vec else None
    
    @property
    def nfaces(self) -> int:
        """표면 요소의 수"""
        return self.faces.shape[0] if self.faces is not None else 0
    
    @property
    def nverts(self) -> int:
        """꼭짓점의 수"""
        return self.verts.shape[0] if self.verts is not None else 0
    
    @property
    def size(self) -> int:
        """표면 요소의 수 (nfaces와 동일)"""
        return self.nfaces
    
    @property
    def n(self) -> int:
        """표면 요소의 수 (nfaces와 동일)"""
        return self.nfaces