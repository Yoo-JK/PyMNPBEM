# compoint.py - 점들의 복합체 클래스
"""
ComPoint class for compound of points in dielectric environment.
Converted from MATLAB MNPBEM @compoint class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
from .compound import Compound
from .point import Point
from .getbemoptions import getbemoptions
from .distmin3 import distmin3
from .indlayer import indlayer


class ComPoint(Compound):
    """점들의 복합체 클래스 - 유전체 환경에서 점들을 그룹핑하여 COMPGREEN 클래스에서 사용 가능하게 함"""
    
    def __init__(self, p, pos: np.ndarray, **kwargs):
        """
        ComPoint 초기화
        
        Parameters:
        -----------
        p : ComParticle
            comparticle 객체
        pos : array
            점들의 위치
        **kwargs : 
            mindist : 주어진 입자로부터 점들의 최소 거리 (예: [1, 0, ...])
            medium : 선택된 매질의 점들을 마스킹
            layer : 레이어 구조
        """
        # Compound 초기화 (빈 입자 리스트로)
        super().__init__(p.eps, [], [])
        
        # 초기화
        self._init_compoint(p, pos, **kwargs)
    
    def _init_compoint(self, p, pos1: np.ndarray, **kwargs):
        """ComPoint 초기화"""
        # P가 COMPARTICLE인지 확인
        assert hasattr(p, 'eps'), "p must be a comparticle object"
        
        # P와 전체 위치 수 저장
        self.pin = p
        self.npos = pos1.shape[0]
        
        # 입력 추출
        op = getbemoptions(**kwargs)
        
        # 최근접 이웃까지의 기본 최소 거리값
        if 'mindist' not in op:
            mindist = np.zeros(p.np)
        else:
            mindist = np.array(op['mindist'])
        
        # mindist가 단일 값인 경우 처리
        if len(mindist) != p.np:
            mindist = np.full(p.np, mindist[0] if len(mindist) == 1 else 0)
        
        # 입자와 위치들 사이의 최소 거리
        r, ind2 = distmin3(p, pos1, np.max(mindist))
        
        # 면 인덱스와 입자 번호 사이의 변환 테이블
        ind2part = np.zeros(p.n, dtype=int)
        for ip in range(p.np):
            ind2part[p.index(ip)] = ip
        
        # 경계에서 충분히 멀리 떨어진 위치들만 유지
        ind1 = np.where(np.abs(r) >= mindist[ind2part[ind2]])[0]
        r = r[ind1]
        
        # 점이 가장 가까운 표면의 내부 또는 외부에 있는지 결정
        inout = np.zeros(len(ind1), dtype=int)
        
        # sig에 따라 pos1이 가장 가까운 표면의 내부 또는 외부에 있음
        mask_inside = r < 0
        mask_outside = r >= 0
        
        inout[mask_inside] = p.inout[ind2part[ind2[ind1[mask_inside]]], 0]
        inout[mask_outside] = p.inout[ind2part[ind2[ind1[mask_outside]]], 1]
        
        # 레이어 구조 처리
        if 'layer' in op:
            # 레이어 구조 저장
            self.layer = op['layer']
            
            # 레이어 구조에 연결된 점들의 인덱스
            layer_indices = np.array(self.layer['ind']).reshape(1, -1)
            indl = np.any(inout.reshape(-1, 1) == layer_indices, axis=1)
            
            # 위치들의 인덱스
            indl1 = ind1[indl]
            
            # 레이어의 점들
            in_layer = self.layer['mindist'](pos1[indl1, 2]) < 1e-10
            
            # 점들을 상위 레이어로 이동
            pos1[indl1[in_layer], 2] += 1e-8
            
            # INOUT에 기판 인덱스 할당
            inout[indl] = self.layer['ind'][indlayer(self.layer, pos1[indl1, 2])]
        
        # 점들을 함께 그룹화
        iotab = np.unique(inout)
        
        # 인덱스와 점 객체들 초기화
        self.ind = []
        self.p = []
        self.inout = []
        
        # 다른 매질들에 대해 반복
        for i, io_val in enumerate(iotab):
            # 주어진 매질에서 점들의 집합에 대한 포인터
            self.ind.append(ind1[inout == io_val])
            
            # 주어진 매질에서 점들의 집합
            self.p.append(Point(pos1[self.ind[i], :]))
            
            # 유전체 함수에 대한 포인터
            self.inout.append([io_val])
        
        # 점들 마스킹
        if 'medium' not in op:
            self.mask = list(range(len(self.inout)))
        else:
            self.mask = []
            for i, io_val in enumerate(iotab):
                if io_val in op['medium']:
                    self.mask.append(i)
        
        # 점들의 복합체
        self.pc = self._vertcat_points([self.p[i] for i in self.mask])
    
    def _vertcat_points(self, points: List[Point]) -> Point:
        """점들을 수직으로 연결"""
        if not points:
            return None
        if len(points) == 1:
            return points[0]
        
        # 모든 점들의 위치를 연결
        all_pos = np.vstack([p.pos for p in points if hasattr(p, 'pos')])
        return Point(all_pos)
    
    def __str__(self):
        """문자열 표현"""
        return f"ComPoint:\n  eps: {len(self.eps)} materials\n  p: {len(self.p)} point groups\n  mask: {self.mask}"
    
    def flip(self, direction: Union[int, List[int]] = 1):
        """주어진 방향들을 따라 compoint 객체 뒤집기"""
        if isinstance(direction, int):
            direction = [direction]
        
        for dir_id in direction:
            for ip in range(len(self.p)):
                if hasattr(self.p[ip], 'pos') and self.p[ip].pos.shape[1] > dir_id:
                    self.p[ip].pos[:, dir_id] = -self.p[ip].pos[:, dir_id]
        
        # 점들의 복합체 업데이트
        self.pc = self._vertcat_points([self.p[i] for i in self.mask])
        
        return self
    
    def select(self, method: str, *args, **kwargs):
        """COMPOINT 객체에서 점들 선택"""
        if method != 'index':
            # 모든 점 객체들에 select 입력 전달
            self.p = [p.select(method, *args, **kwargs) for p in self.p]
        else:
            # 인덱스를 사용한 선택
            indices = args[0]
            
            # 그룹화된 점들에 대한 인덱스
            ipt = []
            ind = []
            
            for i, point_obj in enumerate(self.p):
                if hasattr(point_obj, 'n'):
                    ipt.extend([i] * point_obj.n)
                    ind.extend(range(point_obj.n))
            
            ipt = np.array(ipt)
            ind = np.array(ind)
            
            # 선택된 인덱스 가져오기
            selected_ind = ind[indices]
            selected_ipt = ipt[indices]
            
            # 모든 점들에 대해 반복
            for i in range(len(self.p)):
                mask = selected_ipt == i
                if np.any(mask):
                    self.p[i] = self.p[i].select('index', selected_ind[mask])
        
        # 비어있지 않은 점 객체들에 대한 인덱스
        non_empty = [i for i, p in enumerate(self.p) 
                    if hasattr(p, 'pos') and len(p.pos) > 0]
        
        # 선택된 객체들만 유지
        self.p = [self.p[i] for i in non_empty]
        self.inout = [self.inout[i] for i in non_empty]
        
        # mask 객체 설정
        self.mask = list(range(len(self.p)))
        
        # 점들의 복합체
        self.pc = self._vertcat_points([self.p[i] for i in self.mask])
        
        return self
    
    def __call__(self, valpt: np.ndarray, valdef=np.nan):
        """
        ComPoint 배열 변환
        
        compoint 객체에 대해 계산된 값 배열 valpt가 주어지면,
        compoint 초기화에서 제공된 위치 pos와 동일한 수의 요소를 가진 val 배열을 반환
        """
        # val 배열 할당
        siz = valpt.shape
        
        # val 배열 할당
        if np.isnan(valdef):
            val = np.full([self.npos] + list(siz[1:]), np.nan)
        else:
            val = np.full([self.npos] + list(siz[1:]), valdef)
        
        # compoint와 원래 위치 인덱스 사이 변환
        for i in range(self.np):
            ind = self.index(i)
            val[self.ind[self.mask[i]], :] = valpt[ind, :]
        
        return val
    
    @property
    def np(self):
        """점 그룹의 수"""
        return len(self.mask)
    
    def index(self, i: int):
        """i번째 마스킹된 점 그룹의 인덱스"""
        cumsum = 0
        for j in range(i):
            if hasattr(self.p[self.mask[j]], 'n'):
                cumsum += self.p[self.mask[j]].n
        
        start_idx = cumsum
        end_idx = cumsum + (self.p[self.mask[i]].n if hasattr(self.p[self.mask[i]], 'n') else 0)
        
        return list(range(start_idx, end_idx))
    
    def __getattr__(self, name):
        """복합 객체 속성에 접근 및 compoint 배열 변환"""
        if name in ['pin', 'layer']:
            return object.__getattribute__(self, name)
        else:
            # 부모 클래스의 속성 접근
            return super().__getattribute__(name)