# compound.py - 복합체 기본 클래스
"""
Compound class - Base class for compound of points or particles in dielectric environment.
Converted from MATLAB MNPBEM @compound class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Callable


class Compound:
    """복합체 기본 클래스 - 유전체 환경에서 점들이나 입자들의 복합체를 위한 베이스 클래스"""
    
    def __init__(self, eps: List, p: List, inout: np.ndarray):
        """
        Compound 초기화
        
        Parameters:
        -----------
        eps : list
            유전체 함수들의 리스트
        p : list
            점들이나 입자들의 리스트
        inout : array
            매질 eps의 인덱스
            입자의 경우 n x 2 배열 (경계의 내부와 외부 매질 인덱스)
        """
        self.eps = eps
        self.p = p
        self.inout = np.array(inout) if inout is not None else np.array([])
        
        # 마스크와 입자/점들의 복합체
        self.mask = list(range(len(p)))
        self.pc = self._vertcat(p) if p else None
    
    def _vertcat(self, objects: List) -> Any:
        """객체들을 수직으로 연결"""
        if not objects:
            return None
        if len(objects) == 1:
            return objects[0]
        
        # 첫 번째 객체를 기준으로 시작
        result = objects[0]
        for obj in objects[1:]:
            # 실제 구현에서는 객체별로 적절한 연결 방법 사용
            try:
                result = self._combine_objects(result, obj)
            except:
                # 간단한 fallback
                result = obj
        
        return result
    
    def _combine_objects(self, obj1: Any, obj2: Any) -> Any:
        """두 객체를 결합 (실제 구현은 객체 타입에 따라 다름)"""
        # 기본적인 위치 결합 (Point나 Particle 클래스에서 오버라이드)
        if hasattr(obj1, 'pos') and hasattr(obj2, 'pos'):
            combined_pos = np.vstack([obj1.pos, obj2.pos])
            # 새 객체 생성 (실제로는 적절한 클래스 생성자 사용)
            new_obj = type(obj1)(combined_pos)
            return new_obj
        return obj1
    
    def __str__(self):
        """문자열 표현"""
        return f"Compound:\n  eps: {len(self.eps)} materials\n  p: {len(self.p)} objects\n  mask: {self.mask}"
    
    def connect(self, *args) -> List[List[np.ndarray]]:
        """복합 점들이나 입자들 사이의 연결성"""
        return connect(self, *args)
    
    def dielectric(self, enei: float, inout: int) -> List:
        """내부 또는 외부의 유전체 함수"""
        # 유전체 함수들의 테이블
        eps = [eps_func(enei) for eps_func in self.eps]
        
        # 내부 또는 외부 성분
        if len(self.inout.shape) > 1:
            return [eps[idx] for idx in self.inout[:, inout-1]]
        
        return eps
    
    def __eq__(self, other) -> bool:
        """두 객체 사이의 동등성 테스트 (위치 비교)"""
        if not hasattr(self, 'pc') or not hasattr(other, 'pc'):
            return False
        if self.pc is None or other.pc is None:
            return False
        if not hasattr(self.pc, 'pos') or not hasattr(other.pc, 'pos'):
            return False
        
        pos1 = self.pc.pos.flatten()
        pos2 = other.pc.pos.flatten()
        
        return len(pos1) == len(pos2) and np.allclose(pos1, pos2)
    
    def __ne__(self, other) -> bool:
        """두 객체 사이의 비동등성 테스트"""
        return not self.__eq__(other)
    
    def _expand(self, val) -> np.ndarray:
        """모든 점이나 입자 위치에 대한 셀 배열 확장"""
        sizes = [p.size if hasattr(p, 'size') else len(p.pos) if hasattr(p, 'pos') else 0 
                for p in [self.p[i] for i in self.mask]]
        
        if not isinstance(val, list):
            return np.tile(val, (sum(sizes), 1)) if sum(sizes) > 0 else np.array([])
        else:
            full = []
            for i, size in enumerate(sizes):
                if i < len(val):
                    full.extend([val[i]] * size)
            return np.array(full)
    
    def index(self, ipart: Union[int, List[int]]) -> List[int]:
        """주어진 입자 집합에 대한 인덱스"""
        sizes = [0] + [p.size if hasattr(p, 'size') else len(p.pos) if hasattr(p, 'pos') else 0 
                      for p in [self.p[i] for i in self.mask]]
        cumsum = np.cumsum(sizes)
        
        if isinstance(ipart, int):
            ipart = [ipart]
        
        ind = []
        for i in ipart:
            if i < len(cumsum) - 1:
                ind.extend(range(cumsum[i], cumsum[i + 1]))
        
        return ind
    
    def ipart(self, ind: Union[int, List[int]]) -> Tuple[List[int], List[int]]:
        """입자나 점 번호와 해당 인덱스 찾기"""
        sizes = [0] + [p.size if hasattr(p, 'size') else len(p.pos) if hasattr(p, 'pos') else 0 
                      for p in [self.p[i] for i in self.mask]]
        cumsum = np.cumsum(sizes)
        
        if isinstance(ind, int):
            ind = [ind]
        
        ipart = []
        for i in ind:
            # i가 어느 구간에 속하는지 찾기
            part_idx = np.where(i >= cumsum[:-1])[0]
            if len(part_idx) > 0:
                ipart.append(part_idx[-1])
            else:
                ipart.append(0)
        
        # 상대적 인덱스
        relative_ind = [ind[i] - cumsum[ipart[i]] for i in range(len(ind))]
        
        return ipart, relative_ind
    
    def mask_objects(self, ind: Optional[List[int]] = None):
        """ind로 표시된 점들이나 입자들을 마스킹"""
        if ind is None or len(ind) == 0:
            ind = list(range(len(self.p)))
        
        self.mask = ind
        self.pc = self._vertcat([self.p[i] for i in ind])
        
        return self
    
    def set_properties(self, **kwargs):
        """복합체의 속성 설정"""
        for key, value in kwargs.items():
            if hasattr(self.pc, key):
                setattr(self.pc, key, value)
        
        return self
    
    @property
    def size(self) -> List[int]:
        """점들이나 면들의 위치 수"""
        return [p.size if hasattr(p, 'size') else len(p.pos) if hasattr(p, 'pos') else 0 
                for p in [self.p[i] for i in self.mask]]
    
    @property
    def n(self) -> int:
        """총 점들이나 면들의 수"""
        return sum(self.size)
    
    @property
    def np(self) -> int:
        """점 집합이나 입자들의 수"""
        return len(self.mask)
    
    def eps1(self, enei: float) -> np.ndarray:
        """내부 유전체 함수들"""
        return self._expand(self.dielectric(enei, 1))
    
    def eps2(self, enei: float) -> np.ndarray:
        """외부 유전체 함수들"""
        return self._expand(self.dielectric(enei, 2))
    
    def expand(self, val) -> np.ndarray:
        """값 확장"""
        return self._expand(val)
    
    def __getattr__(self, name):
        """복합 객체의 점이나 입자 속성에 접근"""
        # pc 객체에서 속성 찾기
        if hasattr(self, 'pc') and self.pc is not None and hasattr(self.pc, name):
            return getattr(self.pc, name)
        
        # 기본 속성들
        if name in ['eps', 'mask', 'p', 'inout']:
            return object.__getattribute__(self, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def connect(*args) -> List[List[np.ndarray]]:
    """복합 점들이나 입자들 사이의 연결성"""
    def get_inout(p):
        """마스킹된 inout 속성 가져오기"""
        if hasattr(p, 'inout') and hasattr(p, 'mask'):
            return p.inout[p.mask, :]
        return p.inout if hasattr(p, 'inout') else np.array([[]])
    
    # 입력 추출
    if len(args) == 1:
        # 단일 입자
        inout = [get_inout(args[0])]
    elif len(args) == 2:
        if isinstance(args[1], (list, np.ndarray)) and np.issubdtype(type(args[1][0]), np.integer):
            # 단일 입자와 교체용 인덱스
            ind = np.array(args[1])
            inout_orig = get_inout(args[0])
            inout = [ind[inout_orig]]
        else:
            # 두 입자
            inout = [get_inout(args[0]), get_inout(args[1])]
    elif len(args) == 3:
        # 두 입자와 교체용 인덱스
        ind = np.array(args[2])
        inout = [ind[get_inout(args[0])], ind[get_inout(args[1])]]
    else:
        raise ValueError("Invalid number of arguments")
    
    # 연결성 행렬 계산
    n1 = inout[0].shape[1] if len(inout[0].shape) > 1 else 1
    n2 = inout[-1].shape[1] if len(inout[-1].shape) > 1 else 1
    
    # 배열 할당
    con = [[None for _ in range(n2)] for _ in range(n1)]
    
    # 점들이 서로를 볼 수 있는지 결정
    for i in range(n1):
        for j in range(n2):
            if len(inout[0].shape) > 1:
                io1 = inout[0][:, i]
            else:
                io1 = inout[0].flatten()
            
            if len(inout[-1].shape) > 1:
                io2 = inout[-1][:, j]
            else:
                io2 = inout[-1].flatten()
            
            c1 = np.tile(io1.reshape(-1, 1), (1, len(io2)))
            c2 = np.tile(io2.reshape(1, -1), (len(io1), 1))
            
            con[i][j] = np.zeros_like(c1)
            mask = c1 == c2
            con[i][j][mask] = c1[mask]
    
    return con