# comparticlemirror.py - 거울 대칭을 가진 복합 입자 클래스
"""
ComParticleMirror class for handling compound particles with mirror symmetry.
Converted from MATLAB MNPBEM @comparticlemirror class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict
from .bembase import BemBase
from .comparticle import ComParticle
from .getbemoptions import getbemoptions


class ComParticleMirror(BemBase, ComParticle):
    """거울 대칭을 가진 복합 입자 클래스 - 유전체 환경에서 거울 대칭을 이용한 입자 복합체"""
    
    # 클래스 상수
    name = 'bemparticle'
    needs = ['sym']
    
    def __init__(self, eps, p: List, inout, *args, **kwargs):
        """
        거울 대칭을 가진 ComParticle 초기화
        
        Parameters:
        -----------
        eps : list
            유전체 함수들의 리스트
        p : list
            입자들의 리스트
        inout : array
            매질 eps의 인덱스
        *args, **kwargs : 
            closed 인수 및 옵션들 (sym 키워드 포함해야 함)
            sym: 'x', 'y', 'xy' 중 하나의 거울 대칭 키
        """
        # closed 인수 제거한 옵션 준비
        op = list(args)
        if op and isinstance(op[0], (list, int)):
            op = op[1:]
        
        # ComParticle 초기화 (closed 인수 없이)
        super().__init__(eps, p, inout, *op, **kwargs)
        
        # 거울 대칭으로 초기화
        self._init_mirror(*args, **kwargs)
    
    def _init_mirror(self, *varargin, **kwargs):
        """거울 대칭을 가진 comparticle 객체 초기화"""
        # 입력 처리
        close = []
        options = list(varargin)
        
        for i, arg in enumerate(varargin):
            if isinstance(arg, dict) or isinstance(arg, str):
                if i == 0:
                    close = []
                else:
                    close = list(varargin[:i])
                    options = list(varargin[i:])
                break
        
        # 옵션 가져오기
        if options and isinstance(options[0], dict):
            op = getbemoptions(options[0], {}, *options[1:], **kwargs)
        else:
            op = getbemoptions({}, {}, *options, **kwargs)
        
        # 대칭 설정
        self.sym = op.get('sym', 'x')  # 기본값 'x'
        
        # 거울 대칭을 위한 테이블
        if self.sym in ['x', 'y']:
            self.symtable = np.array([[1, 1], [1, -1]])  # '+', '-'
        elif self.sym == 'xy':
            self.symtable = np.array([
                [1,  1,  1,  1],  # '++'
                [1,  1, -1, -1],  # '+-'
                [1, -1,  1, -1],  # '-+'
                [1, -1, -1,  1]   # '--'
            ])
        else:
            raise ValueError(f"Invalid symmetry key: {self.sym}")
        
        # 거울 대칭을 사용하여 전체 입자 만들기
        mirror = [['x', 'xy'], ['y', 'xy']]
        
        # 입자로 시작
        p = list(self.p)
        inout = np.copy(self.inout)
        
        # 거울 대칭 연산을 적용하여 동등한 입자들 추가
        for k in range(2):  # x, y 방향
            current_len = len(p)
            for i in range(current_len):
                if self.sym in mirror[k]:
                    # k 방향으로 flip된 입자 추가
                    flipped_particle = p[i].flip(k)
                    p.append(flipped_particle)
                    inout = np.vstack([inout, inout[i:i+1, :]])
        
        # 전체 입자 초기화
        self.pfull = ComParticle(self.eps, p, inout, *options, **kwargs)
        
        # 닫힌 표면을 위한 인덱스
        if close:
            self.closed_surface(*close)
    
    def closed_surface(self, *varargin):
        """Green 함수 평가를 위한 입자들의 닫힌 표면 표시"""
        # 동등한 입자 표면에 대한 인덱스
        ind = np.arange(len(self.pfull.p)).reshape(-1, self.symtable.shape[1])
        
        for arg in varargin:
            # pfull에 저장된 입자(들)의 인덱스인 경우
            if not isinstance(arg, list) or not isinstance(arg[0], dict):
                indices = np.array(arg) if isinstance(arg, list) else np.array([arg])
                
                # 동등한 입자들의 테이블 생성
                signs = np.sign(indices.reshape(-1, 1))
                abs_indices = np.abs(indices)
                
                tab = (signs * ind[abs_indices, :]).flatten()
                
                for j in tab:
                    if self.pfull.closed[abs(j)] is None:
                        self.pfull.closed[abs(j)] = tab.tolist()
            
            # 추가 입자인 경우
            else:
                # 동등한 입자들의 테이블
                tab = ind[arg[0], :].flatten()
                
                # 닫힌 입자
                particles_to_combine = [self.pfull.p[idx] for idx in tab] + list(arg[1:])
                p = self._vertcat_particles(particles_to_combine)
                
                # 동등한 입자들에 닫힌 입자 추가
                for j in range(len(tab)):
                    self.pfull.closed[j] = p
    
    def __str__(self):
        """문자열 표현"""
        return f"ComParticleMirror:\n  eps: {len(self.eps)} materials\n  p: {len(self.p)} particles\n  sym: {self.sym}\n  mask: {self.mask}"
    
    def full(self):
        """거울 대칭으로 생성된 전체 입자"""
        return self.pfull
    
    def closed_particle(self, ind: int) -> Tuple[Optional[Any], Optional[int], Optional[np.ndarray]]:
        """입자 ind에 대한 닫힌 표면을 가진 입자 반환"""
        p, direction, _ = self.pfull.closed_particle(ind)
        return p, direction, None  # loc는 항상 None
    
    def symindex(self, tab: Union[List, np.ndarray]) -> Optional[int]:
        """대칭 테이블 내에서 대칭 값들의 인덱스"""
        tab = np.array(tab)
        
        # 각 행과 비교하여 일치하는 행 찾기
        matches = np.all(self.symtable == tab.reshape(1, -1), axis=1)
        indices = np.where(matches)[0]
        
        return indices[0] if len(indices) > 0 else None
    
    def symvalue(self, key: Union[str, List[str]]) -> np.ndarray:
        """주어진 키에 대한 대칭 값들"""
        if isinstance(key, list):
            val = []
            for k in key:
                val.append(self.symvalue(k))
            return np.array(val)
        else:
            if key == '+':
                return np.array([1, 1])
            elif key == '-':
                return np.array([1, -1])
            elif key == '++':
                return np.array([1, 1, 1, 1])
            elif key == '+-':
                return np.array([1, 1, -1, -1])
            elif key == '-+':
                return np.array([1, -1, 1, -1])
            elif key == '--':
                return np.array([1, -1, -1, 1])
            else:
                raise ValueError(f"Invalid symmetry key: {key}")
    
    def mask_particles(self, ind: List[int]):
        """ind로 표시된 입자들 마스킹"""
        # 부모 클래스의 mask 메서드 호출
        super().mask_particles(ind)
        
        # 동등한 입자들에 대한 인덱스
        ip = np.arange(len(self.pfull.p)).reshape(len(self.p), -1)
        
        # 전체 입자 마스킹
        full_indices = ip[ind, :].flatten()
        self.pfull = self.pfull.mask_particles(full_indices.tolist())
        
        return self
    
    def __getattr__(self, name):
        """대칭 및 comparticle 속성에 대한 접근"""
        if name in ['sym', 'symtable', 'pfull']:
            return object.__getattribute__(self, name)
        elif name in ['symindex', 'symvalue', 'closed', 'closedparticle', 'mask']:
            return object.__getattribute__(self, name)
        else:
            # 부모 클래스의 속성 접근
            return super().__getattribute__(name)
    
    # Hidden 메서드들 - 기본적으로 부모 클래스 메서드 사용
    def clean(self, *args, **kwargs):
        """모든 입자 정리"""
        result = super().clean(*args, **kwargs)
        self.pfull = self.pfull.clean(*args, **kwargs)
        return result
    
    def flip(self, *args, **kwargs):
        """모든 입자 뒤집기"""
        result = super().flip(*args, **kwargs)
        self.pfull = self.pfull.flip(*args, **kwargs)
        return result
    
    def flipfaces(self, *args, **kwargs):
        """모든 입자의 면 뒤집기"""
        result = super().flipfaces(*args, **kwargs)
        self.pfull = self.pfull.flipfaces(*args, **kwargs)
        return result
    
    def norm(self, *args, **kwargs):
        """정규화"""
        result = super().norm(*args, **kwargs)
        self.pfull = self.pfull.norm(*args, **kwargs)
        return result
    
    def rot(self, *args, **kwargs):
        """모든 입자 회전"""
        result = super().rot(*args, **kwargs)
        self.pfull = self.pfull.rot(*args, **kwargs)
        return result
    
    def scale(self, *args, **kwargs):
        """모든 입자 스케일링"""
        result = super().scale(*args, **kwargs)
        self.pfull = self.pfull.scale(*args, **kwargs)
        return result
    
    def select(self, *args, **kwargs):
        """입자 선택"""
        result = super().select(*args, **kwargs)
        # 전체 입자도 적절히 선택해야 함
        return result
    
    def shift(self, *args, **kwargs):
        """모든 입자 이동"""
        result = super().shift(*args, **kwargs)
        self.pfull = self.pfull.shift(*args, **kwargs)
        return result