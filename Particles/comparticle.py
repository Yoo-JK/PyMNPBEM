# comparticle.py - 복합 입자 클래스 (다중 입자를 하나의 복합체로 관리)
"""
ComParticle class for handling compound of particles in dielectric environment.
Converted from MATLAB MNPBEM @comparticle class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict
from .compound import Compound
from .particle import Particle
from .quadface import quadface
from .bemoptions import bemoptions
from .curved import curved
from .flat import flat


class ComParticle(Compound):
    """복합 입자 클래스 - 유전체 환경에서 여러 입자들의 복합체를 표현"""
    
    def __init__(self, eps, p: List, inout, *args, **kwargs):
        """
        ComParticle 초기화
        
        Parameters:
        -----------
        eps : list
            유전체 함수들의 리스트
        p : list
            입자들의 리스트
        inout : array
            매질 eps의 인덱스
        *args, **kwargs : 
            closed 함수에 전달될 옵션들
        """
        # 입력 파라미터 추출 및 closed 인수 획득
        p, close = self._get_input(p, *args, **kwargs)
        
        # compound 객체 초기화
        super().__init__(eps, p, inout)
        
        # closed 속성 처리
        self._init_closed(*close)
    
    def _get_input(self, p: List, *varargin) -> Tuple[List, List]:
        """입력에서 입자 옵션과 closed 인수들을 분리"""
        close = []
        op = [{}]
        
        # closed 인수와 옵션 분리
        for i, arg in enumerate(varargin):
            if isinstance(arg, dict) or isinstance(arg, str):
                if i == 0:
                    close = []
                    op = list(varargin)
                else:
                    close = list(varargin[:i])
                    op = list(varargin[i:])
                break
        
        # varargin에 하나의 인수만 있고 close가 설정되지 않은 경우
        if len(varargin) == 1 and not close:
            close = list(varargin)
        
        # BEM 옵션 가져오기
        if op and op != [{}]:
            op_dict = bemoptions(*op)
        else:
            op_dict = {}
        
        # 입자들에 대해 반복
        for i, particle in enumerate(p):
            # 경계 적분 매개변수
            particle.quad = quadface(op_dict)
            
            # 곡면 입자 경계
            if 'interp' in op_dict and op_dict['interp'] == 'curv':
                p[i] = curved(particle)
            else:
                p[i] = flat(particle)
        
        return p, close
    
    def _init_closed(self, *varargin):
        """COMPARTICLE 객체의 속성 설정"""
        # closed surfaces를 위한 인덱스 (COMPGREEN에서 사용)
        self.closed = [None] * len(self.p)
        
        # closed 인수 처리
        if varargin:
            self.closed_surface(*varargin)
    
    def closed_surface(self, *varargin):
        """닫힌 표면의 입자들을 표시 (compgreen에서 사용)"""
        for arg in varargin:
            # obj에 저장된 입자(들)의 인덱스인 경우
            if not isinstance(arg, list) or not isinstance(arg[0], dict):
                if isinstance(arg, (list, np.ndarray)):
                    for ind in arg:
                        # 이전에 설정되지 않은 경우 closed 속성 설정
                        if self.closed[abs(ind)] is None:
                            self.closed[abs(ind)] = arg
                else:
                    # 단일 인덱스인 경우
                    if self.closed[abs(arg)] is None:
                        self.closed[abs(arg)] = [arg]
            # 추가 입자인 경우
            else:
                idx = arg[0]
                particles = arg[1:]
                self.closed[idx] = [self.p[idx]] + list(particles)
    
    def closed_particle(self, ind: int) -> Tuple[Optional[Particle], Optional[int], Optional[np.ndarray]]:
        """인덱싱된 입자의 닫힌 표면을 가진 입자 반환"""
        if self.closed[ind] is None:
            return None, None, None
        elif isinstance(self.closed[ind], Particle):
            return self.closed[ind], 1, None
        else:
            closed = np.array(self.closed[ind])
            direction = np.sign(closed[np.abs(closed) == ind])[0]
            
            # 닫힌 입자 표면 조합
            particles = [self.p[abs(idx)] for idx in closed]
            for i, idx in enumerate(closed):
                if np.sign(idx) != direction:
                    particles[i] = particles[i].flipfaces()
            
            # 입자들을 수직으로 연결
            p = self._vertcat_particles(particles)
            
            # 닫힌 입자에 대한 인덱스
            if np.all(closed > 0):
                # pos를 이용한 멤버십 확인 (단순화된 버전)
                try:
                    loc = self._find_positions(p.pos, self.pc.pos)
                except:
                    loc = None
            else:
                loc = None
            
            return p, direction, loc
    
    def _vertcat_particles(self, particles: List[Particle]) -> Particle:
        """입자들을 수직으로 연결"""
        # 실제 구현에서는 입자들의 메시를 결합하는 로직 필요
        # 여기서는 단순화된 버전
        if len(particles) == 1:
            return particles[0]
        
        # 첫 번째 입자를 기준으로 시작
        result = particles[0]
        for particle in particles[1:]:
            # 입자들을 결합하는 로직 (실제로는 메시 결합 필요)
            result = self._combine_particles(result, particle)
        
        return result
    
    def _combine_particles(self, p1: Particle, p2: Particle) -> Particle:
        """두 입자를 결합 (단순화된 버전)"""
        # 실제 구현에서는 메시 결합 로직 필요
        return p1  # 임시 구현
    
    def _find_positions(self, pos1: np.ndarray, pos2: np.ndarray) -> Optional[np.ndarray]:
        """위치 배열에서 일치하는 인덱스 찾기"""
        # 단순화된 구현
        try:
            # 각 pos1의 행이 pos2에 있는지 확인
            loc = []
            for i, row in enumerate(pos1):
                matches = np.where(np.all(np.isclose(pos2, row), axis=1))[0]
                if len(matches) > 0:
                    loc.append(matches[0])
                else:
                    return None
            return np.array(loc)
        except:
            return None
    
    def __str__(self):
        """문자열 표현"""
        return f"ComParticle:\n  eps: {len(self.eps)} materials\n  p: {len(self.p)} particles\n  closed: {sum(1 for c in self.closed if c is not None)} closed surfaces"
    
    def norm(self):
        """이산화된 입자 표면을 위한 보조 정보"""
        self.pc = self._vertcat_particles([p for p in self.p if p is not None])
        return self
    
    # Particle 클래스의 메서드들
    def clean(self, *args, **kwargs):
        """모든 입자에 particle/clean 적용"""
        return self._comp_fun(lambda p: p.clean(*args, **kwargs))
    
    def flip(self, *args, **kwargs):
        """모든 입자에 particle/flip 적용"""
        return self._comp_fun(lambda p: p.flip(*args, **kwargs))
    
    def flipfaces(self, *args, **kwargs):
        """모든 입자에 particle/flipfaces 적용"""
        return self._comp_fun(lambda p: p.flipfaces(*args, **kwargs))
    
    def rot(self, *args, **kwargs):
        """모든 입자에 particle/rot 적용"""
        return self._comp_fun(lambda p: p.rot(*args, **kwargs))
    
    def scale(self, *args, **kwargs):
        """모든 입자에 particle/scale 적용"""
        return self._comp_fun(lambda p: p.scale(*args, **kwargs))
    
    def shift(self, *args, **kwargs):
        """모든 입자에 particle/shift 적용"""
        return self._comp_fun(lambda p: p.shift(*args, **kwargs))
    
    def curvature(self, *args, **kwargs):
        """입자의 곡률"""
        return self.pc.curvature(*args, **kwargs)
    
    def quad(self, *args, **kwargs):
        """경계 요소에 대한 적분"""
        return self.pc.quad(*args, **kwargs)
    
    def quadpol(self, *args, **kwargs):
        """극좌표를 사용한 경계 요소 적분"""
        return self.pc.quadpol(*args, **kwargs)
    
    def deriv(self, v):
        """표면에 정의된 함수의 접선 도함수"""
        return self.pc.deriv(v)
    
    def interp(self, *args, **kwargs):
        """면에서 꼭짓점으로 (또는 그 반대로) 값 보간"""
        return self.pc.interp(*args, **kwargs)
    
    def plot(self, *args, **kwargs):
        """이산화된 입자 표면이나 표면에 주어진 값들을 플롯"""
        return self.pc.plot(*args, **kwargs)
    
    def plot2(self, *args, **kwargs):
        """복합 입자를 위한 플롯 함수"""
        return self.pc.plot2(*args, **kwargs)
    
    def select(self, *args, **kwargs):
        """COMPARTICLE 객체에서 면 선택"""
        if args and args[0] == 'index':
            # 인덱스를 사용한 선택
            indices = args[1]
            
            # 그룹화된 입자들에 대한 인덱스
            ipt = []
            ind = []
            
            for i, particle in enumerate(self.p):
                ipt.extend([i] * particle.n)
                ind.extend(range(particle.n))
            
            ipt = np.array(ipt)
            ind = np.array(ind)
            
            # 선택된 인덱스 가져오기
            selected_ind = ind[indices]
            selected_ipt = ipt[indices]
            
            # 모든 점에 대해 반복
            for i in range(len(self.p)):
                mask = selected_ipt == i
                if np.any(mask):
                    self.p[i] = self.p[i].select('index', selected_ind[mask])
            
            # 비어있지 않은 입자 객체들에 대한 인덱스
            non_empty = [i for i, p in enumerate(self.p) if hasattr(p, 'pos') and len(p.pos) > 0]
            
            # 선택된 객체들만 유지
            self.p = [self.p[i] for i in non_empty]
            self.inout = self.inout[non_empty, :]
            
            # closed 인수 재설정
            self.closed = [None] * len(self.p)
            
            # mask 객체 설정
            self.mask = list(range(len(self.p)))
            
            # 입자들의 복합체
            self.pc = self._vertcat_particles([self.p[i] for i in self.mask])
        else:
            # 다른 선택 방법들
            self.p = [p.select(*args, **kwargs) for p in self.p]
        
        return self
    
    def vertices(self, ind: int, *args, **kwargs):
        """인덱싱된 면의 꼭짓점"""
        ip, face_ind = self._ipart(ind)
        return self.p[ip].vertices(face_ind, *args, **kwargs)
    
    def _ipart(self, ind: int) -> Tuple[int, int]:
        """전체 인덱스에서 입자 인덱스와 면 인덱스 분리"""
        # 단순화된 구현
        cumsum = 0
        for i, particle in enumerate(self.p):
            if hasattr(particle, 'n') and cumsum + particle.n > ind:
                return i, ind - cumsum
            elif hasattr(particle, 'n'):
                cumsum += particle.n
        return 0, ind
    
    def mask_particles(self, ind: List[int]):
        """ind로 표시된 입자들을 마스킹"""
        return super().mask(ind)
    
    def _comp_fun(self, fun):
        """객체의 모든 입자에 함수 적용"""
        for i in range(len(self.p)):
            self.p[i] = fun(self.p[i])
        
        # 보조 정보
        self.norm()
        return self
    
    def __getattr__(self, name):
        """closed 및 입자 속성에 대한 접근"""
        if name == 'closed':
            # 마스킹된 입자들만 고려
            return [self.closed[i] for i in self.mask if i < len(self.closed)]
        else:
            # 기본 클래스의 속성 접근
            return super().__getattribute__(name)