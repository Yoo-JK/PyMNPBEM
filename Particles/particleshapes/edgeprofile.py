# edgeprofile.py - 부드러운 모서리 프로파일 클래스
"""
EdgeProfile class for smoothed edge profile for use with TRIPOLYGON.
Edge profile for a supercircle (d,z) = (cos(phi^e), sin(phi^e)).
Converted from MATLAB MNPBEM @edgeprofile class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Any, Dict
from scipy.interpolate import splprep, splev, interp1d

from ..getbemoptions import getbemoptions


class EdgeProfile:
    """모서리 프로파일 클래스 - TRIPOLYGON과 함께 사용하기 위한 부드러운 모서리의 프로파일"""
    
    def __init__(self, *args, **kwargs):
        """
        EdgeProfile 초기화
        
        Usage:
        ------
        EdgeProfile() : 빈 모서리 프로파일
        EdgeProfile(height, **kwargs) : 높이로 프로파일 초기화
        EdgeProfile(height, nz, **kwargs) : 높이와 z-값 개수로 초기화
        EdgeProfile(pos, z, **kwargs) : 명시적 위치와 z-값으로 초기화
        
        Parameters:
        -----------
        height : float
            입자의 높이
        nz : int
            z-값의 개수
        pos : array
            모서리 프로파일의 (z,d) 값들
        z : array
            다각형을 압출하기 위한 z-값들
            
        **kwargs :
            e : 초원의 지수 (기본값: 0.4)
            dz : d >= 0인 경우 z의 범위 조정 (기본값: 0.15)
            min : 모서리 프로파일의 최소 z-값
            max : 모서리 프로파일의 최대 z-값
            center : 모서리 프로파일의 중심 z-값
            mode : '00', '10', '01', '11', '20', '02'
                  둥근(0), 직선(1), 부분적으로 둥근(2) 모서리
        """
        self.pos = None  # (z,d) 모서리 프로파일 값들
        self.z = None    # 다각형 압출을 위한 z-값들
        
        self._init(*args, **kwargs)
    
    def _init(self, *args, **kwargs):
        """모서리 프로파일 초기화"""
        if not args:
            return  # 빈 모서리 프로파일
        
        # 명시적 생성자 (POS와 Z)
        if len(args) >= 2 and hasattr(args[0], '__len__') and len(args[0]) > 1:
            self.pos = np.array(args[0])
            self.z = np.array(args[1])
            op = getbemoptions(*args[2:], **kwargs)
        else:
            # 높이 기반 생성자
            if len(args) == 1 or (len(args) >= 2 and not isinstance(args[1], (int, float))):
                height = args[0]
                nz = 7  # 기본값
                op = getbemoptions(*args[1:], **kwargs)
            else:
                height = args[0]
                nz = args[1]
                op = getbemoptions(*args[2:], **kwargs)
            
            # 기본 모드
            mode = op.get('mode', '00')
            
            if mode == '11':
                # 모서리 프로파일
                self.pos = np.array([
                    [np.nan, 0],
                    [0, -0.5 * height],
                    [0, 0.5 * height], 
                    [np.nan, 0]
                ])
                # 대표 z-값들
                self.z = np.linspace(-0.5, 0.5, nz) * height
            else:
                # 초원의 지수
                e = op.get('e', 0.4)
                # d >= 0인 경우의 dz
                dz = op.get('dz', 0.15)
                
                # 초원
                def pows(z_val):
                    return np.sign(z_val) * np.abs(z_val) ** e
                
                # 각도들
                phi = np.linspace(-np.pi/2, np.pi/2, 51)
                
                x = pows(np.cos(phi))
                z_val = pows(np.sin(phi))
                
                # dz 조건에 맞는 인덱스 찾기
                ind = np.argmin(np.abs(z_val - (1 - dz)))
                
                # 모서리 프로파일 만들기
                self.pos = 0.5 * height * np.column_stack([x - x[ind], z_val])
                
                # z를 따른 대표값들
                z_vals = np.linspace(-1, 1, nz)
                self.z = self.pos[ind, 1] * np.abs(z_vals) ** e * np.sign(z_vals)
                
                # 인덱스들
                ind2 = (self.pos[:, 1] > 0) & (self.pos[:, 0] >= 0)
                ind3 = self.pos[:, 1] == 0
                ind4 = (self.pos[:, 1] < 0) & (self.pos[:, 0] >= 0)
                ind5 = (self.pos[:, 1] < 0) & (self.pos[:, 0] < 0)
                
                # 날카로운 위쪽 모서리
                if mode[0] != '0':
                    # 시프트 값
                    dz_shift = 0.5 * height - np.max(self.pos[ind2, 1])
                    
                    # 위쪽 위치들의 d-값 수정
                    if mode[0] == '1':
                        self.pos[ind2, 0] = np.max(self.pos[ind2, 0])
                    
                    # 위쪽 위치들의 z-값 수정
                    self.pos[ind2, 1] = self.pos[ind2, 1] + dz_shift
                    
                    # 선택된 위치들과 z-값들 유지
                    mask = ind2 | ind3 | ind4 | ind5
                    self.pos = np.vstack([self.pos[mask], [np.nan, 0]])
                    self.z[self.z > 0] = self.z[self.z > 0] + dz_shift
                
                # 인덱스들 재계산
                ind1 = (self.pos[:, 1] > 0) & (self.pos[:, 0] < 0)
                ind2 = (self.pos[:, 1] > 0) & (self.pos[:, 0] >= 0)
                ind3 = self.pos[:, 1] == 0
                ind4 = (self.pos[:, 1] < 0) & (self.pos[:, 0] >= 0)
                
                # 날카로운 아래쪽 모서리
                if mode[1] != '0':
                    # 시프트 값
                    dz_shift = 0.5 * height + np.min(self.pos[ind4, 1])
                    
                    # 아래쪽 위치들의 d-값 수정
                    if mode[1] == '1':
                        self.pos[ind4, 0] = np.max(self.pos[ind4, 0])
                    
                    # 아래쪽 위치들의 z-값 수정
                    self.pos[ind4, 1] = self.pos[ind4, 1] - dz_shift
                    
                    # 선택된 위치들과 z-값들 유지
                    mask = ind1 | ind2 | ind3 | ind4
                    self.pos = np.vstack([[np.nan, 0], self.pos[mask]])
                    self.z[self.z < 0] = self.z[self.z < 0] - dz_shift
        
        # 시프트 인수 처리
        dz_final = 0
        if 'max' in op:
            dz_final = op['max'] - np.nanmax(self.pos[:, 1])
        elif 'min' in op:
            dz_final = op['min'] - np.nanmin(self.pos[:, 1])
        elif 'center' in op:
            dz_final = op['center']
        
        # 위치와 z-값들 시프트
        self.pos[:, 1] = self.pos[:, 1] + dz_final
        self.z = self.z + dz_final
    
    def __str__(self):
        """문자열 표현"""
        return f"EdgeProfile:\n  pos: {self.pos.shape if self.pos is not None else None}\n  z: {self.z.shape if self.z is not None else None}"
    
    def plot(self):
        """모서리 프로파일 플롯"""
        if self.pos is not None and self.z is not None:
            # NaN 값들을 제거하여 플롯
            valid_mask = ~np.isnan(self.pos[:, 0]) & ~np.isnan(self.pos[:, 1])
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.pos[valid_mask, 0], self.pos[valid_mask, 1], 'o-', label='Edge Profile')
            plt.plot(np.zeros_like(self.z), self.z, 'r+', label='Representative Z-values')
            plt.xlabel('d')
            plt.ylabel('z')
            plt.title('Edge Profile')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def hshift(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """수평 방향으로 모서리의 노드들 변위"""
        if self.pos is None or len(self.pos) == 0:
            return z
        
        z = np.asarray(z)
        
        # z-값이 적절한 범위에 있는지 확인
        z_min, z_max = np.nanmin(self.pos[:, 1]), np.nanmax(self.pos[:, 1])
        if not np.all((z >= z_min) & (z <= z_max)):
            raise ValueError(f"z values must be in range [{z_min}, {z_max}]")
        
        # NaN이 아닌 인덱스
        valid_mask = ~np.isnan(self.pos[:, 0]) & ~np.isnan(self.pos[:, 1])
        
        # x 방향으로 변위
        if np.sum(valid_mask) < 2:
            return np.zeros_like(z)
        
        # 스플라인 보간
        interp_func = interp1d(self.pos[valid_mask, 1], self.pos[valid_mask, 0], 
                              kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        return interp_func(z)
    
    def vshift(self, z: float, d: float) -> float:
        """수직 방향으로 모서리의 노드들 변위"""
        if self.pos is None or len(self.pos) == 0:
            return 0
        
        # z-값이 위쪽 또는 아래쪽 모서리인지 확인
        z_min, z_max = np.nanmin(self.pos[:, 1]), np.nanmax(self.pos[:, 1])
        
        if not (np.isclose(z, z_min) or np.isclose(z, z_max)):
            raise ValueError("z must be either upper or lower edge position")
        
        # 위쪽 모서리
        if np.isclose(z, z_max):
            if np.isnan(self.pos[-1, 0]):
                return 0
            else:
                upper_mask = self._upper_indices()
                pos = self.pos[upper_mask]
                valid_mask = ~np.isnan(pos[:, 0]) & ~np.isnan(pos[:, 1])
                
                if np.sum(valid_mask) < 2:
                    return 0
                
                d_clamped = np.maximum(np.min(pos[valid_mask, 0]), -np.abs(d))
                interp_func = interp1d(pos[valid_mask, 0], pos[valid_mask, 1], 
                                     kind='cubic', bounds_error=False, fill_value='extrapolate')
                return interp_func(d_clamped) - np.max(pos[valid_mask, 1])
        
        # 아래쪽 모서리
        else:
            if np.isnan(self.pos[0, 0]):
                return 0
            else:
                lower_mask = self._lower_indices()
                pos = self.pos[lower_mask]
                valid_mask = ~np.isnan(pos[:, 0]) & ~np.isnan(pos[:, 1])
                
                if np.sum(valid_mask) < 2:
                    return 0
                
                d_clamped = np.maximum(np.min(pos[valid_mask, 0]), -np.abs(d))
                interp_func = interp1d(pos[valid_mask, 0], pos[valid_mask, 1], 
                                     kind='cubic', bounds_error=False, fill_value='extrapolate')
                return interp_func(d_clamped) - np.min(pos[valid_mask, 1])
    
    def _upper_indices(self) -> np.ndarray:
        """위쪽 모서리를 위한 인덱스들"""
        if len(self.pos) < 2:
            return np.array([])
        
        diff_x = np.diff(self.pos[:, 0])
        ind = diff_x < 0
        ind = np.append(ind, ind[-1] if len(ind) > 0 else False)
        
        # DX가 부호를 바꾸는 첫 번째 요소 찾기
        sign_changes = np.where(ind != ind[-1])[0]
        if len(sign_changes) > 0:
            last_change = sign_changes[-1]
            ind[:last_change+1] = False
        
        return ind
    
    def _lower_indices(self) -> np.ndarray:
        """아래쪽 모서리를 위한 인덱스들"""
        if len(self.pos) < 2:
            return np.array([])
        
        diff_x = np.diff(self.pos[:, 0])
        ind = diff_x > 0
        
        # DX가 부호를 바꾸는 첫 번째 요소 찾기
        sign_changes = np.where(ind != ind[0])[0]
        if len(sign_changes) > 0:
            first_change = sign_changes[0]
            ind[first_change:] = False
        
        return ind
    
    @property
    def dmin(self) -> float:
        """모서리 프로파일의 최소 d-값"""
        return np.nanmin(self.pos[:, 0]) if self.pos is not None else None
    
    @property
    def dmax(self) -> float:
        """모서리 프로파일의 최대 d-값"""
        return np.nanmax(self.pos[:, 0]) if self.pos is not None else None
    
    @property
    def zmin(self) -> float:
        """모서리 프로파일의 최소 z-값"""
        return np.nanmin(self.pos[:, 1]) if self.pos is not None else None
    
    @property
    def zmax(self) -> float:
        """모서리 프로파일의 최대 z-값"""
        return np.nanmax(self.pos[:, 1]) if self.pos is not None else None