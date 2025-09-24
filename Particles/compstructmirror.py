# compstructmirror.py - 거울 대칭을 가진 복합 구조체 클래스
"""
CompStructMirror class for structure of compound points/particles with mirror symmetry.
Converted from MATLAB MNPBEM @compstructmirror class.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
from .compstruct import CompStruct


class CompStructMirror:
    """거울 대칭을 가진 복합 구조체 클래스 - 점들이나 입자들의 복합체를 위한 구조체"""
    
    def __init__(self, p, enei: float, fun: Callable):
        """
        CompStructMirror 초기화
        
        Parameters:
        -----------
        p : object
            점들이나 이산화된 입자 표면
        enei : float
            진공에서 빛의 파장
        fun : callable
            확장 함수
        """
        self.p = p
        self.enei = enei
        self.fun = fun
        self.val = None  # COMPSTRUCT 객체들을 담을 셀 배열
    
    def __str__(self):
        """문자열 표현"""
        return f"CompStructMirror:\n  p: {type(self.p).__name__}\n  enei: {self.enei}\n  val: {len(self.val) if self.val else 0} objects\n  fun: {self.fun.__name__ if hasattr(self.fun, '__name__') else 'function'}"
    
    def full(self) -> Union[Any, Tuple[Any, ...]]:
        """거울 대칭을 가진 객체를 전체 크기로 확장"""
        return self.fun(self)
    
    def expand(self) -> Union[CompStruct, Tuple[CompStruct, ...]]:
        """거울 대칭을 사용하여 구조체를 전체 입자 크기로 확장"""
        if not self.val or len(self.val) == 0:
            raise ValueError("val is empty - no structures to expand")
        
        # 필드 이름 가져오기
        val_dict = self.val[0].__dict__ if hasattr(self.val[0], '__dict__') else {}
        names = [name for name in val_dict.keys() if name != 'symval']
        
        # 반환값 초기화
        results = []
        num_outputs = len(self.val)
        
        for i in range(num_outputs):
            # 전체 입자에 대한 새로운 compstruct 생성
            full_p = self.p.full() if hasattr(self.p, 'full') else self.p
            result = CompStruct(full_p, self.enei)
            
            for field_name in names:
                # 값과 대칭값 가져오기
                val1 = getattr(self.val[i], field_name, None)
                symval = getattr(self.val[i], 'symval', None)
                
                if val1 is None or symval is None:
                    continue
                
                val2 = val1.copy() if hasattr(val1, 'copy') else val1
                
                # 필드가 스칼라인지 벡터인지 결정
                scalar_fields = {'phi', 'phip', 'phi1', 'phi2', 'phi1p', 'phi2p', 
                               'sig', 'sig1', 'sig2'}
                vector_fields = {'a1', 'a1p', 'a2', 'a2p', 'e', 'h', 'h1', 'h2'}
                
                if field_name in scalar_fields:
                    # 스칼라 필드 처리
                    for k in range(1, symval.shape[1]):  # k = 2부터 시작 (MATLAB 1-based)
                        # 대칭값과 곱하기
                        expanded_val = symval[-1, k] * val1
                        val2 = np.concatenate([val2, expanded_val], axis=0)
                
                elif field_name in vector_fields:
                    # 벡터 필드 처리
                    for k in range(1, symval.shape[1]):  # k = 2부터 시작 (MATLAB 1-based)
                        # VAL의 복사본 만들기
                        val3 = val1.copy() if hasattr(val1, 'copy') else np.array(val1)
                        
                        # 대칭값과 곱하기
                        for l in range(3):  # x, y, z 성분
                            if len(val3.shape) == 3:  # (points, 3, other_dims)
                                val3[:, l, :] = symval[l, k] * val1[:, l, :]
                            elif len(val3.shape) == 2:  # (points, 3)
                                val3[:, l] = symval[l, k] * val1[:, l]
                        
                        val2 = np.concatenate([val2, val3], axis=0)
                
                else:
                    raise ValueError(f"Field name '{field_name}' not known!")
                
                # 확장된 필드를 전체 입자를 위한 구조체에 추가
                setattr(result, field_name, val2)
            
            results.append(result)
        
        # 결과 반환 (단일 객체 또는 튜플)
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
    
    def set_val(self, val: List):
        """val 속성 설정"""
        self.val = val
        return self
    
    def get_field_names(self) -> List[str]:
        """val 객체의 필드 이름들 반환"""
        if not self.val or len(self.val) == 0:
            return []
        
        val_dict = self.val[0].__dict__ if hasattr(self.val[0], '__dict__') else {}
        return [name for name in val_dict.keys() if name != 'symval']
    
    def apply_symmetry(self, field_name: str, val: np.ndarray, symval: np.ndarray) -> np.ndarray:
        """특정 필드에 대칭 변환 적용"""
        scalar_fields = {'phi', 'phip', 'phi1', 'phi2', 'phi1p', 'phi2p', 
                        'sig', 'sig1', 'sig2'}
        vector_fields = {'a1', 'a1p', 'a2', 'a2p', 'e', 'h', 'h1', 'h2'}
        
        result = val.copy() if hasattr(val, 'copy') else np.array(val)
        
        if field_name in scalar_fields:
            # 스칼라 필드의 대칭 변환
            for k in range(1, symval.shape[1]):
                expanded_val = symval[-1, k] * val
                result = np.concatenate([result, expanded_val], axis=0)
        
        elif field_name in vector_fields:
            # 벡터 필드의 대칭 변환
            for k in range(1, symval.shape[1]):
                val_copy = val.copy() if hasattr(val, 'copy') else np.array(val)
                
                for l in range(3):  # x, y, z 성분
                    if len(val_copy.shape) == 3:
                        val_copy[:, l, :] = symval[l, k] * val[:, l, :]
                    elif len(val_copy.shape) == 2:
                        val_copy[:, l] = symval[l, k] * val[:, l]
                
                result = np.concatenate([result, val_copy], axis=0)
        
        else:
            raise ValueError(f"Field name '{field_name}' not known!")
        
        return result