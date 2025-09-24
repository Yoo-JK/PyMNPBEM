# tripolygon.py - 다각형으로부터 3D 입자 생성
"""
TRIPOLYGON - 3D particle from polygon.
Converted from MATLAB MNPBEM tripolygon.m
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from ..particle import Particle
from ..polygon3 import Polygon3
from ..getbemoptions import getbemoptions


def tripolygon(poly: Union[List, 'Polygon'], edge: 'EdgeProfile', **kwargs) -> Tuple[Particle, List[Polygon3]]:
    """
    다각형으로부터 3D 입자 생성
    
    Parameters:
    -----------
    poly : list or Polygon
        2D 다각형(들)
    edge : EdgeProfile
        모서리 프로파일
    **kwargs :
        hdata : mesh2d에 전달할 데이터
        opt : mesh2d에 전달할 옵션 구조체
        refine : 모서리를 위한 세분화 함수
        
    Returns:
    --------
    p : Particle
        압출된 입자를 위한 입자 객체
    poly : list of Polygon3
        모서리 다각형들
    """
    # 단일 다각형인 경우 리스트로 변환
    if not isinstance(poly, list):
        poly = [poly]
    
    # 둥근 또는 날카로운 위쪽 및 아래쪽 모서리
    edge_pos = edge.pos
    
    # 모든 모서리가 NaN이 아니거나, 0인 점이 정확히 하나가 아닌 경우 (둥근 모서리)
    all_not_nan = np.all(~np.isnan(edge_pos[:, 0]))
    zero_count = np.sum(edge_pos[:, 0] == 0)
    
    if all_not_nan or zero_count != 1:
        # 둥근 위쪽과 아래쪽 모서리
        
        # polygon3 객체들
        poly1 = [Polygon3(p, edge.zmin) for p in poly]
        poly2 = [Polygon3(p, edge.zmax) for p in poly]
        
        # 플레이트들
        plate1, _ = _create_plate(poly1, edge, direction=-1, **kwargs)
        plate2, poly_result = _create_plate(poly2, edge, direction=1, **kwargs)
        
        # 리본
        ribbon, _, _ = _create_vribbon(poly_result, **kwargs)
        
    # 날카로운 아래쪽 모서리
    elif np.isnan(edge_pos[0, 0]):
        
        # polygon3 객체
        poly_objects = [Polygon3(p, edge.zmax) for p in poly]
        
        # 위쪽 플레이트
        plate1, poly_result = _create_plate(poly_objects, edge, direction=1, **kwargs)
        
        # 리본
        ribbon, _, poly_lower = _create_vribbon(poly_result, **kwargs)
        
        # 아래쪽 플레이트
        poly_lower_set = [p.set_properties(z=edge.zmin) for p in poly_lower]
        plate2, _ = _create_plate(poly_lower_set, edge, direction=-1, **kwargs)
        
    # 날카로운 위쪽 모서리
    else:
        
        # polygon3 객체
        poly_objects = [Polygon3(p, edge.zmin) for p in poly]
        
        # 아래쪽 플레이트
        plate1, poly_result = _create_plate(poly_objects, edge, direction=-1, **kwargs)
        
        # 리본
        ribbon, poly_upper, _ = _create_vribbon(poly_result, **kwargs)
        
        # 위쪽 플레이트
        poly_upper_set = [p.set_properties(z=edge.zmax) for p in poly_upper]
        plate2, _ = _create_plate(poly_upper_set, edge, direction=1, **kwargs)
    
    # 입자들 결합
    p = plate1.vertcat(plate2, ribbon)
    p = p.clean()
    
    return p, poly_result


def _create_plate(poly_list: List[Polygon3], edge: 'EdgeProfile', direction: int = 1, **kwargs) -> Tuple[Particle, List[Polygon3]]:
    """플레이트 생성 (간단화된 버전)"""
    # 실제 구현에서는 plate 함수가 필요
    print(f"Warning: Plate creation is simplified (direction={direction})")
    
    if not poly_list:
        return Particle(np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)), []
    
    # 기본 플레이트 생성 (매우 간단화된 버전)
    # 실제로는 polygon3의 plate 메서드를 사용해야 함
    try:
        first_poly = poly_list[0]
        if hasattr(first_poly, 'plate'):
            plate_particle, result_poly = first_poly.plate(edge=edge, dir=direction, **kwargs)
            return plate_particle, [result_poly] if not isinstance(result_poly, list) else result_poly
    except Exception as e:
        print(f"Warning: Plate creation failed: {e}")
    
    # 폴백: 빈 입자 반환
    empty_verts = np.array([]).reshape(0, 3)
    empty_faces = np.array([]).reshape(0, 3)
    return Particle(empty_verts, empty_faces), poly_list


def _create_vribbon(poly_list: List[Polygon3], **kwargs) -> Tuple[Particle, List[Polygon3], List[Polygon3]]:
    """수직 리본 생성 (간단화된 버전)"""
    # 실제 구현에서는 vribbon 함수가 필요
    print("Warning: Vertical ribbon creation is simplified")
    
    if not poly_list:
        empty_verts = np.array([]).reshape(0, 3)
        empty_faces = np.array([]).reshape(0, 3)
        return Particle(empty_verts, empty_faces), [], []
    
    # 기본 리본 생성 (매우 간단화된 버전)
    try:
        first_poly = poly_list[0]
        if hasattr(first_poly, 'vribbon'):
            ribbon_particle, upper_poly, lower_poly = first_poly.vribbon(**kwargs)
            
            # 결과를 리스트로 변환
            upper_list = [upper_poly] if not isinstance(upper_poly, list) else upper_poly
            lower_list = [lower_poly] if not isinstance(lower_poly, list) else lower_poly
            
            return ribbon_particle, upper_list, lower_list
    except Exception as e:
        print(f"Warning: Vertical ribbon creation failed: {e}")
    
    # 폴백: 빈 입자와 원래 다각형들 반환
    empty_verts = np.array([]).reshape(0, 3)
    empty_faces = np.array([]).reshape(0, 3)
    return Particle(empty_verts, empty_faces), poly_list, poly_list