# sphtriangulate.py - 단위구면 점들의 삼각화
"""
Triangulates a set of points on the unit sphere.
Using stereographical projection. Adapted from Tianli Yu's MATLAB implementation.
Converted from MATLAB MNPBEM misc/sphtriangulate.m
"""

import numpy as np
from typing import Tuple
from scipy.spatial import Delaunay, ConvexHull


def sphtriangulate(verts: np.ndarray) -> np.ndarray:
    """
    단위구면 상의 점들을 삼각화
    
    스테레오그래픽 투영을 사용하여 입력 배열 verts에 대한 삼각화된 표면을 반환합니다.
    
    단계:
    1. 첫 번째 꼭짓점을 투영 중심으로 사용하고 다른 모든 점들을 평면에 투영
    2. Delaunay 삼각화를 호출하여 평면의 점들을 삼각화
    3. 첫 번째 점을 평면의 다른 꼭짓점들의 볼록껍질에 연결
    
    Parameters:
    -----------
    verts : ndarray, shape (n, 3)
        꼭짓점들의 n x 3 행렬
        
    Returns:
    --------
    faces : ndarray, shape (m, 3)
        면들의 m x 3 행렬
    """
    n = verts.shape[0]
    
    if n < 4:
        raise ValueError("At least 4 vertices are required for triangulation")
    
    ## 단계 1: 투영
    # 첫 번째 꼭짓점을 투영 중심으로 선택
    center = verts[0, :]
    
    # 첫 번째 점을 [0, 0, -1]로 회전시키는 회전 행렬 구축
    r3 = -center
    
    if center[2] != 0:
        r2 = np.array([0, -r3[2], r3[1]])
    else:
        r2 = np.array([-r3[1], r3[0], 0])
    
    r2 = r2 / np.linalg.norm(r2)
    r1 = np.cross(r3, r2)
    
    rot = np.array([r1, r2, r3])
    
    # 회전된 꼭짓점 리스트 계산
    vertr = verts[1:n, :] @ rot.T
    
    # 투영 중심 [0, 0, -1]을 사용하여 모든 점들을 z = 0으로 투영
    # 간단한 교차 문제 해결
    tp = -np.ones(n - 1) / (vertr[:, 2] + 1)
    xp = vertr[:, 0] * tp
    yp = vertr[:, 1] * tp
    
    ## 단계 2: 투영된 점들을 삼각화
    if len(xp) < 3:
        raise ValueError("Not enough points for triangulation after projection")
    
    try:
        faces = Delaunay(np.column_stack([xp, yp])).simplices
    except Exception as e:
        raise ValueError(f"Delaunay triangulation failed: {e}")
    
    # 모든 표면 법선이 바깥쪽을 가리키도록 함
    vertp = np.column_stack([xp, yp, np.zeros(n - 1)])
    
    u = vertp[faces[:, 0], :] - vertp[faces[:, 1], :]
    v = vertp[faces[:, 2], :] - vertp[faces[:, 1], :]
    w = np.cross(u, v)
    
    # z 성분이 양수인 면들의 인덱스
    index = w[:, 2] > 0
    # 해당 면들을 뒤집기
    faces[index, :] = np.fliplr(faces[index, :])
    
    ## 단계 3: 투영 중심을 볼록껍질 꼭짓점에 연결하여 
    #          마지막으로 누락된 삼각형들을 삼각화
    try:
        hull = ConvexHull(np.column_stack([xp, yp]))
        hindex = hull.vertices
    except Exception:
        # ConvexHull이 실패하면 모든 경계 점들을 사용
        # 간단한 경계 검출 방법
        from matplotlib.path import Path
        points = np.column_stack([xp, yp])
        
        # 외곽 점들 찾기 (간단한 방법)
        center_2d = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center_2d[1], points[:, 0] - center_2d[0])
        sorted_indices = np.argsort(angles)
        
        # 볼록껍질 근사
        hindex = sorted_indices
    
    hlen = len(hindex)
    
    # MATLAB의 인덱싱을 Python으로 변환 (0-기반 인덱싱)
    # faces에 1을 더해서 첫 번째 점(인덱스 0)을 고려
    faces = faces + 1
    
    # 볼록껍질 삼각형들 추가
    hull_faces = []
    for i in range(hlen):
        # hindex[(i+1) % hlen] + 1, hindex[i] + 1, 1 (0-기반에서는 0)
        next_idx = (i + 1) % hlen
        hull_faces.append([hindex[next_idx] + 1, hindex[i] + 1, 0])
    
    # 최종 면들 결합
    if hull_faces:
        faces = np.vstack([faces, np.array(hull_faces)])
    
    return faces