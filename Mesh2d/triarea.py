import numpy as np


def triarea(p, t):
    """
    TRIAREA: Area of triangles assuming counter-clockwise (CCW) node ordering.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        XY node coordinates
    t : ndarray, shape (M, 3)
        Triangles as indices into p
        
    Returns:
    --------
    A : ndarray, shape (M,)
        Triangle areas (signed)
        
    Notes:
    ------
    This function computes the signed area of triangles using the cross product.
    For counter-clockwise ordered vertices, the area is positive.
    For clockwise ordered vertices, the area is negative.
    
    The area is calculated as:
    A = (v2 - v1) × (v3 - v1) = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    
    where v1, v2, v3 are the three vertices of the triangle.
    
    Darren Engwirda - 2007
    Python conversion - 2025
    """
    
    # Convert inputs to numpy arrays
    p = np.asarray(p)
    t = np.asarray(t, dtype=int)
    
    # Input validation
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("p must be an Nx2 array of coordinates")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("t must be an Mx3 array of triangle indices")
    if np.any(t < 0) or np.any(t >= p.shape[0]):
        raise ValueError("Triangle indices in t are out of bounds")
    
    # Calculate edge vectors from vertex 1 to vertices 2 and 3
    d12 = p[t[:, 1], :] - p[t[:, 0], :]  # Edge from vertex 1 to vertex 2
    d13 = p[t[:, 2], :] - p[t[:, 0], :]  # Edge from vertex 1 to vertex 3
    
    # Calculate signed area using cross product
    # Cross product in 2D: (a × b) = a.x * b.y - a.y * b.x
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]
    
    return A


def triarea_abs(p, t):
    """
    Calculate absolute (unsigned) triangle areas.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        XY node coordinates
    t : ndarray, shape (M, 3)
        Triangle indices
        
    Returns:
    --------
    A : ndarray, shape (M,)
        Absolute triangle areas
    """
    return np.abs(triarea(p, t))


def triarea_with_validation(p, t, check_orientation=True):
    """
    Calculate triangle areas with additional validation and orientation checking.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        XY node coordinates
    t : ndarray, shape (M, 3)
        Triangle indices
    check_orientation : bool, optional
        If True, warn about clockwise triangles (default: True)
        
    Returns:
    --------
    A : ndarray, shape (M,)
        Triangle areas (signed)
    stats : dict
        Dictionary containing area statistics and orientation info
    """
    
    # Calculate signed areas
    A = triarea(p, t)
    
    # Statistics
    positive_areas = A > 0
    negative_areas = A < 0
    zero_areas = np.abs(A) < np.finfo(float).eps * 10
    
    stats = {
        'total_triangles': len(A),
        'ccw_triangles': np.sum(positive_areas),
        'cw_triangles': np.sum(negative_areas),
        'degenerate_triangles': np.sum(zero_areas),
        'min_area': np.min(np.abs(A[~zero_areas])) if not np.all(zero_areas) else 0.0,
        'max_area': np.max(np.abs(A)),
        'mean_area': np.mean(np.abs(A)),
        'total_area': np.sum(np.abs(A))
    }
    
    # Orientation warnings
    if check_orientation:
        if stats['cw_triangles'] > 0:
            print(f"Warning: {stats['cw_triangles']} triangles have clockwise orientation (negative area)")
        if stats['degenerate_triangles'] > 0:
            print(f"Warning: {stats['degenerate_triangles']} degenerate triangles found (zero area)")
    
    return A, stats


def triangle_centroids(p, t):
    """
    Calculate triangle centroids along with areas.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        XY node coordinates
    t : ndarray, shape (M, 3)
        Triangle indices
        
    Returns:
    --------
    centroids : ndarray, shape (M, 2)
        Triangle centroids
    areas : ndarray, shape (M,)
        Triangle areas (absolute)
    """
    
    # Calculate centroids as average of vertices
    centroids = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
    
    # Calculate areas
    areas = triarea_abs(p, t)
    
    return centroids, areas


def mesh_area_stats(p, t):
    """
    Comprehensive area statistics for a triangular mesh.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Node coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity
        
    Returns:
    --------
    stats : dict
        Comprehensive area statistics
    """
    
    areas = triarea_abs(p, t)
    
    # Basic statistics
    stats = {
        'num_triangles': len(areas),
        'total_area': np.sum(areas),
        'mean_area': np.mean(areas),
        'median_area': np.median(areas),
        'std_area': np.std(areas),
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'area_ratio': np.max(areas) / np.min(areas) if np.min(areas) > 0 else np.inf
    }
    
    # Percentiles
    percentiles = [10, 25, 75, 90, 95, 99]
    for p_val in percentiles:
        stats[f'area_p{p_val}'] = np.percentile(areas, p_val)
    
    # Area distribution categories
    mean_area = stats['mean_area']
    stats['small_triangles'] = np.sum(areas < 0.1 * mean_area)  # < 10% of mean
    stats['large_triangles'] = np.sum(areas > 10.0 * mean_area)  # > 10x mean
    stats['normal_triangles'] = len(areas) - stats['small_triangles'] - stats['large_triangles']
    
    return stats


def orient_triangles_ccw(p, t):
    """
    Ensure all triangles have counter-clockwise orientation.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Node coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity
        
    Returns:
    --------
    t_oriented : ndarray, shape (M, 3)
        Triangle connectivity with CCW orientation
    flipped : ndarray, shape (M,)
        Boolean array indicating which triangles were flipped
    """
    
    # Calculate signed areas
    areas = triarea(p, t)
    
    # Find triangles with clockwise orientation (negative area)
    cw_triangles = areas < 0
    
    # Create copy of triangle array
    t_oriented = t.copy()
    
    # Flip clockwise triangles by swapping vertices 2 and 3
    t_oriented[cw_triangles, [1, 2]] = t_oriented[cw_triangles, [2, 1]]
    
    return t_oriented, cw_triangles


def triangle_aspect_ratios(p, t):
    """
    Calculate aspect ratios for triangles.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Node coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity
        
    Returns:
    --------
    aspect_ratios : ndarray, shape (M,)
        Triangle aspect ratios (longest edge / shortest edge)
    areas : ndarray, shape (M,)
        Triangle areas
    """
    
    # Calculate edge lengths
    edges = np.array([
        np.sqrt(np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1)),  # Edge 1-2
        np.sqrt(np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1)),  # Edge 2-3
        np.sqrt(np.sum((p[t[:, 0], :] - p[t[:, 2], :])**2, axis=1))   # Edge 3-1
    ]).T
    
    # Calculate aspect ratios
    min_edges = np.min(edges, axis=1)
    max_edges = np.max(edges, axis=1)
    
    # Avoid division by zero
    aspect_ratios = np.where(min_edges > 0, max_edges / min_edges, np.inf)
    
    # Calculate areas
    areas = triarea_abs(p, t)
    
    return aspect_ratios, areas


# Example usage and testing
def triarea_example():
    """
    Example demonstrating triangle area calculation.
    """
    
    # Create test triangles
    p = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [0.0, 1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [0.5, 0.5]   # Node 4
    ])
    
    t = np.array([
        [0, 1, 2],  # Right triangle, area = 0.5, CCW
        [1, 3, 2],  # Right triangle, area = 0.5, CW (negative)
        [0, 2, 4],  # Triangle with area = 0.25
        [1, 4, 3]   # Triangle with area = 0.25
    ])
    
    print("Triangle area calculation example:")
    print("Nodes:")
    for i, node in enumerate(p):
        print(f"  Node {i}: ({node[0]:.1f}, {node[1]:.1f})")
    
    print("\nTriangles:")
    for i, tri in enumerate(t):
        print(f"  Triangle {i}: nodes {tri}")
    
    # Calculate areas
    areas = triarea(p, t)
    abs_areas = triarea_abs(p, t)
    
    print("\nAreas (signed):", areas)
    print("Areas (absolute):", abs_areas)
    
    # Detailed analysis
    areas_detailed, stats = triarea_with_validation(p, t)
    print("\nMesh statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return p, t, areas


if __name__ == "__main__":
    # Run example
    triarea_example()