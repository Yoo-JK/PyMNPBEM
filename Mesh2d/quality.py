import numpy as np


def quality(p, t):
    """
    QUALITY: Approximate triangle quality.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates, [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Triangles as indices, [n11,n12,n13; n21,n22,n23; etc]
        
    Returns:
    --------
    q : ndarray, shape (M,)
        Triangle qualities. 0 <= q <= 1.
        Higher values indicate better quality triangles.
        
    Notes:
    ------
    This function computes an approximate measure of triangle quality based on
    the ratio of triangle area to the sum of squared edge lengths. The constant
    3.4641 (≈ 2*sqrt(3)) normalizes the quality measure so that an equilateral
    triangle has quality = 1.
    
    The quality measure is defined as:
    q = (2*sqrt(3) * Area) / (sum of squared edge lengths)
    
    For an equilateral triangle: q = 1
    For degenerate triangles: q → 0
    
    Darren Engwirda - 2007
    Python conversion - 2025
    """
    
    # Convert to numpy arrays if necessary
    p = np.asarray(p)
    t = np.asarray(t)
    
    # Input validation
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("p must be an Nx2 array of coordinates")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("t must be an Mx3 array of triangle indices")
    if np.any(t < 0) or np.any(t >= p.shape[0]):
        raise ValueError("Triangle indices in t are out of bounds")
    
    # Extract triangle vertices
    p1 = p[t[:, 0], :]  # First vertex of each triangle
    p2 = p[t[:, 1], :]  # Second vertex of each triangle  
    p3 = p[t[:, 2], :]  # Third vertex of each triangle
    
    # Calculate edge vectors
    d12 = p2 - p1  # Edge from vertex 1 to vertex 2
    d13 = p3 - p1  # Edge from vertex 1 to vertex 3
    d23 = p3 - p2  # Edge from vertex 2 to vertex 3
    
    # Calculate triangle areas using cross product
    # Area = 0.5 * |cross_product(d12, d13)|
    # For 2D vectors: cross_product = d12.x * d13.y - d12.y * d13.x
    areas = np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0])
    
    # Calculate sum of squared edge lengths
    edge_lengths_sq = np.sum(d12**2 + d13**2 + d23**2, axis=1)
    
    # Compute quality metric
    # The constant 3.4641 ≈ 2*sqrt(3) normalizes so that equilateral triangles have q=1
    # This comes from the isoperimetric inequality for triangles
    q = 3.4641 * areas / edge_lengths_sq
    
    # Handle degenerate cases where edge_lengths_sq = 0
    # (This shouldn't happen with valid triangles, but we'll be safe)
    q = np.where(edge_lengths_sq > 0, q, 0.0)
    
    # Ensure quality is bounded between 0 and 1
    # (Due to numerical precision, we might get values slightly > 1)
    q = np.clip(q, 0.0, 1.0)
    
    return q


def quality_detailed(p, t):
    """
    Extended version that returns additional triangle metrics.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates
    t : ndarray, shape (M, 3)
        Triangle indices
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'quality': Triangle quality measures (same as quality() function)
        - 'area': Triangle areas
        - 'perimeter': Triangle perimeters
        - 'aspect_ratio': Aspect ratios (longest edge / shortest edge)
        - 'min_angle': Minimum angles in degrees
        - 'max_angle': Maximum angles in degrees
    """
    
    p = np.asarray(p)
    t = np.asarray(t)
    
    # Get vertices
    p1, p2, p3 = p[t[:, 0], :], p[t[:, 1], :], p[t[:, 2], :]
    
    # Edge vectors and lengths
    d12, d13, d23 = p2 - p1, p3 - p1, p3 - p2
    L12 = np.sqrt(np.sum(d12**2, axis=1))
    L13 = np.sqrt(np.sum(d13**2, axis=1))
    L23 = np.sqrt(np.sum(d23**2, axis=1))
    
    # Areas
    areas = 0.5 * np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0])
    
    # Perimeters
    perimeters = L12 + L13 + L23
    
    # Quality (same as main function)
    edge_lengths_sq = L12**2 + L13**2 + L23**2
    quality_vals = np.where(edge_lengths_sq > 0, 
                           3.4641 * areas / edge_lengths_sq, 0.0)
    quality_vals = np.clip(quality_vals, 0.0, 1.0)
    
    # Aspect ratios
    max_edge = np.maximum.reduce([L12, L13, L23])
    min_edge = np.minimum.reduce([L12, L13, L23])
    aspect_ratios = np.where(min_edge > 0, max_edge / min_edge, np.inf)
    
    # Angles using law of cosines
    # Angle at vertex 1: opposite edge is L23
    cos_angle1 = (L12**2 + L13**2 - L23**2) / (2 * L12 * L13)
    # Angle at vertex 2: opposite edge is L13  
    cos_angle2 = (L12**2 + L23**2 - L13**2) / (2 * L12 * L23)
    # Angle at vertex 3: opposite edge is L12
    cos_angle3 = (L13**2 + L23**2 - L12**2) / (2 * L13 * L23)
    
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle1 = np.clip(cos_angle1, -1.0, 1.0)
    cos_angle2 = np.clip(cos_angle2, -1.0, 1.0) 
    cos_angle3 = np.clip(cos_angle3, -1.0, 1.0)
    
    # Convert to degrees
    angle1 = np.rad2deg(np.arccos(cos_angle1))
    angle2 = np.rad2deg(np.arccos(cos_angle2))
    angle3 = np.rad2deg(np.arccos(cos_angle3))
    
    min_angles = np.minimum.reduce([angle1, angle2, angle3])
    max_angles = np.maximum.reduce([angle1, angle2, angle3])
    
    return {
        'quality': quality_vals,
        'area': areas,
        'perimeter': perimeters,
        'aspect_ratio': aspect_ratios,
        'min_angle': min_angles,
        'max_angle': max_angles
    }


def quality_histogram(p, t, bins=20):
    """
    Generate a histogram of triangle qualities.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates
    t : ndarray, shape (M, 3)
        Triangle indices
    bins : int, optional
        Number of histogram bins (default: 20)
        
    Returns:
    --------
    hist : ndarray
        Histogram counts
    bin_edges : ndarray
        Bin edge values
    stats : dict
        Statistics including mean, min, max, std of quality values
    """
    
    q = quality(p, t)
    
    hist, bin_edges = np.histogram(q, bins=bins, range=(0.0, 1.0))
    
    stats = {
        'mean': np.mean(q),
        'median': np.median(q),
        'min': np.min(q),
        'max': np.max(q),
        'std': np.std(q),
        'num_triangles': len(q),
        'poor_quality_count': np.sum(q < 0.3),  # Triangles with quality < 0.3
        'good_quality_count': np.sum(q > 0.7)   # Triangles with quality > 0.7
    }
    
    return hist, bin_edges, stats