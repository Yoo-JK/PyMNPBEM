import numpy as np
from scipy.sparse import csr_matrix
import warnings


def smoothmesh(p, t, maxit=None, tol=None):
    """
    SMOOTHMESH: Smooth a triangular mesh using Laplacian smoothing.
    
    Laplacian smoothing is an iterative process that generally leads to an
    improvement in the quality of the elements in a triangular mesh.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates, [x1,y1; x2,y2; etc]
    t : ndarray, shape (M, 3)
        Triangles as indices, [n11,n12,n13; n21,n22,n23; etc]
    maxit : int, optional
        Maximum allowable iterations (default: 20)
    tol : float, optional
        Convergence tolerance. Percentage change in edge length must be
        less than tol (default: 0.01)
        
    Returns:
    --------
    p : ndarray, shape (N, 2)
        Smoothed nodal coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity (unchanged)
        
    Notes:
    ------
    The algorithm iteratively moves each interior node to the centroid of its
    neighboring nodes. Boundary nodes are kept fixed to preserve the domain
    boundary. The process continues until convergence (small change in edge
    lengths) or maximum iterations are reached.
    
    Example:
    --------
    [p, t] = smoothmesh(p, t, 10, 0.05)
    
    See also: mesh2d, refine
    
    Darren Engwirda - 2007
    Python conversion - 2025
    """
    
    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    
    # Input validation
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("p must be an Nx2 array")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("t must be an Mx3 array")
    
    # Set default parameters
    if tol is None:
        tol = 0.01
    if maxit is None:
        maxit = 20
    
    # Ensure consistent mesh
    p, t = fixmesh(p, t)
    
    n = p.shape[0]
    
    # Build sparse connectivity matrix
    # Each triangle contributes 6 connections: (n1,n2), (n1,n3), (n2,n1), (n2,n3), (n3,n1), (n3,n2)
    row_indices = np.concatenate([
        t[:, 0], t[:, 0], t[:, 1], t[:, 1], t[:, 2], t[:, 2]
    ])
    col_indices = np.concatenate([
        t[:, 1], t[:, 2], t[:, 0], t[:, 2], t[:, 0], t[:, 1]
    ])
    data = np.ones(len(row_indices))
    
    S = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    # Count number of connections per node
    W = np.array(S.sum(axis=1)).flatten()
    
    # Check for hanging nodes (nodes with no connections)
    if np.any(W == 0):
        raise ValueError("Invalid mesh. Hanging nodes found.")
    
    # Find boundary nodes
    edge = np.vstack([
        t[:, [0, 1]],  # edges 0-1
        t[:, [0, 2]],  # edges 0-2
        t[:, [1, 2]]   # edges 1-2
    ])
    
    # Sort edges so shared edges are adjacent
    edge_sorted = np.sort(edge, axis=1)
    sort_indices = np.lexsort((edge_sorted[:, 1], edge_sorted[:, 0]))
    edge_sorted = edge_sorted[sort_indices, :]
    
    # Find shared edges
    edge_diff = np.diff(edge_sorted, axis=0)
    is_shared = np.all(edge_diff == 0, axis=1)
    
    # Extend shared edge indicator
    shared_mask = np.concatenate([[False], is_shared]) | np.concatenate([is_shared, [False]])
    
    # Separate boundary and internal edges
    boundary_edges = edge_sorted[~shared_mask, :]
    internal_edges = edge_sorted[shared_mask, :]
    
    # Get unique edges (boundary edges + one copy of each internal edge)
    if len(internal_edges) > 0:
        unique_internal = internal_edges[::2, :]  # Take every other internal edge
        unique_edges = np.vstack([boundary_edges, unique_internal])
    else:
        unique_edges = boundary_edges
    
    # Find boundary nodes
    boundary_nodes = np.unique(boundary_edges.ravel()) if len(boundary_edges) > 0 else np.array([])
    
    # Calculate initial edge lengths
    edge_vectors = p[unique_edges[:, 0], :] - p[unique_edges[:, 1], :]
    L = np.maximum(np.sqrt(np.sum(edge_vectors**2, axis=1)), np.finfo(float).eps)
    
    # Laplacian smoothing iterations
    for iter_count in range(1, maxit + 1):
        # Compute new positions as weighted average of neighbors
        # p_new = S * p / W (for each coordinate)
        p_new = np.column_stack([
            S.dot(p[:, 0]) / W,
            S.dot(p[:, 1]) / W
        ])
        
        # Keep boundary nodes fixed
        if len(boundary_nodes) > 0:
            p_new[boundary_nodes, :] = p[boundary_nodes, :]
        
        # Update positions
        p = p_new
        
        # Calculate new edge lengths
        edge_vectors_new = p[unique_edges[:, 0], :] - p[unique_edges[:, 1], :]
        L_new = np.maximum(np.sqrt(np.sum(edge_vectors_new**2, axis=1)), np.finfo(float).eps)
        
        # Check convergence - percentage change in edge lengths
        relative_change = np.abs((L_new - L) / L_new)
        move = np.max(relative_change)
        
        if move < tol:
            break
        
        L = L_new
    
    # Check if maximum iterations reached
    if iter_count == maxit:
        warnings.warn("Maximum number of iterations reached, solution did not converge!")
    
    return p, t


def fixmesh(p, t):
    """
    Fix mesh by removing duplicate nodes and degenerate triangles.
    
    This is a simplified implementation. A full version would include
    more comprehensive mesh repair operations.
    """
    # Remove duplicate points
    p, unique_idx, inverse_idx = np.unique(p, axis=0, return_index=True, return_inverse=True)
    
    # Update triangle indices
    t = inverse_idx[t]
    
    # Remove degenerate triangles (triangles with repeated vertices)
    valid_triangles = ~(
        (t[:, 0] == t[:, 1]) |
        (t[:, 1] == t[:, 2]) |
        (t[:, 2] == t[:, 0])
    )
    t = t[valid_triangles, :]
    
    return p, t


def smoothmesh_advanced(p, t, method='laplacian', maxit=20, tol=0.01, boundary_smooth=False):
    """
    Advanced mesh smoothing with multiple smoothing methods.
    
    Parameters:
    -----------
    p : ndarray, shape (N, 2)
        Node coordinates
    t : ndarray, shape (M, 3)
        Triangle connectivity
    method : str
        Smoothing method: 'laplacian', 'taubin', 'weighted_laplacian'
    maxit : int
        Maximum iterations
    tol : float
        Convergence tolerance
    boundary_smooth : bool
        Whether to allow boundary nodes to move along boundary edges
        
    Returns:
    --------
    p : ndarray
        Smoothed coordinates
    t : ndarray
        Triangle connectivity
    """
    
    if method == 'laplacian':
        return smoothmesh(p, t, maxit, tol)
    elif method == 'taubin':
        return taubin_smooth(p, t, maxit, tol)
    elif method == 'weighted_laplacian':
        return weighted_laplacian_smooth(p, t, maxit, tol)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def taubin_smooth(p, t, maxit=20, tol=0.01, lambda_param=0.5, mu_param=-0.53):
    """
    Taubin smoothing - reduces shrinkage compared to standard Laplacian.
    
    Alternates between expansion (lambda) and contraction (mu) steps.
    """
    p, t = fixmesh(p, t)
    n = p.shape[0]
    
    # Build connectivity matrix
    row_indices = np.concatenate([t[:, 0], t[:, 0], t[:, 1], t[:, 1], t[:, 2], t[:, 2]])
    col_indices = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 2], t[:, 0], t[:, 1]])
    data = np.ones(len(row_indices))
    S = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    W = np.array(S.sum(axis=1)).flatten()
    
    # Find boundary nodes (simplified)
    edge = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
    boundary_nodes = find_boundary_nodes(edge)
    
    for iter_count in range(maxit):
        p_old = p.copy()
        
        # Lambda step (expansion)
        laplacian = np.column_stack([S.dot(p[:, 0]) / W - p[:, 0],
                                   S.dot(p[:, 1]) / W - p[:, 1]])
        p += lambda_param * laplacian
        p[boundary_nodes, :] = p_old[boundary_nodes, :]  # Fix boundary
        
        # Mu step (contraction)
        laplacian = np.column_stack([S.dot(p[:, 0]) / W - p[:, 0],
                                   S.dot(p[:, 1]) / W - p[:, 1]])
        p += mu_param * laplacian
        p[boundary_nodes, :] = p_old[boundary_nodes, :]  # Fix boundary
        
        # Check convergence
        move = np.max(np.sqrt(np.sum((p - p_old)**2, axis=1)))
        if move < tol:
            break
    
    return p, t


def weighted_laplacian_smooth(p, t, maxit=20, tol=0.01):
    """
    Weighted Laplacian smoothing using cotangent weights.
    """
    # This would implement cotangent weighting for better quality
    # For now, fall back to standard Laplacian
    return smoothmesh(p, t, maxit, tol)


def find_boundary_nodes(edges):
    """
    Find boundary nodes from edge list.
    """
    edge_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edge_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1, :]
    return np.unique(boundary_edges.ravel()) if len(boundary_edges) > 0 else np.array([])


def mesh_quality_stats(p, t):
    """
    Compute quality statistics for the mesh.
    """
    from quality import quality  # Assuming quality function is available
    
    q = quality(p, t)
    
    stats = {
        'mean_quality': np.mean(q),
        'min_quality': np.min(q),
        'max_quality': np.max(q),
        'std_quality': np.std(q),
        'poor_triangles': np.sum(q < 0.3),
        'good_triangles': np.sum(q > 0.7),
        'total_triangles': len(q)
    }
    
    return stats


# Example usage
def smoothmesh_example():
    """
    Example of mesh smoothing usage.
    """
    # Create a simple mesh
    p = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]
    ])
    
    t = np.array([
        [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]
    ])
    
    # Add some noise to interior nodes
    interior_nodes = [1, 3, 4, 5, 7]  # Not on boundary
    np.random.seed(42)
    p[interior_nodes, :] += 0.1 * np.random.randn(len(interior_nodes), 2)
    
    print("Before smoothing:")
    print(f"Mesh has {p.shape[0]} nodes and {t.shape[0]} triangles")
    
    # Smooth the mesh
    p_smooth, t_smooth = smoothmesh(p, t, maxit=10, tol=0.01)
    
    print("After smoothing:")
    print(f"Mesh has {p_smooth.shape[0]} nodes and {t_smooth.shape[0]} triangles")
    
    return p, t, p_smooth, t_smooth