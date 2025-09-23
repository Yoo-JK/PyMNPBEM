"""
MESHFACES: 2D unstructured mesh generation for polygonal geometry.

A 2D unstructured triangular mesh is generated based on a piecewise-
linear geometry input. An arbitrary number of polygonal faces can be 
specified, and each face can contain an arbitrary number of cavities. An 
iterative method is implemented to optimise mesh quality. 

If you wish to mesh a single face, use mesh2d instead!

Darren Engwirda : 2005-09
Email           : d_engwirda@hotmail.com
Last updated    : 10/10/2009 with MATLAB 7.0 (Mesh2d v2.4)

Converted to Python by PyMNPBEM project.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import warnings


def meshfaces(node: np.ndarray, 
             edge: np.ndarray = None,
             face: List[np.ndarray] = None,
             hdata: Dict[str, Any] = None,
             options: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    2D unstructured mesh generation for polygonal geometry.
    
    Parameters
    ----------
    node : np.ndarray
        Nx2 array of nodal XY co-ordinates
    edge : np.ndarray, optional
        Mx2 array of edges as indices into node
    face : List[np.ndarray], optional  
        List of arrays containing edge numbers for each face
    hdata : dict, optional
        Element size control structure containing:
        - hmax: maximum global element size
        - edgeh: [[edge_id, size], ...] for edge-specific sizing
        - fun: user size function
        - args: arguments for size function
    options : dict, optional
        Solver options:
        - mlim: convergence tolerance (default 0.02)
        - maxit: max iterations (default 20)
        - dhmax: size function gradient limit (default 0.3)
        - output: display results (default True)
        
    Returns
    -------
    p : np.ndarray
        Nx2 array of mesh node coordinates
    t : np.ndarray
        Mx3 array of triangles (counter-clockwise)
    fnum : np.ndarray
        Mx1 array of face numbers for each triangle
    stats : dict
        Mesh statistics
    """
    ts = time.time()
    
    try:
        # Input validation and defaults
        if edge is None:
            edge = np.array([])
        if face is None:
            face = []
        if hdata is None:
            hdata = {}
        if options is None:
            options = {}
            
        # Get user options
        options = _get_options(options)
        
        # Check geometry and attempt repairs
        if options['output']:
            print('Checking Geometry')
        node, edge, face, hdata = _check_geometry(node, edge, face, hdata)
        
    except Exception as e:
        raise RuntimeError(f"Geometry initialization failed: {str(e)}")
    
    # Quadtree decomposition for background mesh
    quad_start = time.time()
    qtree = _quadtree(node, edge, hdata, options['dhmax'], options['output'])
    t_quad = time.time() - quad_start
    
    # Discretise boundary edges
    pbnd = _boundary_nodes(qtree['p'], qtree['t'], qtree['h'], 
                          node, edge, options['output'])
    
    # Mesh each face separately
    p = np.empty((0, 2))
    t = np.empty((0, 3), dtype=int)
    fnum = np.empty(0, dtype=int)
    
    for k, face_edges in enumerate(face):
        # Mesh kth polygon
        pnew, tnew = _mesh_poly(node, edge[face_edges, :], qtree, pbnd, options)
        
        # Add to global lists
        if len(t) > 0:
            tnew += len(p)
        t = np.vstack([t, tnew]) if len(t) > 0 else tnew
        p = np.vstack([p, pnew]) if len(p) > 0 else pnew
        fnum = np.concatenate([fnum, (k+1) * np.ones(len(tnew), dtype=int)])
    
    # Ensure consistent, CCW ordered triangulation
    p, t, fnum = _fix_mesh(p, t, fnum)
    
    # Element quality
    q = _quality(p, t)
    
    # Method statistics
    stats = {
        'Time': time.time() - ts,
        'Triangles': len(t),
        'Nodes': len(p),
        'Mean_quality': np.mean(q),
        'Min_quality': np.min(q)
    }
    
    if options['output']:
        _plot_mesh(p, t, fnum, face, node, edge, q, options, stats)
    
    return p, t, fnum, stats


def _boundary_nodes(ph: np.ndarray, th: np.ndarray, hh: np.ndarray,
                   node: np.ndarray, edge: np.ndarray, output: bool) -> np.ndarray:
    """
    Discretise geometry based on edge size requirements from background mesh.
    
    Parameters
    ----------
    ph : np.ndarray
        Background mesh nodes
    th : np.ndarray
        Background mesh triangles
    hh : np.ndarray
        Size function at background nodes
    node : np.ndarray
        Geometry vertices
    edge : np.ndarray
        Geometry edges
    output : bool
        Print progress messages
        
    Returns
    -------
    np.ndarray
        Discretized boundary nodes
    """
    p = node.copy()
    e = edge.copy()
    
    # Find which background triangle contains each point
    tri = Delaunay(ph)
    i = tri.find_simplex(p)
    
    # Interpolate size function to boundary points
    h = _triangle_interpolate(ph, th, hh, p, i)
    
    if output:
        print('Placing Boundary Nodes')
        
    iter_count = 1
    while True:
        # Edge lengths
        dxy = p[e[:, 1], :] - p[e[:, 0], :]
        L = np.sqrt(np.sum(dxy**2, axis=1))
        
        # Size function on edges
        he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])
        
        # Split long edges
        ratio = L / he
        split = ratio >= 1.5
        
        if np.any(split):
            # Split edges at midpoint
            n1 = e[split, 0]
            n2 = e[split, 1]
            pm = 0.5 * (p[n1, :] + p[n2, :])
            n3 = np.arange(len(pm)) + len(p)
            
            # Update edge list
            e[split, 1] = n3
            new_edges = np.column_stack([n3, n2])
            e = np.vstack([e, new_edges])
            p = np.vstack([p, pm])
            
            # Size function at new nodes
            i_new = tri.find_simplex(pm)
            h_new = _triangle_interpolate(ph, th, hh, pm, i_new)
            h = np.concatenate([h, h_new])
        else:
            break
            
        iter_count += 1
    
    # Create node-to-edge connectivity matrix
    ne = len(e)
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([np.arange(ne), np.arange(ne)])
    data = np.concatenate([-np.ones(ne), np.ones(ne)])
    S = csr_matrix((data, (row, col)), shape=(len(p), ne))
    
    # Smooth boundary nodes
    if output:
        print('Smoothing Boundaries')
    
    del_val = 0.0
    tol = 0.02
    maxit = 50
    
    for iter_num in range(maxit):
        del_old = del_val
        
        # Spring-based smoothing
        F = he / L - 1.0
        F_vec = S @ (dxy * F[:, np.newaxis])
        F_vec[:len(node), :] = 0.0  # Don't move original vertices
        p = p + 0.2 * F_vec
        
        # Check convergence
        dxy = p[e[:, 1], :] - p[e[:, 0], :]
        L_new = np.sqrt(np.sum(dxy**2, axis=1))
        del_val = np.max(np.abs((L_new - L) / L_new))
        
        if del_val < tol:
            break
        elif iter_num == maxit - 1:
            warnings.warn('Boundary smoothing did not converge')
            
        L = L_new
        
        if del_val > del_old:
            # Re-interpolate size function
            i = tri.find_simplex(p)
            h = _triangle_interpolate(ph, th, hh, p, i)
            he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])
    
    return p


def _get_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate user options."""
    defaults = {
        'mlim': 0.02,
        'maxit': 20,
        'dhmax': 0.3,
        'output': True,
        'debug': False
    }
    
    if not options:
        return defaults
    
    if not isinstance(options, dict):
        raise TypeError('OPTIONS must be a dictionary')
    
    result = defaults.copy()
    
    for key, value in options.items():
        if key not in defaults:
            raise ValueError(f'Invalid option: {key}')
        
        if key in ['mlim', 'dhmax']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f'{key} must be a positive number')
        elif key == 'maxit':
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        elif key in ['output', 'debug']:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be a boolean')
        
        result[key] = value
    
    return result


def _check_geometry(node: np.ndarray, edge: np.ndarray, 
                   face: List[np.ndarray], hdata: Dict[str, Any]) -> Tuple:
    """Check and repair geometry."""
    # Basic validation
    if not isinstance(node, np.ndarray) or node.shape[1] != 2:
        raise ValueError('NODE must be Nx2 array')
    
    if len(edge) > 0:
        if not isinstance(edge, np.ndarray) or edge.shape[1] != 2:
            raise ValueError('EDGE must be Mx2 array')
        if np.max(edge) >= len(node) or np.min(edge) < 0:
            raise ValueError('EDGE contains invalid node indices')
    
    # For now, return inputs unchanged (geometry repair is complex)
    return node, edge, face, hdata


def _quadtree(node: np.ndarray, edge: np.ndarray, hdata: Dict[str, Any],
             dhmax: float, output: bool) -> Dict[str, np.ndarray]:
    """
    Generate background quadtree mesh for size function.
    Placeholder implementation.
    """
    if output:
        print('Generating background mesh')
    
    # Simple background mesh - in practice would use quadtree
    bbox = np.array([
        [np.min(node[:, 0]), np.min(node[:, 1])],
        [np.max(node[:, 0]), np.max(node[:, 1])]
    ])
    
    # Expand bounding box
    center = np.mean(bbox, axis=0)
    size = np.max(bbox[1] - bbox[0])
    bbox = np.array([
        center - 0.6 * size,
        center + 0.6 * size
    ])
    
    # Create simple grid
    n = 20
    x = np.linspace(bbox[0, 0], bbox[1, 0], n)
    y = np.linspace(bbox[0, 1], bbox[1, 1], n)
    X, Y = np.meshgrid(x, y)
    p = np.column_stack([X.ravel(), Y.ravel()])
    
    # Simple triangulation
    tri = Delaunay(p)
    t = tri.simplices
    
    # Default size function
    h = np.full(len(p), hdata.get('hmax', 0.1))
    
    return {'p': p, 't': t, 'h': h}


def _mesh_poly(node: np.ndarray, edge: np.ndarray, qtree: Dict[str, np.ndarray],
              pbnd: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mesh a single polygon.
    Placeholder implementation.
    """
    # For now, return simple triangulation of polygon boundary
    if len(edge) == 0:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)
    
    # Get polygon vertices in order
    poly_nodes = []
    current = edge[0, 0]
    poly_nodes.append(current)
    
    for _ in range(len(edge)):
        # Find next edge
        next_edges = edge[edge[:, 0] == current, 1]
        if len(next_edges) == 0:
            next_edges = edge[edge[:, 1] == current, 0]
        
        if len(next_edges) > 0:
            current = next_edges[0]
            if current not in poly_nodes:
                poly_nodes.append(current)
    
    if len(poly_nodes) < 3:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)
    
    # Get polygon coordinates
    p = node[poly_nodes, :]
    
    # Simple triangulation
    try:
        tri = Delaunay(p)
        return p, tri.simplices
    except:
        return np.empty((0, 2)), np.empty((0, 3), dtype=int)


def _fix_mesh(p: np.ndarray, t: np.ndarray, fnum: np.ndarray) -> Tuple:
    """Ensure consistent CCW triangle orientation."""
    if len(t) == 0:
        return p, t, fnum
    
    # Check orientation of each triangle
    v1 = p[t[:, 1], :] - p[t[:, 0], :]
    v2 = p[t[:, 2], :] - p[t[:, 0], :]
    area = 0.5 * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Flip clockwise triangles
    cw = area < 0
    t[cw, [1, 2]] = t[cw, [2, 1]]
    
    return p, t, fnum


def _quality(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute triangle quality measure."""
    if len(t) == 0:
        return np.array([])
    
    # Edge lengths
    d1 = np.sqrt(np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1))
    d2 = np.sqrt(np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1))
    d3 = np.sqrt(np.sum((p[t[:, 0], :] - p[t[:, 2], :])**2, axis=1))
    
    # Semi-perimeter and area
    s = 0.5 * (d1 + d2 + d3)
    area = 0.5 * np.abs((p[t[:, 1], 0] - p[t[:, 0], 0]) * 
                       (p[t[:, 2], 1] - p[t[:, 0], 1]) -
                       (p[t[:, 2], 0] - p[t[:, 0], 0]) * 
                       (p[t[:, 1], 1] - p[t[:, 0], 1]))
    
    # Quality = 4*sqrt(3)*area / (d1^2 + d2^2 + d3^2)
    q = 4 * np.sqrt(3) * area / (d1**2 + d2**2 + d3**2)
    
    return np.nan_to_num(q, 0)


def _triangle_interpolate(ph: np.ndarray, th: np.ndarray, hh: np.ndarray,
                         p: np.ndarray, tri_indices: np.ndarray) -> np.ndarray:
    """Interpolate values at points using triangular mesh."""
    h = np.zeros(len(p))
    
    for i, ti in enumerate(tri_indices):
        if ti >= 0:
            # Barycentric interpolation
            tri = th[ti]
            A = ph[tri[1], :] - ph[tri[0], :]
            B = ph[tri[2], :] - ph[tri[0], :]
            C = p[i, :] - ph[tri[0], :]
            
            det = A[0]*B[1] - A[1]*B[0]
            if abs(det) > 1e-12:
                u = (C[0]*B[1] - C[1]*B[0]) / det
                v = (A[0]*C[1] - A[1]*C[0]) / det
                w = 1 - u - v
                h[i] = w*hh[tri[0]] + u*hh[tri[1]] + v*hh[tri[2]]
            else:
                h[i] = hh[tri[0]]
        else:
            # Point outside mesh - use nearest neighbor
            distances = np.sum((ph - p[i, :])**2, axis=1)
            nearest = np.argmin(distances)
            h[i] = hh[nearest]
    
    return h


def _plot_mesh(p: np.ndarray, t: np.ndarray, fnum: np.ndarray,
              face: List[np.ndarray], node: np.ndarray, edge: np.ndarray,
              q: np.ndarray, options: Dict[str, Any], stats: Dict[str, Any]):
    """Plot the generated mesh."""
    plt.figure(figsize=(12, 8))
    
    # Plot mesh nodes
    plt.plot(p[:, 0], p[:, 1], 'b.', markersize=1)
    
    # Color mesh by face
    colors = ['b', 'r', 'g', 'orange', 'm', 'c', 'y']
    
    for k in range(len(face)):
        face_tris = t[fnum == k+1]
        if len(face_tris) > 0:
            color = colors[k % len(colors)]
            for tri in face_tris:
                tri_coords = p[tri]
                tri_plot = plt.Polygon(tri_coords, fill=False, 
                                     edgecolor=color, linewidth=0.5)
                plt.gca().add_patch(tri_plot)
    
    # Plot original geometry
    if len(edge) > 0:
        for e in edge:
            plt.plot([node[e[0], 0], node[e[1], 0]], 
                    [node[e[0], 1], node[e[1], 1]], 'k-', linewidth=2)
    
    # Highlight low quality triangles if in debug mode
    if options['debug'] and len(q) > 0:
        low_q = q < 0.5
        if np.any(low_q):
            pc = np.mean(p[t[low_q], :], axis=1)
            plt.plot(pc[:, 0], pc[:, 1], 'r.', markersize=4)
    
    plt.axis('equal')
    plt.title('Mesh Generation Results')
    plt.grid(True, alpha=0.3)
    
    print(f"Statistics: {stats}")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Simple example with two squares
    node = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],  # Square 1
        [2, 0], [3, 0], [3, 1], [2, 1]   # Square 2
    ])
    
    edge = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Square 1 edges (0-3)
        [4, 5], [5, 6], [6, 7], [7, 4]   # Square 2 edges (4-7)
    ])
    
    face = [
        np.array([0, 1, 2, 3]),  # Face 1: edges 0-3
        np.array([4, 5, 6, 7])   # Face 2: edges 4-7
    ]
    
    # Generate mesh
    p, t, fnum, stats = meshfaces(node, edge, face)
    print(f"Generated mesh with {len(p)} nodes and {len(t)} triangles")