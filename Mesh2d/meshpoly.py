import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import warnings

def meshpoly(node, edge, qtree, p, options):
    """
    MESHPOLY: Core meshing routine called by mesh2d and meshfaces.
    
    Do not call this routine directly, use mesh2d or meshfaces instead!
    
    Parameters:
    -----------
    node : ndarray, shape (N, 2)
        Array of geometry XY co-ordinates
    edge : ndarray, shape (M, 2)
        Array of connections between NODE, defining geometry edges
    qtree : object
        Quadtree data structure, defining background mesh and element size function
    p : ndarray, shape (Q, 2)
        Array of potential boundary nodes
    options : object
        Meshing options data structure
        
    Returns:
    --------
    p : ndarray, shape (N, 2)
        Array of triangle nodes
    t : ndarray, shape (M, 3)
        Array of triangles as indices into P
    
    Notes:
    ------
    Mesh2d is a delaunay based algorithm with a "Laplacian-like" smoothing
    operation built into the mesh generation process.
    
    An unbalanced quadtree decomposition is used to evaluate the element size
    distribution required to resolve the geometry. The quadtree is
    triangulated and used as a background mesh to store the element size data.
    
    The main method attempts to optimise the node location and mesh topology
    through an iterative process. In each step a constrained delaunay
    triangulation is generated with a series of "Laplacian-like" smoothing
    operations used to improve triangle quality.
    
    Based on:
    [1] P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB.
        SIAM Review, Volume 46 (2), pp. 329-345, June 2004
    
    Darren Engwirda : 2005-09
    Python conversion: 2025
    """
    
    # Parameters
    shortedge = 0.75
    longedge = 1.5
    smalltri = 0.25
    largetri = 4.0
    qlimit = 0.5
    dt = 0.2
    
    # Statistics structure
    stats = {
        't_init': 0.0, 't_tri': 0.0, 't_inpoly': 0.0, 't_edge': 0.0,
        't_sparse': 0.0, 't_search': 0.0, 't_smooth': 0.0, 't_density': 0.0,
        'n_tri': 0
    }
    
    # Initialise mesh
    tic = time.time()
    if options.output:
        print('Initialising Mesh')
    
    p, fix, tndx = initmesh(p, qtree.p, qtree.t, qtree.h, node, edge)
    stats['t_init'] = time.time() - tic
    
    # Main loop
    if options.output:
        print('Iteration   Convergence (%)')
    
    for iter_num in range(1, options.maxit + 1):
        # Ensure unique node list
        p, unique_idx = np.unique(p, axis=0, return_inverse=True)
        fix = unique_idx[fix]
        tndx = tndx[np.unique(np.arange(len(tndx)), return_index=True)[1]]
        
        # Constrained Delaunay triangulation
        tic = time.time()
        p, t = cdt(p, node, edge)
        stats['n_tri'] += 1
        stats['t_tri'] += time.time() - tic
        
        # Get unique edges
        tic = time.time()
        e = getedges(t, p.shape[0])
        stats['t_edge'] += time.time() - tic
        
        # Sparse node-to-edge connectivity matrix
        tic = time.time()
        nume = e.shape[0]
        row_idx = np.concatenate([e.ravel(), e.ravel()])
        col_idx = np.concatenate([np.arange(nume), np.arange(nume)])
        data = np.concatenate([np.ones(nume), -np.ones(nume)])
        S = csr_matrix((data, (row_idx, col_idx)), shape=(p.shape[0], nume))
        stats['t_sparse'] += time.time() - tic
        
        # Find enclosing triangle and interpolate size function
        tic = time.time()
        tndx = mytsearch(qtree.p[:, 0], qtree.p[:, 1], qtree.t, 
                        p[:, 0], p[:, 1], tndx)
        hn = tinterp(qtree.p, qtree.t, qtree.h, p, tndx)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])
        stats['t_search'] += time.time() - tic
        
        # Calculate edge vectors and lengths
        edgev = p[e[:, 0], :] - p[e[:, 1], :]
        L = np.maximum(np.sqrt(np.sum(edgev**2, axis=1)), np.finfo(float).eps)
        
        # Inner smoothing sub-iterations
        time_smooth = time.time()
        move = 1.0
        done = False
        
        for subiter in range(iter_num - 1):
            moveold = move
            
            # Spring based smoothing
            L0 = h * np.sqrt(np.sum(L**2) / np.sum(h**2))
            F = np.maximum(L0 / L - 1.0, -0.1)
            F_vec = S.dot(edgev * F[:, np.newaxis])
            F_vec[fix, :] = 0.0
            p = p + dt * F_vec
            
            # Measure convergence
            edgev = p[e[:, 0], :] - p[e[:, 1], :]
            L0 = np.maximum(np.sqrt(np.sum(edgev**2, axis=1)), np.finfo(float).eps)
            move = np.max(np.abs((L0 - L) / L))
            L = L0
            
            if move < options.mlim:
                done = True
                break
        
        stats['t_smooth'] += time.time() - time_smooth
        
        if options.output:
            convergence = 100.0 * min(1.0, options.mlim / max(move, np.finfo(float).eps))
            print(f'{iter_num:2d}           {convergence:2.1f}')
        
        # Final triangulation for this iteration
        tic = time.time()
        p, t = cdt(p, node, edge)
        stats['n_tri'] += 1
        stats['t_tri'] += time.time() - tic
        
        # Get edges and calculate ratios
        tic = time.time()
        e = getedges(t, p.shape[0])
        stats['t_edge'] += time.time() - tic
        
        edgev = p[e[:, 0], :] - p[e[:, 1], :]
        L = np.maximum(np.sqrt(np.sum(edgev**2, axis=1)), np.finfo(float).eps)
        
        tic = time.time()
        tndx = mytsearch(qtree.p[:, 0], qtree.p[:, 1], qtree.t,
                        p[:, 0], p[:, 1], tndx)
        hn = tinterp(qtree.p, qtree.t, qtree.h, p, tndx)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])
        stats['t_search'] += time.time() - tic
        
        r = L / h
        
        # Check main loop convergence
        if done and np.max(r) < 3.0:
            break
        else:
            if iter_num == options.maxit:
                warnings.warn('Maximum number of iterations reached. Solution did not converge!')
        
        # Nodal density control
        tic = time.time()
        if iter_num < options.maxit:
            # Estimate required triangle area from size function
            Ah = 0.5 * tricentre(t, hn)**2
            
            # Remove nodes
            tri_areas = np.abs(triarea(p, t))
            i = np.where(tri_areas < smalltri * Ah)[0]  # Small area triangles
            k = np.where(np.sum(np.abs(S.toarray()), axis=1) < 2)[0]  # Nodes with <2 connections
            j = np.where(r < shortedge)[0]  # Short edges
            
            if len(j) > 0 or len(k) > 0 or len(i) > 0:
                prob = np.zeros(p.shape[0], dtype=bool)
                if len(j) > 0:
                    prob[e[j, :].ravel()] = True
                if len(i) > 0:
                    prob[t[i, :].ravel()] = True
                if len(k) > 0:
                    prob[k] = True
                prob[fix] = False  # Don't remove fixed nodes
                
                pnew = p[~prob, :]
                tndx = tndx[~prob]
                
                # Re-index fix array
                j_reindex = np.zeros(p.shape[0], dtype=int)
                j_reindex[~prob] = 1
                j_reindex = np.cumsum(j_reindex)
                fix = j_reindex[fix] - 1  # Convert to 0-based indexing
            else:
                pnew = p.copy()
            
            # Add new nodes
            large_tri = tri_areas > largetri * Ah
            r_long = longest(p, t) / tricentre(t, hn)
            k_quality = (r_long > longedge) & (quality(p, t) < qlimit)
            
            if np.any(large_tri | k_quality):
                k_idx = np.where(k_quality & ~large_tri)[0]
                i_idx = np.where(large_tri)[0]
                
                # Add new nodes at circumcentres
                combined_t = np.vstack([t[i_idx, :], t[k_idx, :]])
                cc = circumcircle(p, combined_t)
                
                # Don't add multiple points in one circumcircle
                ok = np.concatenate([np.ones(len(i_idx), dtype=bool),
                                   np.zeros(len(k_idx), dtype=bool)])
                
                for ii in range(len(i_idx), cc.shape[0]):
                    x, y = cc[ii, 0], cc[ii, 1]
                    in_circle = False
                    
                    for jj in np.where(ok)[0]:
                        dx = (x - cc[jj, 0])**2
                        if dx < cc[jj, 2] and (dx + (y - cc[jj, 1])**2) < cc[jj, 2]:
                            in_circle = True
                            break
                    
                    if not in_circle:
                        ok[ii] = True
                
                cc = cc[ok, :]
                internal_points = inpoly(cc[:, :2], node, edge)
                cc = cc[internal_points, :]
                
                # Add new nodes
                pnew = np.vstack([pnew, cc[:, :2]])
                tndx = np.concatenate([tndx, np.zeros(cc.shape[0], dtype=int)])
            
            p = pnew
        
        stats['t_density'] += time.time() - tic
    
    if hasattr(options, 'debug') and options.debug:
        print(stats)
    
    return p, t


def cdt(p, node, edge):
    """Approximate geometry-constrained Delaunay triangulation."""
    tri = Delaunay(p)
    t = tri.simplices
    
    # Impose geometry constraints - take triangles with internal centroids
    centroids = tricentre(t, p)
    i = inpoly(centroids, node, edge)
    t = t[i, :]
    
    return p, t


def initmesh(p, ph, th, hh, node, edge):
    """Initialise the mesh nodes."""
    # Find boundary nodes for current geometry
    i = findedge(p, node, edge, 1.0e-8)
    p = p[i > 0, :]
    fix = np.arange(p.shape[0])
    
    # Add internal nodes from quadtree
    inside, on_boundary = inpoly(ph, node, edge, return_on_boundary=True)
    internal_nodes = inside & ~on_boundary
    p = np.vstack([p, ph[internal_nodes, :]])
    tndx = np.zeros(p.shape[0], dtype=int)
    
    return p, fix, tndx


def getedges(t, n):
    """Get the unique edges and boundary nodes in a triangulation."""
    # Get all edges
    edges = np.vstack([
        np.sort(t[:, [0, 1]], axis=1),
        np.sort(t[:, [0, 2]], axis=1),
        np.sort(t[:, [1, 2]], axis=1)
    ])
    
    # Sort edges
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    
    # Find shared edges
    edge_diff = np.diff(edges, axis=0)
    is_shared = np.all(edge_diff == 0, axis=1)
    is_shared = np.concatenate([[False], is_shared]) | np.concatenate([is_shared, [False]])
    
    # Separate boundary and internal edges
    bnd_edges = edges[~is_shared, :]
    int_edges = edges[is_shared, :]
    
    # Take every other internal edge (since they appear twice)
    unique_int_edges = int_edges[::2, :]
    
    # Combine boundary and unique internal edges
    e = np.vstack([bnd_edges, unique_int_edges])
    
    return e


def tricentre(t, f):
    """Interpolate nodal F to the centroid of the triangles T."""
    if f.ndim == 1:
        f = f[:, np.newaxis]
    return (f[t[:, 0], :] + f[t[:, 1], :] + f[t[:, 2], :]) / 3.0


def longest(p, t):
    """Return the length of the longest edge in each triangle."""
    d1 = np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1)
    d2 = np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1)
    d3 = np.sum((p[t[:, 0], :] - p[t[:, 2], :])**2, axis=1)
    
    return np.sqrt(np.maximum.reduce([d1, d2, d3]))


# Helper functions that need to be implemented elsewhere or imported
def mytsearch(px, py, t, x, y, tndx):
    """Triangle search function - needs implementation."""
    # Placeholder - implement triangle search algorithm
    return tndx


def tinterp(p, t, h, nodes, tndx):
    """Triangle interpolation function - needs implementation."""
    # Placeholder - implement interpolation
    return np.ones(nodes.shape[0])


def inpoly(points, node, edge, return_on_boundary=False):
    """Point-in-polygon test - needs implementation."""
    # Placeholder - implement point-in-polygon algorithm
    if return_on_boundary:
        return np.ones(points.shape[0], dtype=bool), np.zeros(points.shape[0], dtype=bool)
    return np.ones(points.shape[0], dtype=bool)


def findedge(p, node, edge, tol):
    """Find edge function - needs implementation."""
    # Placeholder - implement edge finding
    return np.ones(p.shape[0])


def triarea(p, t):
    """Calculate triangle areas."""
    v1 = p[t[:, 1], :] - p[t[:, 0], :]
    v2 = p[t[:, 2], :] - p[t[:, 0], :]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def quality(p, t):
    """Calculate triangle quality - needs implementation."""
    # Placeholder - implement quality metric
    return np.ones(t.shape[0])


def circumcircle(p, t):
    """Calculate circumcircles - needs implementation."""
    # Placeholder - implement circumcircle calculation
    return np.zeros((t.shape[0], 3))