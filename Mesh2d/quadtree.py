import numpy as np
from scipy.spatial import ConvexHull
import warnings


def quadtree(node, edge, hdata, dhmax, output=True):
    """
    QUADTREE: 2D quadtree decomposition of polygonal geometry.
    
    The polygon is first rotated so that the minimal enclosing rectangle is
    aligned with the Cartesian XY axes. The long axis is aligned with Y. This
    ensures that the quadtree generated for a geometry input that has
    undergone arbitrary rotations in the XY plane is always the same.
    
    Parameters:
    -----------
    node : ndarray, shape (N, 2)
        [x1,y1; x2,y2; etc] geometry vertices
    edge : ndarray, shape (M, 2)
        [n11,n12; n21,n22; etc] geometry edges as connections in NODE
    hdata : dict
        User defined size function structure
    dhmax : float
        Maximum allowable relative gradient in the size function
    output : bool, optional
        Print progress messages (default: True)
        
    Returns:
    --------
    p : ndarray, shape (K, 2)
        Background mesh nodes
    t : ndarray, shape (L, 3)
        Background mesh triangles
    h : ndarray, shape (K,)
        Size function value at p
        
    Notes:
    ------
    The bounding box is recursively subdivided until the dimension of each
    box matches the local geometry feature size. The geometry feature size is
    based on the minimum distance between linear geometry segments.
    
    A size function is obtained at the quadtree vertices based on the minimum
    neighbouring box dimension at each vertex. This size function is gradient
    limited to produce a smooth function.
    
    Darren Engwirda : 2007
    Python conversion : 2025
    """
    
    # Convert inputs to numpy arrays
    node = np.asarray(node)
    edge = np.asarray(edge)
    
    # Bounding box
    XYmax = np.max(node, axis=0)
    XYmin = np.min(node, axis=0)
    
    # Rotate NODE so that the long axis of the minimum bounding rectangle is
    # aligned with the Y axis
    theta = minrectangle(node)
    node = rotate(node, theta)
    
    # Rotated XY edge endpoints
    edgexy = np.column_stack([node[edge[:, 0], :], node[edge[:, 1], :]])
    
    # LOCAL FEATURE SIZE
    if output:
        print('Estimating local geometry feature size')
    
    # Get size function data
    hmax, edgeh, fun, args = gethdata(hdata)
    
    # Insert test points along the boundaries at which the LFS can be approximated
    wm = 0.5 * (edgexy[:, [0, 1]] + edgexy[:, [2, 3]])  # Edge midpoints
    length = np.sqrt(np.sum((edgexy[:, [2, 3]] - edgexy[:, [0, 1]])**2, axis=1))  # Edge length
    L = 2.0 * dist2poly(wm, edgexy, 2.0 * length)  # Estimate the LFS
    
    # Add more points where edges are separated by less than their length
    r = 2.0 * length / L
    r = np.round((r - 2.0) / 2.0).astype(int)
    add = np.where(r > 0)[0]
    
    if len(add) > 0:
        num = 2 * np.sum(r[add])
        start = wm.shape[0]
        wm = np.vstack([wm, np.zeros((num, 2))])
        L = np.concatenate([L, np.zeros(num)])
        next_idx = start
        
        for j in range(len(add)):
            ce = add[j]
            num_pts = r[ce]
            tmp = np.arange(1, num_pts + 1) / (num_pts + 1)
            end_idx = next_idx + 2 * num_pts
            
            x1, y1 = edgexy[ce, 0], edgexy[ce, 1]
            x2, y2 = edgexy[ce, 2], edgexy[ce, 3]
            xm, ym = wm[ce, 0], wm[ce, 1]
            
            wm[next_idx:end_idx, :] = np.column_stack([
                np.concatenate([x1 + tmp * (xm - x1), xm + tmp * (x2 - xm)]),
                np.concatenate([y1 + tmp * (ym - y1), ym + tmp * (y2 - ym)])
            ])
            
            L[next_idx:end_idx] = L[ce]
            next_idx = end_idx
        
        L[start:] = dist2poly(wm[start:, :], edgexy, L[start:])
    
    # Handle boundary layer size functions
    if edgeh is not None and len(edgeh) > 0:
        for j in range(edgeh.shape[0]):
            if L[edgeh[j, 0]] >= edgeh[j, 1]:
                cw = edgeh[j, 0]
                r = int(np.ceil(2.0 * length[cw] / edgeh[j, 1] / 2.0))
                tmp = np.arange(1, r) / r
                
                x1, y1 = edgexy[cw, 0], edgexy[cw, 1]
                x2, y2 = edgexy[cw, 2], edgexy[cw, 3]
                xm, ym = wm[cw, 0], wm[cw, 1]
                
                new_points = np.column_stack([
                    np.concatenate([x1 + tmp * (xm - x1), xm + tmp * (x2 - xm)]),
                    np.concatenate([y1 + tmp * (ym - y1), ym + tmp * (y2 - ym)])
                ])
                
                wm = np.vstack([wm, new_points])
                L[cw] = edgeh[j, 1]
                L = np.concatenate([L, edgeh[j, 1] * np.ones(2 * len(tmp))])
    
    # Sort the LFS points based on y-value for faster point location
    sort_idx = np.argsort(wm[:, 1])
    wm = wm[sort_idx, :]
    L = L[sort_idx]
    nw = wm.shape[0]
    
    # UNBALANCED QUADTREE DECOMPOSITION
    if output:
        print('Quadtree decomposition')
    
    xymin = np.min(np.vstack([edgexy[:, [0, 1]], edgexy[:, [2, 3]]]), axis=0)
    xymax = np.max(np.vstack([edgexy[:, [0, 1]], edgexy[:, [2, 3]]]), axis=0)
    
    dim = 2.0 * np.max(xymax - xymin)
    xm = 0.5 * (xymin[0] + xymax[0])
    ym = 0.5 * (xymin[1] + xymax[1])
    
    # Setup boxes with consistent CCW node order
    p = np.array([
        [xm - 0.5 * dim, ym - 0.5 * dim],  # bottom left
        [xm + 0.5 * dim, ym - 0.5 * dim],  # bottom right
        [xm + 0.5 * dim, ym + 0.5 * dim],  # top right
        [xm - 0.5 * dim, ym + 0.5 * dim]   # top left
    ])
    b = np.array([[0, 1, 2, 3]])  # 0-based indexing
    
    # User defined size functions
    pr = rotate(p, -theta)
    h = userhfun(pr[:, 0], pr[:, 1], fun, args, hmax, XYmin, XYmax)
    
    pblock = 5 * nw
    bblock = pblock
    
    np_count = p.shape[0]
    nb = b.shape[0]
    test = np.ones(nb, dtype=bool)
    
    while True:
        vec = np.where(test[:nb])[0]
        if len(vec) == 0:
            break
        
        N = np_count
        for k in range(len(vec)):
            m = vec[k]
            
            # Corner nodes
            n1, n2, n3, n4 = b[m, :]
            x1, y1 = p[n1, 0], p[n1, 1]
            x2, y4 = p[n2, 0], p[n4, 1]
            
            # Binary search to find first wm with y>=ymin for current box
            start = binary_search_y(wm[:, 1], y1, nw)
            
            # Initialize LFS as the min of corner user defined size function values
            LFS = 1.5 * min(h[n1], h[n2], h[n3], h[n4])
            
            # Loop through all WM in box and take min LFS
            for i in range(start, nw):
                if wm[i, 1] <= y4:
                    if (wm[i, 0] >= x1) and (wm[i, 0] <= x2) and (L[i] < LFS):
                        LFS = L[i]
                else:
                    break
            
            # Split box into 4 if necessary
            if (x2 - x1) >= LFS:
                # Allocate memory on demand
                if (np_count + 5) >= p.shape[0]:
                    p = np.vstack([p, np.zeros((pblock, 2))])
                    pblock = 2 * pblock
                
                if (nb + 3) >= b.shape[0]:
                    b = np.vstack([b, np.zeros((bblock, 4), dtype=int)])
                    test = np.concatenate([test, np.ones(bblock, dtype=bool)])
                    bblock = 2 * bblock
                
                xm_box = x1 + 0.5 * (x2 - x1)
                ym_box = y1 + 0.5 * (y4 - y1)
                
                # New nodes
                p[np_count:np_count + 5, :] = np.array([
                    [xm_box, ym_box],  # center
                    [xm_box, y1],      # bottom mid
                    [x2, ym_box],      # right mid
                    [xm_box, y4],      # top mid
                    [x1, ym_box]       # left mid
                ])
                
                # New boxes
                b[m, :] = [n1, np_count + 1, np_count, np_count + 4]          # Box 1
                b[nb, :] = [np_count + 1, n2, np_count + 2, np_count]         # Box 2
                b[nb + 1, :] = [np_count, np_count + 2, n3, np_count + 3]     # Box 3
                b[nb + 2, :] = [np_count + 4, np_count, np_count + 3, n4]     # Box 4
                
                nb += 3
                np_count += 5
            else:
                test[m] = False
        
        # User defined size function at new nodes
        if np_count > N:
            pr = rotate(p[N:np_count, :], -theta)
            h = np.concatenate([h, userhfun(pr[:, 0], pr[:, 1], fun, args, hmax, XYmin, XYmax)])
    
    p = p[:np_count, :]
    b = b[:nb, :]
    
    # Remove duplicate nodes
    p, unique_idx, inv_idx = np.unique(p, axis=0, return_index=True, return_inverse=True)
    h = h[unique_idx]
    b = inv_idx[b]
    
    # FORM SIZE FUNCTION
    if output:
        print('Forming element size function')
    
    # Unique edges
    edges = np.vstack([
        np.sort(b[:, [0, 1]], axis=1),
        np.sort(b[:, [1, 2]], axis=1),
        np.sort(b[:, [2, 3]], axis=1),
        np.sort(b[:, [3, 0]], axis=1)
    ])
    e = np.unique(edges, axis=0)
    edge_lengths = np.sqrt(np.sum((p[e[:, 0], :] - p[e[:, 1], :])**2, axis=1))
    
    # Initial h - minimum neighbouring edge length
    for k in range(len(e)):
        Lk = edge_lengths[k]
        if Lk < h[e[k, 0]]:
            h[e[k, 0]] = Lk
        if Lk < h[e[k, 1]]:
            h[e[k, 1]] = Lk
    
    h = np.minimum(h, hmax)
    
    # Gradient limiting
    tol = 1.0e-6
    while True:
        h_old = h.copy()
        for k in range(len(e)):
            n1, n2 = e[k, :]
            Lk = edge_lengths[k]
            if h[n1] > h[n2]:
                dh = (h[n1] - h[n2]) / Lk
                if dh > dhmax:
                    h[n1] = h[n2] + dhmax * Lk
            else:
                dh = (h[n2] - h[n1]) / Lk
                if dh > dhmax:
                    h[n2] = h[n1] + dhmax * Lk
        
        if np.linalg.norm((h - h_old) / h, ord=np.inf) < tol:
            break
    
    # TRIANGULATE QUADTREE
    if output:
        print('Triangulating quadtree')
    
    if b.shape[0] == 1:
        # Split box diagonally into 2 triangles
        t = np.array([[b[0, 0], b[0, 1], b[0, 2]],
                      [b[0, 0], b[0, 2], b[0, 3]]])
    else:
        # Get node-to-node connectivity
        n2n = create_node_connectivity(e, p.shape[0])
        
        # Check for regular boxes with no mid-side nodes
        num = np.array([n2n[b[i, j], 0] for i in range(b.shape[0]) for j in range(4)]).reshape(b.shape[0], 4)
        reg = np.all(num <= 4, axis=1)
        
        # Split regular boxes diagonally into 2 triangles
        t = np.vstack([
            b[reg, [0, 1, 2]],
            b[reg, [0, 2, 3]]
        ])
        
        if not np.all(reg):
            # Triangulate irregular boxes
            irregular_boxes = b[~reg, :]
            t_irregular = triangulate_irregular_boxes(irregular_boxes, p, h, n2n)
            t = np.vstack([t, t_irregular])
    
    # Remove duplicate nodes and clean up
    good_id = np.where(h != 0)[0]
    p = p[good_id, :]
    h = h[good_id]
    
    # Reindex triangles
    reindex_map = {old_id: new_id for new_id, old_id in enumerate(good_id)}
    t_new = np.zeros_like(t)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t_new[i, j] = reindex_map.get(t[i, j], -1)
    
    # Remove triangles with invalid indices
    valid_triangles = np.all(t_new >= 0, axis=1)
    t = t_new[valid_triangles, :]
    
    # Undo rotation
    p = rotate(p, -theta)
    
    return p, t, h


# Helper functions
def binary_search_y(y_array, target, n):
    """Binary search for first element >= target."""
    if y_array[0] >= target:
        return 0
    elif y_array[n-1] < target:
        return n
    
    lower, upper = 0, n - 1
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if y_array[mid] < target:
            lower = mid
        else:
            upper = mid
    
    return upper


def create_node_connectivity(edges, num_nodes):
    """Create node-to-node connectivity matrix."""
    # First column is count, max 8 neighbors due to quadtree
    n2n = np.zeros((num_nodes, 9), dtype=int)
    
    for k in range(len(edges)):
        n1, n2 = edges[k, :]
        # Add n2 as neighbor of n1
        n2n[n1, 0] += 1
        n2n[n1, n2n[n1, 0]] = n2
        # Add n1 as neighbor of n2
        n2n[n2, 0] += 1
        n2n[n2, n2n[n2, 0]] = n1
    
    return n2n


def triangulate_irregular_boxes(boxes, p, h, n2n):
    """Triangulate irregular quadtree boxes."""
    t_list = []
    
    for box in boxes:
        n1, n2, n3, n4 = box
        
        # Assemble node list for box in CCW order
        nlist = assemble_node_list_ccw(box, p, n2n)
        nnode = len(nlist)
        
        if nnode == 4:
            # No mid-side nodes - split diagonally
            t_list.extend([[n1, n2, n3], [n1, n3, n4]])
        elif nnode == 5:
            # One mid-side node - create 3 triangles
            mid_node = find_mid_node(nlist, box)
            t_list.extend(triangulate_with_one_midnode(box, mid_node))
        else:
            # Multiple mid-side nodes - add centroid
            centroid_idx = len(p)
            p_new = np.mean(p[nlist, :], axis=0)
            h_new = np.mean(h[nlist])
            
            # Add triangles connecting to centroid
            for j in range(nnode):
                next_j = (j + 1) % nnode
                t_list.append([nlist[j], centroid_idx, nlist[next_j]])
    
    return np.array(t_list) if t_list else np.array([]).reshape(0, 3)


def assemble_node_list_ccw(box, p, n2n):
    """Assemble node list for a box in counter-clockwise order."""
    n1, n2, n3, n4 = box
    nlist = [n1]
    
    # Simplified implementation - for full version, need to traverse edges CCW
    # This is a placeholder that needs full geometric traversal logic
    return list(box)  # Simplified version


def find_mid_node(nlist, box):
    """Find the mid-side node in a 5-node list."""
    for i, node in enumerate(nlist):
        if node not in box:
            return node
    return nlist[0]  # Fallback


def triangulate_with_one_midnode(box, mid_node):
    """Create 3 triangles for box with one mid-side node."""
    n1, n2, n3, n4 = box
    # Simplified - needs proper geometric logic
    return [[n1, mid_node, n4], [mid_node, n2, n3], [n4, mid_node, n3]]


def minrectangle(p):
    """Find rotation angle for minimum bounding rectangle."""
    n = p.shape[0]
    if n <= 2:
        return 0.0
    
    # Convex hull
    hull = ConvexHull(p)
    hull_points = p[hull.vertices, :]
    
    # Check all hull edge segments
    A_old = np.inf
    theta_best = 0.0
    
    for i in range(len(hull.vertices)):
        j = (i + 1) % len(hull.vertices)
        edge = hull_points[j, :] - hull_points[i, :]
        angle = -np.arctan2(edge[1], edge[0])
        
        # Rotate and compute bounding rectangle area
        p_rot = rotate(hull_points, angle)
        dxy = np.max(p_rot, axis=0) - np.min(p_rot, axis=0)
        A = dxy[0] * dxy[1]
        
        if A < A_old:
            A_old = A
            theta_best = angle
    
    # Ensure long axis is aligned with Y
    p_rot = rotate(p, theta_best)
    dxy = np.max(p_rot, axis=0) - np.min(p_rot, axis=0)
    if dxy[0] > dxy[1]:
        theta_best += 0.5 * np.pi
    
    return theta_best


def rotate(p, theta):
    """Rotate 2D points by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, s], [-s, c]])
    return p @ rotation_matrix


def userhfun(x, y, fun, args, hmax, xymin, xymax):
    """Evaluate user defined size function."""
    if fun is not None and callable(fun):
        h = fun(x, y, *args)
        if h.shape != x.shape:
            raise ValueError('Incorrect user defined size function. SIZE(H) must equal SIZE(X).')
    else:
        h = np.full_like(x, np.inf)
    
    h = np.minimum(h, hmax)
    
    # Limit to domain
    out = ((x > xymax[0]) | (x < xymin[0]) | 
           (y > xymax[1]) | (y < xymin[1]))
    h[out] = np.inf
    
    if np.any(h <= 0.0):
        raise ValueError('Incorrect user defined size function. H must be positive.')
    
    return h


def gethdata(hdata):
    """Parse user defined size function data."""
    # Default values
    d_hmax = np.inf
    d_edgeh = None
    d_fun = None
    d_args = []
    
    if hdata is None:
        return d_hmax, d_edgeh, d_fun, d_args
    
    if not isinstance(hdata, dict):
        raise ValueError('HDATA must be a dictionary')
    
    # Extract values with validation
    hmax = hdata.get('hmax', d_hmax)
    if not np.isscalar(hmax) or hmax <= 0:
        raise ValueError('HDATA.hmax must be a positive scalar')
    
    edgeh = hdata.get('edgeh', d_edgeh)
    if edgeh is not None:
        edgeh = np.asarray(edgeh)
        if edgeh.ndim != 2 or edgeh.shape[1] != 2 or np.any(edgeh < 0):
            raise ValueError('HDATA.edgeh must be a positive Kx2 array')
    
    fun = hdata.get('fun', d_fun)
    if fun is not None and not callable(fun):
        raise ValueError('HDATA.fun must be a callable function')
    
    args = hdata.get('args', d_args)
    if not isinstance(args, (list, tuple)):
        raise ValueError('HDATA.args must be a list or tuple')
    
    return hmax, edgeh, fun, args


def dist2poly(points, edgexy, max_dist):
    """
    Calculate distance from points to polygon edges.
    
    This is a placeholder implementation that needs to be completed
    with proper distance-to-polygon calculation.
    """
    # Placeholder - implement actual distance calculation
    return np.full(points.shape[0], 1.0)