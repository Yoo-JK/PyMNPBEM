import numpy as np
from scipy.sparse import csr_matrix
import warnings


def checkgeometry(node, edge=None, face=None, hdata=None):
    """
    Check a geometry input for MESH2D.
    
    Parameters:
    -----------
    node : ndarray, shape (N, 2)
        Array of XY geometry nodes
    edge : ndarray, shape (M, 2), optional
        Array of connections between nodes in NODE
    face : list of lists, optional
        List of edges in each face
    hdata : dict, optional
        Dictionary defining size function data
        
    Returns:
    --------
    node : ndarray, shape (N', 2)
        Cleaned array of XY geometry nodes
    edge : ndarray, shape (M', 2)
        Cleaned array of connections
    face : list of lists
        Cleaned list of edges in each face
    hdata : dict
        Updated hdata dictionary
        
    The following checks are performed:
    1. Unique edges in EDGE
    2. Only nodes referenced in EDGE are kept
    3. Unique nodes in NODE
    4. No "hanging" nodes and no "T-junctions"
    
    Checks for self-intersecting geometry are NOT done because this can be 
    expensive for big inputs.
    
    HDATA and FACE may be re-indexed to maintain consistency.
    """
    
    # Input validation
    if node is None:
        raise ValueError('Insufficient inputs')
    
    node = np.asarray(node, dtype=float)
    nNode = node.shape[0]
    
    # Build edge if not passed
    if edge is None:
        edge = np.column_stack([
            np.arange(1, nNode),     # 1 to nNode-1 (MATLAB 1-based)
            np.arange(2, nNode + 1)  # 2 to nNode (MATLAB 1-based)
        ])
        edge = np.vstack([edge, [nNode, 1]])  # Add [nNode, 1]
        edge = edge - 1  # Convert to 0-based indexing
    else:
        edge = np.asarray(edge, dtype=int)
    
    # Build face if not passed
    if face is None:
        face = [list(range(1, edge.shape[0] + 1))]  # MATLAB 1-based
        face = [[x - 1 for x in face[0]]]  # Convert to 0-based
    
    # Initialize hdata if needed
    if hdata is None:
        hdata = {}
    
    # Check inputs
    if node.size != 2 * nNode:
        raise ValueError('NODE must be an Nx2 array')
    
    if edge.size != 2 * edge.shape[0]:
        raise ValueError('EDGE must be an Mx2 array')
    
    if np.max(edge) >= nNode or np.min(edge) < 0:
        raise ValueError('Invalid EDGE')
    
    for k in range(len(face)):
        if not face[k] or np.any(np.array(face[k]) < 0) or np.any(np.array(face[k]) >= edge.shape[0]):
            raise ValueError(f'Invalid FACE[{k}]')
    
    # Check if we've got size data attached to edges
    edgeh = 'edgeh' in hdata and hdata['edgeh'] is not None and len(hdata['edgeh']) > 0
    
    # Remove un-used nodes and re-index
    i = np.unique(edge.flatten())
    del_nodes = nNode - len(i)
    
    if del_nodes > 0:
        node = node[i, :]
        j = np.zeros(nNode, dtype=int)
        j[i] = 1
        j = np.cumsum(j)
        edge = j[edge]
        
        # Remove self-edges
        valid_edges = edge[:, 0] != edge[:, 1]
        j = np.zeros(edge.shape[0], dtype=int)
        j[valid_edges] = 1
        j = np.cumsum(j)
        edge = edge[valid_edges, :]
        
        # Update faces
        for k in range(len(face)):
            face[k] = np.unique(j[np.array(face[k])]).tolist()
        
        print(f'WARNING: {del_nodes} un-used node(s) removed')
        nNode = node.shape[0]
    
    # Remove duplicate nodes and re-index
    unique_nodes, unique_idx, inverse_idx = np.unique(node, axis=0, return_index=True, return_inverse=True)
    del_nodes = nNode - len(unique_nodes)
    
    if del_nodes > 0:
        node = unique_nodes
        edge = inverse_idx[edge]
        
        # Remove self-edges
        valid_edges = edge[:, 0] != edge[:, 1]
        j = np.zeros(edge.shape[0], dtype=int)
        j[valid_edges] = 1
        j = np.cumsum(j)
        edge = edge[valid_edges, :]
        
        # Update faces
        for k in range(len(face)):
            face[k] = np.unique(j[np.array(face[k])]).tolist()
        
        print(f'WARNING: {del_nodes} duplicate node(s) removed')
        nNode = node.shape[0]
    
    # Remove duplicate edges
    nEdge = edge.shape[0]
    sorted_edges = np.sort(edge, axis=1)
    unique_edges, unique_idx, inverse_idx = np.unique(sorted_edges, axis=0, return_index=True, return_inverse=True)
    
    if edgeh:
        hdata['edgeh'][:, 0] = inverse_idx[hdata['edgeh'][:, 0]]
        j = np.zeros(edge.shape[0], dtype=int)
        j[unique_idx] = 1
        j = np.cumsum(j)
        edge = edge[unique_idx, :]
        
        # Update faces
        for k in range(len(face)):
            face[k] = np.unique(j[np.array(face[k])]).tolist()
    
    del_edges = nEdge - edge.shape[0]
    if del_edges > 0:
        print(f'WARNING: {del_edges} duplicate edge(s) removed')
        nEdge = edge.shape[0]
    
    # Sparse node-to-edge connectivity matrix
    nEdge = edge.shape[0]
    row_indices = edge.flatten()
    col_indices = np.tile(np.arange(nEdge), 2)
    data = np.ones(2 * nEdge)
    
    S = csr_matrix((data, (row_indices, col_indices)), shape=(nNode, nEdge))
    
    # Check for closed geometry loops
    node_degrees = np.array(S.sum(axis=1)).flatten()
    open_nodes = np.where(node_degrees < 2)[0]
    
    if len(open_nodes) > 0:
        raise ValueError(f'Open geometry contours detected at node(s): {open_nodes + 1}')  # Convert back to 1-based for error message
    
    # Note: T-junction check is commented out in original code
    # t_junction_nodes = np.where(node_degrees > 2)[0]
    # if len(t_junction_nodes) > 0:
    #     raise ValueError(f'Multiple geometry branches detected at node(s): {t_junction_nodes + 1}')
    
    return node, edge, face, hdata