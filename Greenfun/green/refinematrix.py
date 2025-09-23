import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def refinematrix(p1, p2, **options):
    """
    REFINEMATRIX - Refinement matrix for Green functions.
    
    Parameters:
    -----------
    p1 : object
        Discretized particle boundary 1
    p2 : object  
        Discretized particle boundary 2
    **options : dict
        Option parameters:
        - AbsCutoff : absolute distance for integration refinement (default: 0)
        - RelCutoff : relative distance for integration refinement (default: 0) 
        - memsize : deal at most with matrices of size memsize (default: 2e7)
        
    Returns:
    --------
    mat : sparse matrix
        Refinement matrix:
        2 - diagonal elements
        1 - off-diagonal elements for refinement
    """
    
    # Set default options
    AbsCutoff = options.get('AbsCutoff', 0)
    RelCutoff = options.get('RelCutoff', 0)
    memsize = options.get('memsize', int(2e7))
    
    # Positions of points or particles
    pos1 = p1.pos
    pos2 = p2.pos
    
    # Boundary element radius
    rad2 = boundary_radius(p2).T
    
    # Radius for relative distances
    try:
        rad = boundary_radius(p1)
    except:
        rad = rad2
    
    # Allocate output array
    mat = csr_matrix((p1.n, p2.n))
    
    # Work through full matrix size N1 x N2 in portions of memsize
    chunk_size = max(1, int(memsize / p1.n))
    ind2 = list(range(0, p2.n, chunk_size)) + [p2.n]
    
    # Loop over portions
    for i in range(1, len(ind2)):
        # Index to positions
        i2 = slice(ind2[i-1], ind2[i])
        pos2_chunk = pos2[i2, :]
        
        # Distance between positions
        d = cdist(pos1, pos2_chunk)
        
        # Subtract radius from distances to get approximate distance between
        # POS1 and boundary elements
        rad2_chunk = rad2[i2]
        d2 = d - rad2_chunk
        
        # Distances in units of boundary element radius
        if rad.shape[0] != 1:
            id_dist = d2 / rad[:, np.newaxis]
        else:
            id_dist = d2 / rad2_chunk
        
        # Find diagonal elements and elements for refinement
        diag_row, diag_col = np.where(d == 0)
        refine_row, refine_col = np.where(
            ((d2 < AbsCutoff) | (id_dist < RelCutoff)) & (d != 0)
        )
        
        # Set diagonal elements (value = 2) and refinement elements (value = 1)
        if len(diag_row) > 0:
            diag_data = np.full(len(diag_row), 2)
            mat = mat + csr_matrix(
                (diag_data, (diag_row, diag_col + ind2[i-1])),
                shape=(p1.n, p2.n)
            )
        
        if len(refine_row) > 0:
            refine_data = np.full(len(refine_row), 1)
            mat = mat + csr_matrix(
                (refine_data, (refine_row, refine_col + ind2[i-1])),
                shape=(p1.n, p2.n)
            )
    
    return mat


def boundary_radius(p):
    """
    Calculate boundary element radius.
    
    This is a placeholder for misc.bradius() function.
    """
    # Placeholder implementation
    # In actual MNPBEM, this would calculate proper boundary element radius
    if hasattr(p, 'area'):
        # Approximate radius from area (assuming circular elements)
        return np.sqrt(p.area / np.pi)
    else:
        # Default radius
        return np.ones(p.n) * 0.1