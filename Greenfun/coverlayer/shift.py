import numpy as np
from ..particle import Particle
from ..bemoptions import get_bemoptions
from ..misc import misc


def shift(p1, d, *args, **kwargs):
    """
    Shift boundary for creation of cover layer structure.
    
    Parameters
    ----------
    p1 : object
        Particle
    d : float or array_like
        Shift distance
    *args, **kwargs : optional
        Options and property pairs
        
    Properties
    ----------
    nvec : array_like, optional
        Shift vertices along directions of NVEC
        
    Returns
    -------
    object
        Shifted boundary
    """
    if np.isscalar(d):
        d = np.full(p1.nverts, d)
    
    # Get options
    op = get_bemoptions(*args, **kwargs)
    
    # Get normal vector
    if hasattr(op, 'nvec'):
        nvec = op.nvec
    else:
        # Interpolate normal vectors from faces to vertices
        nvec = p1.interp(p1.nvec)
    
    if p1.verts2 is None or len(p1.verts2) == 0:
        # Unique vertices
        # When shifting vertices, we have to be careful about particle
        # boundaries with duplicate vertices
        _, i1, i2 = np.unique(misc.round(p1.verts, 4), axis=0, return_index=True, return_inverse=True)
        
        # Particle with shifted vertices
        shifted_verts = p1.verts + nvec[i1[i2], :] * d[:, np.newaxis]
        p2 = Particle(shifted_verts, p1.faces, *args, **kwargs)
    else:
        # Interpolate d and nvec to midpoints?
        d = _interp2(p1, d)
        if nvec.shape[0] != p1.verts2.shape[0]:
            nvec = _interp2(p1, nvec)
        
        # Particle with shifted vertices
        shifted_verts = p1.verts2 + nvec * d[:, np.newaxis]
        p2 = Particle(shifted_verts, p1.faces2, *args, **kwargs)
    
    return p2


def _interp2(p, v):
    """
    Interpolate normal vectors from VERTS to VERTS2.
    
    Parameters
    ----------
    p : object
        Particle object
    v : array_like
        Vector field to interpolate
        
    Returns
    -------
    array_like
        Interpolated vector field
    """
    # Index to triangles and quadrilaterals
    ind3, ind4 = p.index34()
    
    # Allocate output
    v2 = np.zeros((p.verts2.shape[0], v.shape[1] if len(v.shape) > 1 else 1))
    
    if len(ind3) > 0:
        # Triangle indices
        i1 = p.faces2[ind3, 0]  # 1-based to 0-based
        i10 = p.faces[ind3, 0]
        i2 = p.faces2[ind3, 1]
        i20 = p.faces[ind3, 1]
        i3 = p.faces2[ind3, 2]
        i30 = p.faces[ind3, 2]
        i4 = p.faces2[ind3, 4]  # 5-based to 4-based
        i5 = p.faces2[ind3, 5]
        i6 = p.faces2[ind3, 6]
        
        # Assign output
        v2[i1, :] = v[i10, :]
        v2[i2, :] = v[i20, :]
        v2[i3, :] = v[i30, :]
        v2[i4, :] = 0.5 * (v[i10, :] + v[i20, :])
        v2[i5, :] = 0.5 * (v[i20, :] + v[i30, :])
        v2[i6, :] = 0.5 * (v[i30, :] + v[i10, :])
    
    if len(ind4) > 0:
        # Quadrilateral indices
        i1 = p.faces2[ind4, 0]
        i10 = p.faces[ind4, 0]
        i2 = p.faces2[ind4, 1]
        i20 = p.faces[ind4, 1]
        i3 = p.faces2[ind4, 2]
        i30 = p.faces[ind4, 2]
        i4 = p.faces2[ind4, 3]
        i40 = p.faces[ind4, 3]
        i5 = p.faces2[ind4, 4]
        i6 = p.faces2[ind4, 5]
        i7 = p.faces2[ind4, 6]
        i8 = p.faces2[ind4, 7]
        i9 = p.faces2[ind4, 8]
        
        # Assign output
        v2[i1, :] = v[i10, :]
        v2[i2, :] = v[i20, :]
        v2[i3, :] = v[i30, :]
        v2[i4, :] = v[i40, :]
        v2[i5, :] = 0.5 * (v[i10, :] + v[i20, :])
        v2[i6, :] = 0.5 * (v[i20, :] + v[i30, :])
        v2[i7, :] = 0.5 * (v[i30, :] + v[i40, :])
        v2[i8, :] = 0.5 * (v[i40, :] + v[i10, :])
        v2[i9, :] = 0.25 * (v[i10, :] + v[i20, :] + v[i30, :] + v[i40, :])
    
    # Unique vertices
    _, i1, i2 = np.unique(misc.round(p.verts2, 4), axis=0, return_index=True, return_inverse=True)
    v2 = v2[i1[i2], :]
    
    return v2