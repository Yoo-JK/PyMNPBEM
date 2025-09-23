def mesh2d(node, edge=None, hdata=None, options=None):
    """
    2D unstructured mesh generation for a polygon.
    
    A 2D unstructured triangular mesh is generated based on a piecewise-
    linear geometry input. The polygon can contain an arbitrary number of 
    cavities. An iterative method is implemented to optimise mesh quality.
    
    Parameters:
    -----------
    node : ndarray, shape (N, 2)
        Geometry nodes as an array [x1 y1; x2 y2; etc], specified in 
        consecutive order
    edge : ndarray, shape (M, 2), optional
        Connectivity between the points in NODE as a list of edges
        [n1 n2; n2 n3; etc]
    hdata : dict, optional
        Dictionary containing user defined element size information:
        - 'hmax': Max allowable global element size
        - 'edgeh': Element size on specified geometry edges [[e1,h1], [e2,h2], ...]
        - 'fun': User defined size function (callable)
        - 'args': Additional arguments for hdata['fun']
    options : dict, optional
        Dictionary containing tuning parameters:
        - 'mlim': Convergence tolerance (default: 0.02)
        - 'maxit': Maximum iterations (default: 20)
        - 'dhmax': Maximum relative gradient in size function (default: 0.3)
        - 'output': Display mesh and statistics (default: True)
        
    Returns:
    --------
    p : ndarray, shape (N', 2)
        Array of nodal XY co-ordinates
    t : ndarray, shape (M', 3)
        Array of triangles as indices into P, with counter-clockwise ordering
    stats : dict
        Algorithm statistics (debugging output)
    """
    
    # Assume 1 face containing all edges
    from .meshfaces import meshfaces  # Assuming meshfaces is implemented
    
    p, t, _, stats = meshfaces(node, edge, None, hdata, options)
    
    return p, t, stats