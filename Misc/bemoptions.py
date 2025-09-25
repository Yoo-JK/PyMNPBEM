def bemoptions(op=None, **kwargs):
    """
    BEMOPTIONS - Set standard options for MNPBEM simulation.
    
    Parameters:
    -----------
    op : dict, optional
        Option structure from previous call
    **kwargs : dict
        Option name and value pairs to be added to option structure
        
    Returns:
    --------
    dict
        Structure with standard or user-defined options
    """
    # Handle case that no op structure is provided
    if op is None:
        op = {}
        
        op['sim'] = 'ret'           # Retarded simulation on default
        op['waitbar'] = 1           # Show progress of simulation
        op['RelCutoff'] = 3         # Cutoff parameter for face integration
        # op['pinfty'] = trisphere(256, 2)  # Unit sphere at infinity for spectra
        op['order'] = 5             # Order for exp(1i * k * r) expansion
        op['interp'] = 'flat'       # Particle surface interpolation ('flat' or 'curv')
    
    # Add property pairs from kwargs
    for key, value in kwargs.items():
        op[key] = value
    
    return op