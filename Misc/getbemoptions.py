def getbemoptions(*args, **kwargs):
    """
    GETBEMOPTIONS - Get options for MNPBEM simulation.
    
    Parameters:
    -----------
    *args : variable arguments
        Can contain:
        - dict: options structure
        - list: substructure names to extract
    **kwargs : keyword arguments
        Property pairs that override existing options
        
    Returns:
    --------
    dict
        Structure with options; relevant properties can be accessed 
        with op[fieldname]
        
    Example:
    --------
    op = {'sim': 'ret', 'RelCutoff': 0.1}
    op['green'] = {'RelCutoff': 0.2, 'AbsCutoff': 1.0}
    result = getbemoptions(op, ['green'], RelCutoff=0.4)
    
    Result:
    {
        'sim': 'ret',
        'RelCutoff': 0.4,
        'green': {'RelCutoff': 0.2, 'AbsCutoff': 1.0},
        'AbsCutoff': 1.0
    }
    """
    # Initialize options structure
    op = {}
    sub = None
    
    # Process positional arguments
    for arg in args:
        if isinstance(arg, dict):
            # Options structure - copy all fields
            for key, value in arg.items():
                op[key] = value
            
            # Extract fields from substructures if sub is defined
            if sub is not None:
                op = _extract_substructures(op, sub)
                
        elif isinstance(arg, list):
            # List of substructure names
            sub = arg
            # Extract fields from substructures
            op = _extract_substructures(op, sub)
            
        elif isinstance(arg, str):
            raise ValueError("String arguments should be passed as keyword arguments")
        else:
            raise ValueError("Arguments must be dict or list of substructure names")
    
    # Process keyword arguments (property pairs)
    for key, value in kwargs.items():
        op[key] = value
    
    return op


def _extract_substructures(op, sub_names):
    """
    Extract fields from substructures.
    
    Parameters:
    -----------
    op : dict
        Options dictionary
    sub_names : list
        List of substructure names to extract from
        
    Returns:
    --------
    dict
        Updated options dictionary
    """
    # Get existing field names
    existing_keys = list(op.keys())
    
    # Loop over substructure names
    for sub_name in sub_names:
        if sub_name in existing_keys and isinstance(op[sub_name], dict):
            # Copy all fields from substructure to main structure
            for field_name, field_value in op[sub_name].items():
                op[field_name] = field_value
    
    return op