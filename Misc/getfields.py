import inspect

def getfields(param, *field_names):
    """
    GETFIELDS - Assign the fields of a structure in the calling program.
    
    This function extracts fields from a dictionary/object and assigns them
    as variables in the calling function's namespace.
    
    Parameters:
    -----------
    param : dict or object
        Structure/dictionary containing fields to extract
    *field_names : str, optional
        Specific field names to extract. If not provided, all fields are extracted.
        
    Note:
    -----
    This function modifies the calling function's local variables by adding
    new variables with names matching the dictionary keys.
    
    Example:
    --------
    def example_function():
        options = {'alpha': 1.0, 'beta': 2.0, 'gamma': 3.0}
        getfields(options)  # Creates local variables alpha, beta, gamma
        print(alpha)  # Prints 1.0
        
        # Or extract specific fields only:
        getfields(options, 'alpha', 'beta')  # Creates only alpha and beta
    """
    # Get the calling frame
    frame = inspect.currentframe().f_back
    
    # Get field names based on input type
    if isinstance(param, dict):
        available_fields = list(param.keys())
        get_value = lambda name: param[name]
    else:
        # Handle object with attributes
        available_fields = [name for name in dir(param) if not name.startswith('_')]
        get_value = lambda name: getattr(param, name)
    
    # Determine which fields to extract
    if field_names:
        # Extract only specified fields that exist
        fields_to_extract = [name for name in field_names if name in available_fields]
    else:
        # Extract all fields
        fields_to_extract = available_fields
    
    # Assign fields to calling function's local namespace
    for field_name in fields_to_extract:
        try:
            value = get_value(field_name)
            frame.f_locals[field_name] = value
        except (KeyError, AttributeError):
            # Skip fields that don't exist
            continue


def getfields_return(param, *field_names):
    """
    Alternative implementation that returns values instead of modifying caller's namespace.
    
    This is often more Pythonic and safer than modifying the caller's namespace.
    
    Parameters:
    -----------
    param : dict or object
        Structure/dictionary containing fields to extract
    *field_names : str, optional
        Specific field names to extract
        
    Returns:
    --------
    dict
        Dictionary containing extracted fields
        
    Example:
    --------
    options = {'alpha': 1.0, 'beta': 2.0, 'gamma': 3.0}
    extracted = getfields_return(options, 'alpha', 'beta')
    alpha = extracted['alpha']
    beta = extracted['beta']
    """
    if isinstance(param, dict):
        available_fields = list(param.keys())
        get_value = lambda name: param[name]
    else:
        available_fields = [name for name in dir(param) if not name.startswith('_')]
        get_value = lambda name: getattr(param, name)
    
    # Determine which fields to extract
    if field_names:
        fields_to_extract = [name for name in field_names if name in available_fields]
    else:
        fields_to_extract = available_fields
    
    # Create result dictionary
    result = {}
    for field_name in fields_to_extract:
        try:
            result[field_name] = get_value(field_name)
        except (KeyError, AttributeError):
            continue
    
    return result