def memsize():
    """
    MEMSIZE - Memory size used for working successively through large arrays.
    
    Returns:
    --------
    int
        Memory size (number of elements)
        
    Notes:
    ------
    When working successively through large arrays, we split the array into
    portions of maximal size m to avoid out of memory problems. The value
    of m is an approximate value which does not guarantee out of memory
    problems. In case of limited or more exhaustive available memory one
    might consider decreasing or increasing this value, or to pass a
    different memory size in the options array.
    """
    m = 5000 * 5000
    return m