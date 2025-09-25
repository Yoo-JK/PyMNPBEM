from .bemplot import BemPlot

def arrowplot(pos, vec, **kwargs):
    """
    ARROWPLOT - Plot vectors at given positions using arrows.
    
    Parameters:
    -----------
    pos : array_like
        Positions where arrows are plotted
    vec : array_like
        Vectors to be plotted
    **kwargs : dict
        Additional properties:
        fun : callable, optional
            Plot function
        scale : float, optional
            Scale factor for vectors
        sfun : callable, optional
            Scaling function for vectors
            
    Returns:
    --------
    object
        Handle to the plotted arrows
    """
    # Plot arrows using BemPlot class
    h = BemPlot.get(**kwargs).plotarrow(pos, vec, **kwargs)
    
    return h