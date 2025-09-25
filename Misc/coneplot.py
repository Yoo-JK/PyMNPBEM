from .bemplot import BemPlot

def coneplot(pos, vec, **kwargs):
    """
    CONEPLOT - Plot vectors at given positions using cones.
    
    Parameters:
    -----------
    pos : array_like
        Positions where cones are plotted
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
        Handle to the plotted cones
    """
    # Plot cones using BemPlot class
    h = BemPlot.get(**kwargs).plotcone(pos, vec, **kwargs)
    
    return h