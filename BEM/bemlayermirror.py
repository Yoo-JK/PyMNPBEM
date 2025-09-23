"""
BEM Layer Mirror - Python Implementation

Dummy class, BEM solvers for layer and mirror symmetry not implemented.
Translated from MATLAB MNPBEM toolbox @bemlayermirror class.
"""

from ..Base.bembase import BemBase


class BemLayerMirror(BemBase):
    """
    Dummy class, BEM solvers for layer and mirror symmetry not implemented.
    
    This is a placeholder class that raises an error when instantiated,
    indicating that BEM solvers for layers and mirror symmetry are not
    yet implemented in the MNPBEM toolbox.
    """
    
    # BemBase abstract properties
    name = 'bemsolver'
    needs = ['sim', 'layer', 'sym']
    
    def __init__(self, *args, **kwargs):
        """
        Dummy class constructor.
        
        Raises:
        -------
        NotImplementedError
            Always raises this error as BEM solvers for layers and 
            mirror symmetry are not implemented.
        """
        raise NotImplementedError(
            'BEM solvers for layers and mirror symmetry not implemented'
        )


# For consistency with MATLAB naming
bemlayermirror = BemLayerMirror