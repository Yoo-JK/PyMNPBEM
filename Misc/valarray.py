import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Any, Dict, Union, Tuple
import warnings

class ValArray:
    """
    Value array for plotting.
    
    This class manages value arrays associated with particle surfaces
    for visualization purposes. It supports both regular value arrays
    and true color arrays.
    """
    
    def __init__(self, p, val=None, mode=None):
        """
        Initialize ValArray object.
        
        Parameters:
        -----------
        p : object
            Particle object with vertices and faces
        val : array_like, optional
            Value array to plot
        mode : str, optional
            'truecolor' for RGB color values
        """
        self.p = None
        self.val = None
        self.truecolor = False
        self.h = None  # Handle to graphics object
        
        self._init(p, val, mode)
    
    def _init(self, p, val=None, mode=None):
        """Initialize valarray object."""
        self.p = p
        
        if val is not None:
            # Expand to full size or interpolate from faces to vertices
            val = np.asarray(val)
            
            if val.shape[0] == 1:
                # Replicate single value to all vertices
                val = np.tile(val, (p.nverts, 1))
            elif val.shape[0] == p.n:
                # Interpolate from faces to vertices
                val = self._interp_faces_to_vertices(p, val)
            
            # Save value array
            self.val = val
            self.truecolor = (mode == 'truecolor')
        else:
            # Set default orange color values
            self.val = np.tile([1.0, 0.7, 0.0], (p.nverts, 1))
            self.truecolor = True
    
    def init2(self, val=None, mode=None):
        """
        Re-initialize valarray with new values.
        
        Parameters:
        -----------
        val : array_like, optional
            Value array to plot
        mode : str, optional
            'truecolor' for RGB color values
        """
        if val is not None:
            val = np.asarray(val)
            
            if val.shape[0] == 1:
                val = np.tile(val, (self.p.nverts, 1))
            elif val.shape[0] == self.p.n:
                val = self._interp_faces_to_vertices(self.p, val)
            
            self.val = val
            self.truecolor = (mode == 'truecolor')
        else:
            # Set default orange color values
            self.val = np.tile([1.0, 0.7, 0.0], (self.p.nverts, 1))
            self.truecolor = True
    
    def _interp_faces_to_vertices(self, p, val):
        """
        Interpolate values from faces to vertices.
        
        This is a simplified interpolation - in practice, this would depend
        on the specific particle object structure.
        """
        # Simple averaging interpolation (placeholder implementation)
        if hasattr(p, 'faces') and hasattr(p, 'nverts'):
            vertex_vals = np.zeros((p.nverts, val.shape[1] if val.ndim > 1 else 1))
            vertex_counts = np.zeros(p.nverts)
            
            for i, face in enumerate(p.faces):
                for vertex_idx in face:
                    if vertex_idx < p.nverts:
                        vertex_vals[vertex_idx] += val[i] if val.ndim == 1 else val[i, :]
                        vertex_counts[vertex_idx] += 1
            
            # Avoid division by zero
            nonzero_mask = vertex_counts > 0
            vertex_vals[nonzero_mask] /= vertex_counts[nonzero_mask, np.newaxis] if val.ndim > 1 else vertex_counts[nonzero_mask]
            
            return vertex_vals
        else:
            return val
    
    def is_base(self, p) -> bool:
        """
        Check for equality of particle objects.
        
        Parameters:
        -----------
        p : object
            Discretized particle object
            
        Returns:
        --------
        bool
            True if particle of self is same as p
        """
        if (hasattr(p, 'verts') and hasattr(self.p, 'verts') and
            p.verts.shape == self.p.verts.shape):
            return np.allclose(p.verts, self.p.verts)
        return False
    
    def is_page(self) -> bool:
        """
        Check if value array is multi-dimensional.
        
        Returns:
        --------
        bool
            True if array is multi-dimensional for paging
        """
        if self.truecolor:
            return False
        else:
            return (self.val.shape[1] != 1 if self.val.ndim > 1 else False) or self.val.ndim > 2
    
    def page_size(self, val=None):
        """
        Get paging size of valarray object.
        
        Parameters:
        -----------
        val : array_like, optional
            Alternative value array to check
            
        Returns:
        --------
        tuple
            Size dimensions for paging
        """
        if val is None:
            val = self.val
        
        val = np.asarray(val)
        siz = val.shape
        return siz[1:] if len(siz) > 1 else (1,)
    
    def depends(self, *properties) -> bool:
        """
        Check whether object uses one of the given properties.
        
        Parameters:
        -----------
        *properties : str
            Properties to check ('fun', 'ind')
            
        Returns:
        --------
        bool
            True if object depends on properties
        """
        if 'ind' in properties and self.is_page():
            return True
        elif 'fun' in properties and not self.truecolor:
            return True
        else:
            return False
    
    def min(self, opt) -> Optional[float]:
        """
        Get minimum value of array.
        
        Parameters:
        -----------
        opt : object
            Plot options with fun and ind attributes
            
        Returns:
        --------
        float or None
            Minimum value or None for truecolor arrays
        """
        if self.truecolor:
            return None
        else:
            val = self._get_values(opt)
            return np.min(val) if val is not None else None
    
    def max(self, opt) -> Optional[float]:
        """
        Get maximum value of array.
        
        Parameters:
        -----------
        opt : object
            Plot options with fun and ind attributes
            
        Returns:
        --------
        float or None
            Maximum value or None for truecolor arrays
        """
        if self.truecolor:
            return None
        else:
            val = self._get_values(opt)
            return np.max(val) if val is not None else None
    
    def get_minmax(self, opt) -> Tuple[Optional[float], Optional[float]]:
        """
        Get min and max values for color scaling.
        
        Parameters:
        -----------
        opt : object
            Plot options
            
        Returns:
        --------
        tuple
            (min_value, max_value) or (None, None) for truecolor
        """
        return self.min(opt), self.max(opt)
    
    def _get_values(self, opt):
        """
        Extract values based on plot options.
        
        Parameters:
        -----------
        opt : object
            Plot options with fun and ind attributes
            
        Returns:
        --------
        np.ndarray
            Processed values for plotting
        """
        if self.truecolor:
            return self.val
        elif not self.is_page():
            return opt['fun'](self.val) if isinstance(opt, dict) else opt.fun(self.val)
        else:
            ind = opt['ind'] if isinstance(opt, dict) else opt.ind
            if ind is not None:
                if isinstance(ind, (list, tuple)):
                    # Handle multidimensional indexing
                    return opt['fun'](self.val[:, ind]) if isinstance(opt, dict) else opt.fun(self.val[:, ind])
                else:
                    return opt['fun'](self.val[:, ind-1]) if isinstance(opt, dict) else opt.fun(self.val[:, ind-1])  # Convert to 0-based indexing
            else:
                return opt['fun'](self.val) if isinstance(opt, dict) else opt.fun(self.val)
    
    def __call__(self, opt):
        """
        Extract values using plot options.
        
        Parameters:
        -----------
        opt : object or dict
            Plot options with fun and ind fields
            
        Returns:
        --------
        np.ndarray
            Values for plotting
        """
        return self._get_values(opt)
    
    def plot(self, opt, **kwargs):
        """
        Plot value array on particle surface.
        
        Parameters:
        -----------
        opt : object or dict
            Plot options
        **kwargs : dict
            Additional plotting options (e.g., FaceAlpha)
            
        Returns:
        --------
        ValArray
            Self for method chaining
        """
        # Get values for plotting
        val = self._get_values(opt)
        
        # Get plotting options
        face_alpha = kwargs.get('FaceAlpha', 1.0)
        
        if self.h is None:
            # Create new plot
            ax = plt.gca()
            
            if hasattr(self.p, 'faces') and hasattr(self.p, 'verts'):
                # Create 3D surface plot
                if hasattr(ax, 'zaxis'):  # 3D axes
                    if self.truecolor and val.shape[1] >= 3:
                        # True color plotting
                        collection = Poly3DCollection(
                            self.p.verts[self.p.faces],
                            facecolors=val[self.p.faces].mean(axis=1),
                            alpha=face_alpha,
                            edgecolors='none'
                        )
                    else:
                        # Scalar value plotting
                        collection = Poly3DCollection(
                            self.p.verts[self.p.faces],
                            alpha=face_alpha,
                            edgecolors='none'
                        )
                        collection.set_array(val.flatten() if val.ndim > 1 else val)
                    
                    self.h = ax.add_collection3d(collection)
                else:
                    # 2D plotting
                    warnings.warn("2D plotting of 3D surface not fully implemented")
            else:
                warnings.warn("Particle object missing required attributes (faces, verts)")
        else:
            # Update existing plot
            if hasattr(self.h, 'set_array') and not self.truecolor:
                self.h.set_array(val.flatten() if val.ndim > 1 else val)
            elif hasattr(self.h, 'set_facecolors') and self.truecolor:
                if val.shape[1] >= 3:
                    self.h.set_facecolors(val[self.p.faces].mean(axis=1))
            
            if 'FaceAlpha' in kwargs and hasattr(self.h, 'set_alpha'):
                self.h.set_alpha(face_alpha)
        
        return self
    
    def __str__(self) -> str:
        """String representation."""
        return (f"ValArray(vertices: {self.p.nverts if hasattr(self.p, 'nverts') else 'unknown'}, "
                f"values: {self.val.shape if self.val is not None else 'None'}, "
                f"truecolor: {self.truecolor})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()