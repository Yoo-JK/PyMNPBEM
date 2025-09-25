import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Any, Dict, Union, Tuple, List
import warnings

class VecArray:
    """
    Vector array for plotting.
    
    This class manages vector arrays at specified positions and provides
    visualization as either arrows (quiver) or cones.
    """
    
    def __init__(self, pos: np.ndarray, vec: np.ndarray, mode: str = 'cone'):
        """
        Initialize VecArray object.
        
        Parameters:
        -----------
        pos : array_like
            Vector positions (N x 3 array)
        vec : array_like  
            Vector array (N x 3 or N x 3 x M for multi-dimensional)
        mode : str, optional
            Visualization mode: 'cone' or 'arrow' (default: 'cone')
        """
        self.pos = np.asarray(pos)
        self.vec = np.asarray(vec)
        self.mode = mode
        self.h = None  # Handle to graphics object
        self.color = 'blue'  # Default color for arrow plotting
    
    def init2(self, vec: np.ndarray, mode: Optional[str] = None):
        """
        Re-initialize vecarray with new vectors.
        
        Parameters:
        -----------
        vec : array_like
            Vector array
        mode : str, optional
            Visualization mode: 'cone' or 'arrow'
        """
        self.vec = np.asarray(vec)
        if mode is not None:
            self.mode = mode
    
    def is_base(self, pos: np.ndarray) -> bool:
        """
        Check for equality of vector positions.
        
        Parameters:
        -----------
        pos : array_like
            Vector positions to compare
            
        Returns:
        --------
        bool
            True if vector positions are the same
        """
        pos = np.asarray(pos)
        return (pos.shape == self.pos.shape and 
                np.allclose(pos, self.pos))
    
    def is_page(self) -> bool:
        """
        Check if vector array is multi-dimensional.
        
        Returns:
        --------
        bool
            True if array is multi-dimensional for paging
        """
        return self.vec.ndim > 2
    
    def page_size(self, vec: Optional[np.ndarray] = None) -> Tuple[int, ...]:
        """
        Get paging size of vecarray object.
        
        Parameters:
        -----------
        vec : array_like, optional
            Alternative vector array to check
            
        Returns:
        --------
        tuple
            Size dimensions for paging
        """
        if vec is None:
            vec = self.vec
        
        vec = np.asarray(vec)
        if vec.ndim <= 2:
            return (1,)
        else:
            return vec.shape[2:]
    
    def depends(self, *properties) -> bool:
        """
        Check whether object uses one of the given properties.
        
        Parameters:
        -----------
        *properties : str
            Properties to check ('fun', 'ind', 'scale', 'sfun')
            
        Returns:
        --------
        bool
            True if object depends on properties
        """
        if 'ind' in properties and self.is_page():
            return True
        elif any(prop in properties for prop in ['fun', 'scale', 'sfun']):
            return True
        else:
            return False
    
    def _get_vectors(self, opt):
        """
        Extract vectors based on plot options.
        
        Parameters:
        -----------
        opt : object or dict
            Plot options with fun and ind attributes
            
        Returns:
        --------
        np.ndarray
            Processed vectors for plotting
        """
        if self.is_page():
            ind = opt['ind'] if isinstance(opt, dict) else opt.ind
            if ind is not None:
                vec = self.vec[:, :, ind-1]  # Convert to 0-based indexing
            else:
                vec = self.vec[:, :, 0]  # Default to first page
        else:
            vec = self.vec
        
        # Apply function if provided
        if isinstance(opt, dict) and 'fun' in opt:
            return opt['fun'](vec)
        elif hasattr(opt, 'fun'):
            return opt.fun(vec)
        else:
            return vec
    
    def __call__(self, opt):
        """
        Extract vectors using plot options.
        
        Parameters:
        -----------
        opt : object or dict
            Plot options with fun and ind fields
            
        Returns:
        --------
        np.ndarray
            Vectors for plotting
        """
        return self._get_vectors(opt)
    
    def min(self, opt) -> float:
        """
        Get minimum vector magnitude.
        
        Parameters:
        -----------
        opt : object
            Plot options
            
        Returns:
        --------
        float
            Minimum vector magnitude
        """
        vec = self._get_vectors(opt)
        return np.min(np.linalg.norm(vec, axis=1))
    
    def max(self, opt) -> float:
        """
        Get maximum vector magnitude.
        
        Parameters:
        -----------
        opt : object
            Plot options
            
        Returns:
        --------
        float
            Maximum vector magnitude
        """
        vec = self._get_vectors(opt)
        return np.max(np.linalg.norm(vec, axis=1))
    
    def get_minmax(self, opt) -> Tuple[float, float]:
        """
        Get min and max vector magnitudes.
        
        Parameters:
        -----------
        opt : object
            Plot options
            
        Returns:
        --------
        tuple
            (min_magnitude, max_magnitude)
        """
        return self.min(opt), self.max(opt)
    
    def _create_cone_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create cone geometry for 3D plotting.
        
        Returns:
        --------
        tuple
            (vertices, faces) for cone geometry
        """
        # Create cylinder coordinates for cone
        theta = np.linspace(0, 2*np.pi, 21)  # 20 segments
        
        # Cone profile: [tip, wide_base, narrow_base, shaft_bottom, shaft_top]
        r_profile = np.array([0, 0.6, 0.5, 0.5, 0])
        z_profile = np.array([2, 0, 0, -1, -1])
        
        vertices = []
        faces = []
        
        # Create vertices for each ring
        for i, (r, z) in enumerate(zip(r_profile, z_profile)):
            if r == 0:  # Tip or bottom point
                vertices.append([0, 0, z])
            else:
                for t in theta[:-1]:  # Exclude last point (same as first)
                    vertices.append([r * np.cos(t), r * np.sin(t), z])
        
        vertices = np.array(vertices)
        
        # Create faces (simplified triangulation)
        n_theta = len(theta) - 1
        
        # Tip faces
        for i in range(n_theta):
            faces.append([0, 1 + i, 1 + (i + 1) % n_theta])
        
        # Side faces between rings
        base_idx = 1
        for ring in range(1, len(r_profile) - 2):
            next_base = base_idx + n_theta
            for i in range(n_theta):
                i_next = (i + 1) % n_theta
                faces.extend([
                    [base_idx + i, next_base + i, base_idx + i_next],
                    [base_idx + i_next, next_base + i, next_base + i_next]
                ])
            base_idx = next_base
        
        return vertices, np.array(faces)
    
    def _plot_cones(self, vec: np.ndarray, scale: np.ndarray):
        """
        Plot vectors as 3D cones.
        
        Parameters:
        -----------
        vec : np.ndarray
            Vector array
        scale : np.ndarray
            Scaling factors for each vector
        """
        ax = plt.gca()
        if not hasattr(ax, 'zaxis'):
            # Create 3D axes if needed
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        
        # Get cone geometry
        cone_verts, cone_faces = self._create_cone_geometry()
        
        all_vertices = []
        all_faces = []
        all_colors = []
        
        n_cone_verts = len(cone_verts)
        
        for i, (pos, v, s) in enumerate(zip(self.pos, vec, scale)):
            if s <= 0:  # Skip if no scaling
                continue
                
            # Calculate rotation to align cone with vector
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
                
            v_unit = v / v_norm
            
            # Default cone points in +z direction
            z_axis = np.array([0, 0, 1])
            
            # Calculate rotation axis and angle
            if np.allclose(v_unit, z_axis):
                rotation_matrix = np.eye(3)
            elif np.allclose(v_unit, -z_axis):
                rotation_matrix = np.diag([1, 1, -1])
            else:
                # Rodrigues' rotation formula
                k = np.cross(z_axis, v_unit)
                k = k / np.linalg.norm(k)
                cos_angle = np.dot(z_axis, v_unit)
                sin_angle = np.sqrt(1 - cos_angle**2)
                
                K = np.array([[0, -k[2], k[1]],
                             [k[2], 0, -k[0]],
                             [-k[1], k[0], 0]])
                
                rotation_matrix = (np.eye(3) + sin_angle * K + 
                                 (1 - cos_angle) * np.dot(K, K))
            
            # Scale and rotate cone vertices
            scaled_verts = cone_verts * s
            rotated_verts = scaled_verts @ rotation_matrix.T
            translated_verts = rotated_verts + pos
            
            all_vertices.extend(translated_verts)
            all_faces.extend(cone_faces + i * n_cone_verts)
            all_colors.extend([v_norm] * n_cone_verts)
        
        if all_vertices:
            all_vertices = np.array(all_vertices)
            all_faces = np.array(all_faces)
            all_colors = np.array(all_colors)
            
            # Create 3D collection
            collection = Poly3DCollection(all_vertices[all_faces], alpha=0.8)
            collection.set_array(all_colors[::n_cone_verts])  # One color per cone
            collection.set_edgecolors('none')
            
            self.h = ax.add_collection3d(collection)
    
    def _plot_arrows(self, vec: np.ndarray, scale: np.ndarray, **kwargs):
        """
        Plot vectors as arrows using quiver.
        
        Parameters:
        -----------
        vec : np.ndarray
            Vector array
        scale : np.ndarray
            Scaling factors for each vector
        **kwargs : dict
            Additional plotting options
        """
        ax = plt.gca()
        if not hasattr(ax, 'zaxis'):
            # Create 3D axes if needed
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        
        # Get color
        color = kwargs.get('color', self.color)
        
        # Scale vectors
        scaled_vec = vec * scale[:, np.newaxis]
        
        # Create quiver plot
        self.h = ax.quiver(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                          scaled_vec[:, 0], scaled_vec[:, 1], scaled_vec[:, 2],
                          color=color, alpha=0.8)
    
    def plot(self, opt, **kwargs):
        """
        Plot vector array.
        
        Parameters:
        -----------
        opt : object or dict
            Plot options with scale, sfun attributes
        **kwargs : dict
            Additional plotting options
            
        Returns:
        --------
        VecArray
            Self for method chaining
        """
        # Get vectors for plotting
        vec = self._get_vectors(opt)
        
        # Calculate vector magnitudes
        vec_lengths = np.linalg.norm(vec, axis=1)
        
        # Apply scaling function and scaling factor
        scale_factor = opt['scale'] if isinstance(opt, dict) else opt.scale
        scale_func = opt.get('sfun', lambda x: x) if isinstance(opt, dict) else getattr(opt, 'sfun', lambda x: x)
        
        if scale_factor > 0:
            max_len = np.max(vec_lengths) if np.max(vec_lengths) > 0 else 1
            scale = scale_factor * scale_func(vec_lengths / max_len)
        else:
            scale = -scale_factor * scale_func(vec_lengths)
        
        # Remove previous plot
        if self.h is not None:
            try:
                self.h.remove()
            except:
                pass
            self.h = None
        
        # Create new plot based on mode
        if self.mode == 'cone':
            self._plot_cones(vec, scale)
        elif self.mode == 'arrow':
            self._plot_arrows(vec, scale, **kwargs)
        else:
            warnings.warn(f"Unknown plot mode: {self.mode}, using 'arrow'")
            self._plot_arrows(vec, scale, **kwargs)
        
        return self
    
    def __str__(self) -> str:
        """String representation."""
        return (f"VecArray(positions: {self.pos.shape[0]}, "
                f"vectors: {self.vec.shape}, mode: {self.mode})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()