import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union, Any
from scipy.special import roots_legendre

class QuadFace:
    """
    Integration over triangular or quadrilateral boundary elements.
    
    This class provides numerical integration points and weights for
    boundary element method calculations on triangular and quadrilateral
    surface elements.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize quadface object.
        
        Parameters:
        -----------
        rule : int, optional
            Integration rule (1-19, see triangle_unit_set), default: 18
        refine : int, optional
            Number of triangle subdivisions for refinement
        npol : int or list, optional
            Number of points for polar integration [n_radial, n_angular]
        """
        # Standard triangle integration points
        self.x = None
        self.y = None
        self.w = None
        
        # Polar integration points
        self.npol = None
        self.x3 = None  # Polar triangle integration
        self.y3 = None
        self.w3 = None
        self.x4 = None  # Polar quadrilateral integration
        self.y4 = None
        self.w4 = None
        
        self._init(**kwargs)
    
    def _init(self, **kwargs):
        """Initialize quadface object with options."""
        # Default options
        options = {
            'rule': 18,
            'npol': [7, 5]
        }
        options.update(kwargs)
        
        # Ensure npol is a 2-element array
        if isinstance(options['npol'], int):
            options['npol'] = [options['npol'], options['npol']]
        
        self.npol = options['npol']
        
        # Triangle integration
        self.x, self.y, self.w = self._triangle_unit_set(options['rule'])
        
        # Refine triangles if requested
        if 'refine' in options:
            self.x, self.y, self.w = self._tri_subdivide(
                self.x, self.y, self.w, options['refine']
            )
        
        # Polar triangle integration
        self._setup_polar_triangle()
        
        # Polar quadrilateral integration
        self._setup_polar_quadrilateral()
    
    def _triangle_unit_set(self, rule: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Set quadrature rule in unit triangle.
        
        Parameters:
        -----------
        rule : int
            Rule number (1-19)
            
        Returns:
        --------
        tuple
            (x, y, weights) for triangle integration
        """
        # Rule 18: 28 points, precision 11
        if rule == 18:
            a = 1.0 / 3.0
            b = 0.9480217181434233
            c = 0.02598914092828833
            d = 0.8114249947041546
            e = 0.09428750264792270
            f = 0.01072644996557060
            g = 0.4946367750172147
            p = 0.5853132347709715
            q = 0.2073433826145142
            r = 0.1221843885990187
            s = 0.4389078057004907
            t = 0.6779376548825902
            u = 0.04484167758913055
            v = 0.27722066752827925
            w = 0.8588702812826364
            x = 0.0
            y = 0.1411297187173636

            w1 = 0.08797730116222190
            w2 = 0.008744311553736190
            w3 = 0.03808157199393533
            w4 = 0.01885544805613125
            w5 = 0.07215969754474100
            w6 = 0.06932913870553720
            w7 = 0.04105631542928860
            w8 = 0.007362383783300573

            xtab = np.array([a, b, c, c, d, e, e, f, g, g, p, q, q,
                           r, s, s, t, t, u, u, v, v, w, w, x, x, y, y])
            ytab = np.array([a, c, b, c, e, d, e, g, f, g, q, p, q,
                           s, r, s, u, v, t, v, t, u, x, y, w, y, w, x])
            weight = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w5, w5, w5,
                             w6, w6, w6, w7, w7, w7, w7, w7, w7, w8, w8, w8, w8, w8, w8])
            
            return xtab, ytab, weight
        
        # Rule 9: 7 points, precision 5 (commonly used)
        elif rule == 9:
            a = 1.0 / 3.0
            b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
            c = (6.0 - np.sqrt(15.0)) / 21.0
            d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
            e = (6.0 + np.sqrt(15.0)) / 21.0
            u = 0.225
            v = (155.0 - np.sqrt(15.0)) / 1200.0
            w = (155.0 + np.sqrt(15.0)) / 1200.0

            xtab = np.array([a, b, c, c, d, e, e])
            ytab = np.array([a, c, b, c, e, d, e])
            weight = np.array([u, v, v, v, w, w, w])
            
            return xtab, ytab, weight
        
        # Rule 1: 1 point, precision 1 (centroid rule)
        elif rule == 1:
            a = 1.0 / 3.0
            w = 1.0
            return np.array([a]), np.array([a]), np.array([w])
        
        else:
            # Default to rule 9 if unsupported rule requested
            print(f"Warning: Rule {rule} not implemented, using rule 9")
            return self._triangle_unit_set(9)
    
    def _tri_subdivide(self, xtab: np.ndarray, ytab: np.ndarray, 
                      wtab: np.ndarray, nsub: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine triangle integration by subdivision.
        
        Parameters:
        -----------
        xtab, ytab : np.ndarray
            Original integration points
        wtab : np.ndarray
            Original weights
        nsub : int
            Number of subdivisions
            
        Returns:
        --------
        tuple
            Refined (x, y, weights)
        """
        x = []
        y = []
        w = []
        h = 1.0 / nsub
        
        for i in range(nsub):
            for j in range(nsub - i):
                # Triangle pointing upwards
                x.extend(i + xtab)
                y.extend(j + ytab)
                w.extend(wtab)
                
                # Triangle pointing downwards (if not at edge)
                if j != nsub - 1 - i:
                    x.extend(i + 1 - xtab)
                    y.extend(j + 1 - ytab)
                    w.extend(wtab)
        
        x = np.array(x) * h
        y = np.array(y) * h
        w = np.array(w) / (len(x) / len(xtab))
        
        return x, y, w
    
    def _lgl_nodes(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Legendre-Gauss-Lobatto nodes and weights.
        
        Parameters:
        -----------
        n : int
            Number of points
            
        Returns:
        --------
        tuple
            (nodes, weights)
        """
        # Use Gauss-Legendre quadrature as approximation
        nodes, weights = roots_legendre(n)
        return nodes, weights
    
    def _setup_polar_triangle(self):
        """Setup polar integration for triangles."""
        # Radius and angle integration
        x1, w1 = self._lgl_nodes(self.npol[0])
        rho = 0.5 * (x1 + 1 + 1e-6)  # Avoid exact zero
        
        x2, w2 = self._lgl_nodes(self.npol[1])
        phi = (270 + 60 * x2) * np.pi / 180
        
        # Rotation angle for 3-fold symmetry
        phi0 = 120 * np.pi / 180
        
        # Create meshgrid and flatten
        rho_mesh, phi_mesh = np.meshgrid(rho, phi, indexing='ij')
        rho_flat = rho_mesh.flatten()
        phi_flat = phi_mesh.flatten()
        
        # Radius scaling
        rad = 1.0 / np.abs(2 * np.sin(phi_flat))
        
        # Three rotated copies for triangle symmetry
        phi_all = np.concatenate([phi_flat, phi_flat + phi0, phi_flat + 2*phi0])
        rho_all = np.tile(rho_flat, 3)
        rad_all = np.tile(rad, 3)
        
        # Cartesian coordinates
        x = rho_all * rad_all * np.cos(phi_all)
        y = rho_all * rad_all * np.sin(phi_all)
        
        # Transform to unit triangle coordinates
        x_tri = (1 - np.sqrt(3) * x - y) / 3
        y_tri = (1 + np.sqrt(3) * x - y) / 3
        
        # Integration weights
        w_mesh = np.outer(w1, w2)
        w_base = np.tile(w_mesh.flatten(), 3)
        w_tri = w_base * rho_all * rad_all**2
        w_tri = w_tri / np.sum(w_tri)  # Normalize
        
        self.x3 = x_tri
        self.y3 = y_tri
        self.w3 = w_tri
    
    def _setup_polar_quadrilateral(self):
        """Setup polar integration for quadrilaterals."""
        # Radius and angle integration
        x1, w1 = self._lgl_nodes(self.npol[0])
        rho = 0.5 * (x1 + 1 + 1e-6)
        
        x2, w2 = self._lgl_nodes(self.npol[1])
        phi = (90 + 45 * x2) * np.pi / 180
        
        # Rotation angle for 4-fold symmetry
        phi0 = np.pi / 2
        
        # Create meshgrid and flatten
        rho_mesh, phi_mesh = np.meshgrid(rho, phi, indexing='ij')
        rho_flat = rho_mesh.flatten()
        phi_flat = phi_mesh.flatten()
        
        # Radius scaling
        rad = 1.0 / np.abs(np.sin(phi_flat))
        
        # Four rotated copies for quadrilateral symmetry
        phi_all = np.concatenate([phi_flat, phi_flat + phi0, 
                                phi_flat + 2*phi0, phi_flat + 3*phi0])
        rho_all = np.tile(rho_flat, 4)
        rad_all = np.tile(rad, 4)
        
        # Cartesian coordinates
        x = rho_all * rad_all * np.cos(phi_all)
        y = rho_all * rad_all * np.sin(phi_all)
        
        # Integration weights
        w_mesh = np.outer(w1, w2)
        w_base = np.tile(w_mesh.flatten(), 4)
        w_quad = w_base * rho_all * rad_all**2
        w_quad = 4 * w_quad / np.sum(w_quad)  # Normalize
        
        self.x4 = x
        self.y4 = y
        self.w4 = w_quad
    
    def _adapt_rule(self, verts: np.ndarray, tri_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt triangle integration to actual triangle vertices.
        
        Parameters:
        -----------
        verts : np.ndarray
            Vertex coordinates
        tri_indices : List[int]
            Triangle vertex indices
            
        Returns:
        --------
        tuple
            (positions, weights)
        """
        # Extract triangle vertices
        v1 = verts[tri_indices[0]]
        v2 = verts[tri_indices[1]]
        v3 = verts[tri_indices[2]]
        
        # Transform integration points to physical triangle
        pos = (self.x[:, np.newaxis] * v1 + 
               self.y[:, np.newaxis] * v2 + 
               (1 - self.x - self.y)[:, np.newaxis] * v3)
        
        # Compute area factor (Jacobian)
        edge1 = v1 - v3
        edge2 = v2 - v3
        normal = np.cross(edge1, edge2)
        area_factor = 0.5 * np.linalg.norm(normal)
        
        # Scale weights by area
        weights = self.w * area_factor
        
        return pos, weights
    
    def __call__(self, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt integration points to boundary element.
        
        Usage:
        ------
        pos, w = obj(verts)              # Single array of vertices
        pos, w = obj(v1, v2, v3)         # Triangle vertices
        pos, w = obj(v1, v2, v3, v4)     # Quadrilateral vertices
        pos, w = obj(p, ind)             # Particle object and face index
        
        Returns:
        --------
        tuple
            (positions, weights) for integration
        """
        if len(args) == 1:
            verts = np.asarray(args[0])
        elif len(args) == 2 and not isinstance(args[0], np.ndarray):
            # Handle particle object case (p, ind)
            p, ind = args
            if hasattr(p, 'faces') and hasattr(p, 'verts'):
                face_indices = p.faces[ind]
                verts = p.verts[face_indices]
            else:
                raise ValueError("Invalid particle object")
        else:
            # Multiple vertex arguments
            verts = np.vstack(args)
        
        if verts.shape[0] == 3:
            # Triangle
            pos, w = self._adapt_rule(verts, [0, 1, 2])
        elif verts.shape[0] == 4:
            # Quadrilateral - divide into two triangles
            pos_a, w_a = self._adapt_rule(verts, [0, 1, 2])
            pos_b, w_b = self._adapt_rule(verts, [2, 3, 0])
            pos = np.vstack([pos_a, pos_b])
            w = np.concatenate([w_a, w_b])
        else:
            raise ValueError("Only triangular and quadrilateral elements supported")
        
        return pos, w
    
    def plot(self, **kwargs):
        """
        Plot integration points for boundary element integration.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional plotting parameters
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Triangle integration points
        ax1.plot(self.x, self.y, 'b.', label='Triangle points')
        ax1.plot(1 - self.x3, 1 - self.y3, 'r.', label='Polar triangle')
        
        # Plot triangle boundaries
        triangle_x = [0, 1, 0, 0]
        triangle_y = [0, 0, 1, 0]
        ax1.plot(triangle_x, triangle_y, 'k-', linewidth=1)
        
        # Plot unit square
        square_x = [1, 1, 0, 1]
        square_y = [0, 1, 1, 0]
        ax1.plot(square_x, square_y, 'k-', linewidth=1)
        
        ax1.set_aspect('equal')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'nz = {len(self.x)} ({len(self.x3)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Polar quadrilateral integration points
        ax2.plot(self.x4, self.y4, 'r.', label='Polar quad')
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.05, 1.05)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'nz = {len(self.x4)}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation."""
        return (f"QuadFace(points: {len(self.x)}, "
                f"polar_tri: {len(self.x3) if self.x3 is not None else 0}, "
                f"polar_quad: {len(self.x4) if self.x4 is not None else 0})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()