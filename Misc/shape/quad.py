import numpy as np

class Quad:
    """Quadrilateral shape element."""
    
    def __init__(self, node):
        """
        Initialize quadrilateral shape element.
        
        Parameters
        ----------
        node : int
            Number of nodes (4 or 9)
        """
        self.node = node
    
    def __repr__(self):
        """String representation."""
        return f"Quad(node={self.node})"
    
    def __call__(self, x, y):
        """Evaluate quadrilateral shape functions."""
        return self.eval(x, y)
    
    def eval(self, x, y):
        """
        Evaluate shape function.
        
        Parameters
        ----------
        x, y : array_like
            Quadrilateral coordinates
            
        Returns
        -------
        s : ndarray
            Quadrilateral shape function
        """
        # Reshape quadrilateral coordinates
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if self.node == 4:
            s = 0.25 * np.column_stack([
                (1 - x) * (1 - y),
                (1 + x) * (1 - y),
                (1 + x) * (1 + y),
                (1 - x) * (1 + y)
            ])
        elif self.node == 9:
            # Assembly function
            def fun(x_vals, y_vals):
                return np.column_stack([
                    x_vals[:, 0] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 2],
                    x_vals[:, 0] * y_vals[:, 2], x_vals[:, 1] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 1],
                    x_vals[:, 1] * y_vals[:, 2], x_vals[:, 0] * y_vals[:, 1], x_vals[:, 1] * y_vals[:, 1]
                ])
            
            sx = np.column_stack([
                0.5 * x * (x - 1),
                1 - x**2,
                0.5 * x * (1 + x)
            ])
            sy = np.column_stack([
                0.5 * y * (y - 1),
                1 - y**2,
                0.5 * y * (1 + y)
            ])
            
            s = fun(sx, sy)
        else:
            raise ValueError(f"Unsupported node count: {self.node}")
            
        return s
    
    def deriv(self, x, y, key):
        """
        Derivative of shape function.
        
        Parameters
        ----------
        x, y : array_like
            Quadrilateral coordinates
        key : str
            Derivative type: 'x', 'y', 'xx', 'yy', 'xy'
            
        Returns
        -------
        sp : ndarray
            Derivative of quadrilateral shape function
        """
        # Reshape quadrilateral coordinates
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if self.node == 4:
            return self._deriv4(x, y, key)
        elif self.node == 9:
            return self._deriv9(x, y, key)
        else:
            raise ValueError(f"Unsupported node count: {self.node}")
    
    def _deriv4(self, x, y, key):
        """Derivative for 4-node quadrilateral."""
        if key == 'x':
            sp = 0.25 * np.column_stack([
                -(1 - y), (1 - y), (1 + y), -(1 + y)
            ])
        elif key == 'y':
            sp = 0.25 * np.column_stack([
                -(1 - x), -(1 + x), (1 + x), (1 - x)
            ])
        elif key == 'xy':
            sp = np.tile(0.25 * np.array([1, -1, 1, -1]), (len(x), 1))
        else:  # 'xx', 'yy', or other
            sp = np.tile(np.array([0, 0, 0, 0]), (len(x), 1))
            
        return sp
    
    def _deriv9(self, x, y, key):
        """Derivative for 9-node quadrilateral."""
        # Shape functions
        sx0 = np.column_stack([
            0.5 * x * (x - 1),
            1 - x**2,
            0.5 * x * (1 + x)
        ])
        sy0 = np.column_stack([
            0.5 * y * (y - 1),
            1 - y**2,
            0.5 * y * (1 + y)
        ])
        
        # Derivatives of shape functions
        sx1 = np.column_stack([x - 0.5, -2 * x, x + 0.5])
        sx2 = np.tile([1, -2, 1], (len(x), 1))
        sy1 = np.column_stack([y - 0.5, -2 * y, y + 0.5])
        sy2 = np.tile([1, -2, 1], (len(y), 1))
        
        # Assembly function
        def fun(x_vals, y_vals):
            return np.column_stack([
                x_vals[:, 0] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 2],
                x_vals[:, 0] * y_vals[:, 2], x_vals[:, 1] * y_vals[:, 0], x_vals[:, 2] * y_vals[:, 1],
                x_vals[:, 1] * y_vals[:, 2], x_vals[:, 0] * y_vals[:, 1], x_vals[:, 1] * y_vals[:, 1]
            ])
        
        if key == 'x':
            sp = fun(sx1, sy0)
        elif key == 'y':
            sp = fun(sx0, sy1)
        elif key == 'xx':
            sp = fun(sx2, sy0)
        elif key == 'yy':
            sp = fun(sx0, sy2)
        elif key == 'xy':
            sp = fun(sx1, sy1)
        else:
            raise ValueError(f"Unsupported derivative key: {key}")
            
        return sp
    
    # Property-like access for derivatives
    def x(self, x, y):
        """x-derivative of shape functions."""
        return self.deriv(x, y, 'x')
    
    def y(self, x, y):
        """y-derivative of shape functions."""
        return self.deriv(x, y, 'y')
    
    def xx(self, x, y):
        """xx-derivative of shape functions."""
        return self.deriv(x, y, 'xx')
    
    def yy(self, x, y):
        """yy-derivative of shape functions."""
        return self.deriv(x, y, 'yy')
    
    def xy(self, x, y):
        """xy-derivative of shape functions."""
        return self.deriv(x, y, 'xy')


# Example usage and testing
if __name__ == "__main__":
    # Test 4-node quadrilateral
    quad4 = Quad(4)
    print("4-node quadrilateral:", quad4)
    
    # Test coordinates
    x_test = np.array([0.0, 0.5])
    y_test = np.array([0.0, 0.5])
    
    # Evaluate shape functions
    s4 = quad4(x_test, y_test)
    print("Shape functions (4-node):", s4)
    
    # Test derivatives
    dx4 = quad4.x(x_test, y_test)
    print("x-derivative (4-node):", dx4)
    
    # Test 9-node quadrilateral
    quad9 = Quad(9)
    print("\n9-node quadrilateral:", quad9)
    
    s9 = quad9(x_test, y_test)
    print("Shape functions (9-node):", s9)
    
    dx9 = quad9.x(x_test, y_test)
    print("x-derivative (9-node):", dx9)