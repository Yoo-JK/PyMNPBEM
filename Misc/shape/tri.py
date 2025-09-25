import numpy as np

class Tri:
    """Triangular shape element."""
    
    def __init__(self, node):
        """
        Initialize triangular shape element.
        
        Parameters
        ----------
        node : int
            Number of nodes (3 or 6)
        """
        self.node = node
    
    def __repr__(self):
        """String representation."""
        return f"Tri(node={self.node})"
    
    def __call__(self, x, y):
        """Evaluate triangular shape functions."""
        return self.eval(x, y)
    
    def eval(self, x, y):
        """
        Evaluate shape function.
        
        Parameters
        ----------
        x, y : array_like
            Triangular coordinates
            
        Returns
        -------
        s : ndarray
            Triangular shape function
        """
        # Reshape triangular coordinates
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # Third triangular coordinate
        z = 1 - x - y
        
        if self.node == 3:
            s = np.column_stack([x, y, z])
        elif self.node == 6:
            s = np.column_stack([
                x * (2 * x - 1),
                y * (2 * y - 1),
                z * (2 * z - 1),
                4 * x * y,
                4 * y * z,
                4 * z * x
            ])
        else:
            raise ValueError(f"Unsupported node count: {self.node}")
            
        return s
    
    def deriv(self, x, y, key):
        """
        Derivative of shape function.
        
        Parameters
        ----------
        x, y : array_like
            Triangular coordinates
        key : str
            Derivative type: 'x', 'y', 'xx', 'yy', 'xy'
            
        Returns
        -------
        sp : ndarray
            Derivative of triangular shape function
        """
        # Reshape triangular coordinates
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if self.node == 3:
            return self._deriv3(x, y, key)
        elif self.node == 6:
            return self._deriv6(x, y, key)
        else:
            raise ValueError(f"Unsupported node count: {self.node}")
    
    def _deriv3(self, x, y, key):
        """Derivative for 3-node triangle."""
        if key == 'x':
            sp = np.array([1, 0, -1])
        elif key == 'y':
            sp = np.array([0, 1, -1])
        else:  # 'xx', 'yy', 'xy', or other
            sp = np.array([0, 0, 0])
        
        # Reshape output array
        sp = np.tile(sp, (len(x), 1))
        return sp
    
    def _deriv6(self, x, y, key):
        """Derivative for 6-node triangle."""
        z = 1 - x - y
        
        if key == 'x':
            sp = np.column_stack([
                4 * x - 1,
                0 * y,
                1 - 4 * z,
                4 * y,
                -4 * y,
                4 * (z - x)
            ])
        elif key == 'y':
            sp = np.column_stack([
                0 * x,
                4 * y - 1,
                1 - 4 * z,
                4 * x,
                4 * (z - y),
                -4 * x
            ])
        elif key == 'xx':
            sp = np.tile([4, 0, 4, 0, 0, -8], (len(x), 1))
        elif key == 'yy':
            sp = np.tile([0, 4, 4, 0, -8, 0], (len(x), 1))
        elif key == 'xy':
            sp = np.tile([0, 0, 4, 4, -4, -4], (len(x), 1))
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
    # Test 3-node triangle
    tri3 = Tri(3)
    print("3-node triangle:", tri3)
    
    # Test coordinates
    x_test = np.array([0.2, 0.3])
    y_test = np.array([0.3, 0.2])
    
    # Evaluate shape functions
    s3 = tri3(x_test, y_test)
    print("Shape functions (3-node):", s3)
    print("Sum check (should be 1):", np.sum(s3, axis=1))
    
    # Test derivatives
    dx3 = tri3.x(x_test, y_test)
    print("x-derivative (3-node):", dx3)
    
    dy3 = tri3.y(x_test, y_test)
    print("y-derivative (3-node):", dy3)
    
    # Test 6-node triangle
    tri6 = Tri(6)
    print("\n6-node triangle:", tri6)
    
    s6 = tri6(x_test, y_test)
    print("Shape functions (6-node):", s6)
    print("Sum check (should be 1):", np.sum(s6, axis=1))
    
    dx6 = tri6.x(x_test, y_test)
    print("x-derivative (6-node):", dx6)
    
    # Test second derivatives
    dxx6 = tri6.xx(x_test, y_test)
    print("xx-derivative (6-node):", dxx6)
    
    dxy6 = tri6.xy(x_test, y_test)
    print("xy-derivative (6-node):", dxy6)