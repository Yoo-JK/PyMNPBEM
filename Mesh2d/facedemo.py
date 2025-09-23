import numpy as np


def facedemo(n):
    """
    Example polygonal geometries for MESHFACES.
    
    Parameters:
    -----------
    n : int
        Demo number (1 or 2)
        
    Returns:
    --------
    node : ndarray
        Node coordinates
    edge : ndarray
        Edge connectivity  
    face : list
        Face definitions
    """
    
    if n == 1:
        node = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [1.01, 0.0], [1.01, 1.0], [3.0, 0.0], [3.0, 1.0]
        ])
        edge = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], [1, 4],
            [4, 5], [5, 2], [4, 6], [6, 7], [7, 5]
        ])
        face = [
            [0, 1, 2, 3],
            [4, 5, 6, 1], 
            [7, 8, 9, 5]
        ]
        return node, edge, face
        
    elif n == 2:
        # Geometry
        dtheta = np.pi / 36
        theta = np.arange(-np.pi, np.pi - dtheta, dtheta)
        node1 = np.column_stack([np.cos(theta), np.sin(theta)])
        node2 = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
        
        edge1 = np.column_stack([
            np.arange(len(node1)),
            np.concatenate([np.arange(1, len(node1)), [0]])
        ])
        edge2 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        
        edge = np.vstack([edge1, edge2 + len(node1)])
        node = np.vstack([node1, node2])
        
        face = [
            list(range(len(edge1))),
            list(range(len(edge)))
        ]
        return node, edge, face
    
    else:
        raise ValueError('Invalid demo. N must be between 1-2')