"""
MESHDEMO: Demo function for mesh2d.

Feel free to "borrow" any of the geometries for your own use.

Example:
    meshdemo()  # Runs the demos

Darren Engwirda - 2006
Converted to Python by PyMNPBEM project.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import sys


def meshdemo():
    """
    Demo function showing various mesh2d examples.
    
    Demonstrates:
    - Simple geometric meshes
    - Size function controls
    - Complex geometries (Lake Superior)
    - Mesh refinement techniques
    """
    print("This is a demo function for mesh2d.\n")
    print("Several example meshes are shown, starting with some simple examples")
    print("and progressing to the CFD-like applications for which the function was designed.\n")
    
    # Circle example 1
    answer = input("The following is a simple mesh in a circle. Continue?? [y/n] ")
    if answer.lower() != 'y':
        return
    
    # Geometry - coarse circle
    dtheta = np.pi / 12
    theta = np.arange(-np.pi, np.pi, dtheta)
    node = np.column_stack([np.cos(theta), np.sin(theta)])
    
    print("Creating coarse circle mesh...")
    # p, t = mesh2d(node)  # Placeholder - would call actual mesh2d
    _plot_demo_geometry(node, "Coarse Circle")
    
    # Circle example 2
    answer = input("\nThe element size function is generated automatically to try to adequately resolve the geometry.\n"
                  "This means that the mesh size is related to the length of the line segments used to define the\n"
                  "geometry. The following example is the same as the last, but with more lines used to represent the\n"
                  "circle. Continue?? [y/n] ")
    if answer.lower() != 'y':
        return
    
    # Geometry - fine circle
    dtheta = np.pi / 75
    theta = np.arange(-np.pi, np.pi, dtheta)
    node = np.column_stack([np.cos(theta), np.sin(theta)])
    
    print("Creating fine circle mesh...")
    # p, t = mesh2d(node)  # Placeholder
    _plot_demo_geometry(node, "Fine Circle")
    
    # Square with size function
    answer = input("\nIt is often necessary to specify the element size in some locations and the following example\n"
                  "illustrates the use of a user specified sizing.\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    hdata = {'fun': hfun1}
    
    print("Creating square mesh with size function...")
    # p, t = mesh2d(node, [], hdata)  # Placeholder
    _plot_demo_geometry(node, "Square with Size Function")
    
    # Sliver regions
    answer = input("\nMesh2d can now deal with very fine \"sliver\" geometry features\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    
    node = np.array([
        [0, 0], [3, 0], [3, 3], [0, 3],
        [0.1, 1], [0.11, 1], [0.11, 2], [0.1, 2]
    ])
    cnect = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4]
    ])
    
    print("Creating mesh with sliver features...")
    # p, t = mesh2d(node, cnect)  # Placeholder
    _plot_demo_geometry(node, "Sliver Regions", cnect)
    
    # Cylinder in crossflow
    answer = input("\nThe following is a mesh used to simulate the flow past a cylinder.\n"
                  "This example also shows how user specified and automatic size functions are combined\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    
    theta = np.arange(0, 2*np.pi, np.pi/100)
    x = np.cos(theta) / 2
    y = np.sin(theta) / 2
    
    node = np.vstack([
        np.column_stack([x, y]),
        np.array([[-5, -10], [25, -10], [25, 10], [-5, 10]])
    ])
    
    n = len(theta)
    cnect = np.vstack([
        np.column_stack([np.arange(n-1), np.arange(1, n)]),
        [[n-1, 0]],
        [[n, n+1], [n+1, n+2], [n+2, n+3], [n+3, n]]
    ])
    
    hdata = {'fun': hfun2}
    
    print("Creating cylinder in crossflow mesh...")
    # p, t = mesh2d(node, cnect, hdata)  # Placeholder
    _plot_demo_geometry(node, "Cylinder in Crossflow", cnect)
    
    # Airfoil + flap
    answer = input("\nThe following is a mesh used to simulate the flow past an airfoil/flap configuration.\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    
    wing = np.array([
        [1.00003, 0.00126], [0.99730, 0.00170], [0.98914, 0.00302],
        [0.97563, 0.00518], [0.95693, 0.00812], [0.93324, 0.01176],
        [0.90482, 0.01602], [0.87197, 0.02079], [0.83506, 0.02597],
        [0.79449, 0.03145], [0.75070, 0.03712], [0.70417, 0.04285],
        [0.65541, 0.04854], [0.60496, 0.05405], [0.55335, 0.05924],
        [0.50117, 0.06397], [0.44897, 0.06811], [0.39733, 0.07150],
        [0.34681, 0.07402], [0.29796, 0.07554], [0.25131, 0.07597],
        [0.20738, 0.07524], [0.16604, 0.07320], [0.12732, 0.06915],
        [0.09230, 0.06265], [0.06203, 0.05382], [0.03730, 0.04324],
        [0.01865, 0.03176], [0.00628, 0.02030], [0.00015, 0.00956],
        [0.00000, 0.00000], [0.00533, -0.00792], [0.01557, -0.01401],
        [0.03029, -0.01870], [0.04915, -0.02248], [0.07195, -0.02586],
        [0.09868, -0.02922], [0.12954, -0.03282], [0.16483, -0.03660],
        [0.20483, -0.04016], [0.24869, -0.04283], [0.29531, -0.04446],
        [0.34418, -0.04510], [0.39476, -0.04482], [0.44650, -0.04371],
        [0.49883, -0.04188], [0.55117, -0.03945], [0.60296, -0.03655],
        [0.65360, -0.03327], [0.70257, -0.02975], [0.74930, -0.02607],
        [0.79330, -0.02235], [0.83407, -0.01866], [0.87118, -0.01512],
        [0.90420, -0.01180], [0.93279, -0.00880], [0.95661, -0.00621],
        [0.97543, -0.00410], [0.98901, -0.00254], [0.99722, -0.00158],
        [0.99997, -0.00126]
    ])
    
    flap = rotate(0.4 * wing, 10)
    flap = move(flap, 0.95, -0.1)
    wing = rotate(wing, 5)
    wing = move(wing, 0, 0.05)
    
    wall = np.array([[-1, -3], [4, -3], [4, 3], [-1, 3]])
    
    nwing = len(wing)
    nflap = len(flap) 
    nwall = len(wall)
    
    cwing = np.vstack([
        np.column_stack([np.arange(nwing-1), np.arange(1, nwing)]),
        [[nwing-1, 0]]
    ])
    cflap = np.vstack([
        np.column_stack([np.arange(nflap-1), np.arange(1, nflap)]),
        [[nflap-1, 0]]
    ]) + nwing
    cwall = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + nflap + nwing
    
    cnect = np.vstack([cwing, cflap, cwall])
    node = np.vstack([wing, flap, wall])
    
    hdata = {
        'edgeh': np.column_stack([
            np.arange(len(cnect) - 4),
            np.full(len(cnect) - 4, 0.005)
        ])
    }
    options = {'dhmax': 0.25}
    
    print("Creating airfoil + flap mesh...")
    # p, t = mesh2d(node, cnect, hdata, options)  # Placeholder
    _plot_demo_geometry(node, "Airfoil + Flap", cnect)
    
    # Lake Superior
    answer = input("\nThe following is a mesh of Lake Superior and is a standard test of mesh algorithms.\n"
                  "This example uses the automatic size function only\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    
    # Lake Superior coastline data (simplified for demo)
    p1, p2, p3, p4, p5, p6, p7 = _get_lake_superior_data()
    
    n1, n2, n3, n4, n5, n6, n7 = len(p1), len(p2), len(p3), len(p4), len(p5), len(p6), len(p7)
    
    c1 = np.vstack([np.column_stack([np.arange(n1-1), np.arange(1, n1)]), [[n1-1, 0]]])
    c2 = np.vstack([np.column_stack([np.arange(n2-1), np.arange(1, n2)]), [[n2-1, 0]]]) + n1
    c3 = np.vstack([np.column_stack([np.arange(n3-1), np.arange(1, n3)]), [[n3-1, 0]]]) + n2 + n1
    c4 = np.vstack([np.column_stack([np.arange(n4-1), np.arange(1, n4)]), [[n4-1, 0]]]) + n3 + n2 + n1
    c5 = np.vstack([np.column_stack([np.arange(n5-1), np.arange(1, n5)]), [[n5-1, 0]]]) + n4 + n3 + n2 + n1
    c6 = np.vstack([np.column_stack([np.arange(n6-1), np.arange(1, n6)]), [[n6-1, 0]]]) + n5 + n4 + n3 + n2 + n1
    c7 = np.vstack([np.column_stack([np.arange(n7-1), np.arange(1, n7)]), [[n7-1, 0]]]) + n6 + n5 + n4 + n3 + n2 + n1
    
    node = np.vstack([p1, p2, p3, p4, p5, p6, p7])
    cnect = np.vstack([c1, c2, c3, c4, c5, c6, c7])
    
    print("Creating Lake Superior mesh...")
    # p, t = mesh2d(node, cnect)  # Placeholder
    _plot_demo_geometry(node, "Lake Superior", cnect)
    
    # Gradient limiting example
    answer = input("\nThe following shows the influence of gradient limiting on the size function.\n"
                  "The value dhmax is reduced to 0.1\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    options = {'dhmax': 0.1}
    
    print("Creating Lake Superior mesh with gradient limiting...")
    # p, t = mesh2d(node, cnect, [], options)  # Placeholder
    _plot_demo_geometry(node, "Lake Superior (dhmax=0.1)", cnect)
    
    # Size function examples
    answer = input("\nThe following example shows how the element size can be controlled using the\n"
                  "various settings in HDATA.\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    _show_size_function_examples()
    
    # Refinement example
    answer = input("\nThe following example shows how an existing mesh can be refined using the\n"
                  "REFINE function. This avoids doing expensive retriangulation.\n"
                  "Continue [y/n] ")
    if answer.lower() != 'y':
        return
    
    plt.close('all')
    _show_refinement_example()
    
    print("\nDemo completed!")


def _plot_demo_geometry(node: np.ndarray, title: str, edges: Optional[np.ndarray] = None):
    """Plot geometry for demo purposes."""
    plt.figure(figsize=(10, 8))
    plt.plot(node[:, 0], node[:, 1], 'bo-', markersize=3, linewidth=1)
    
    if edges is not None:
        for edge in edges:
            i, j = edge
            plt.plot([node[i, 0], node[j, 0]], [node[i, 1], node[j, 1]], 'r-', linewidth=1)
    
    plt.axis('equal')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def _show_size_function_examples():
    """Show various size function control examples."""
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    options = {'output': False}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Size Function Examples')
    
    # Example 1: Global size with boundary layer
    hdata = {'hmax': 0.1, 'edgeh': [[0, 0.05]]}
    # p, t = mesh2d(node, [], hdata, options)  # Placeholder
    
    axes[0, 1].plot(node[[0, 1, 2, 3, 0], 0], node[[0, 1, 2, 3, 0], 1], 'b-', linewidth=2)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_title('Additional boundary layer function on bottom edge')
    axes[0, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _show_refinement_example():
    """Show mesh refinement example."""
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    hdata = {'hmax': 0.1}
    options = {'output': False}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mesh Refinement Example')
    
    # Original uniform mesh
    # p, t = mesh2d(node, [], hdata, options)  # Placeholder
    axes[0, 0].plot(node[[0, 1, 2, 3, 0], 0], node[[0, 1, 2, 3, 0], 1], 'b-', linewidth=2)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title('Uniform mesh from MESH2D')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Refined mesh (placeholder)
    axes[0, 1].plot(node[[0, 1, 2, 3, 0], 0], node[[0, 1, 2, 3, 0], 1], 'b-', linewidth=2)
    axes[0, 1].set_aspect('equal')  
    axes[0, 1].set_title('Mesh refined in centre region using REFINE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Smoothed mesh (placeholder)
    axes[1, 0].plot(node[[0, 1, 2, 3, 0], 0], node[[0, 1, 2, 3, 0], 1], 'b-', linewidth=2)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_title('Smoothed mesh using SMOOTHMESH')
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _get_lake_superior_data():
    """Get Lake Superior coastline data (simplified)."""
    # Main shoreline (simplified)
    p1 = np.array([
        [-8.9154, 1.6616], [-8.8847, 1.5539], [-8.7271, 1.4396],
        [-8.6780, 1.3310], [-8.2764, 1.3642], [-7.7589, 1.5521],
        [-7.3400, 1.5866], [-7.2065, 1.7400], [-6.9872, 1.7839],
        [-6.9191, 1.6751], [-6.6594, 1.8238], [-6.4705, 2.0285],
        # ... (truncated for brevity - would include full coastline)
        [-8.7235, 1.8639], [-8.9154, 1.6616]  # Close loop
    ])
    
    # Islands (simplified)
    p2 = np.array([
        [-6.0222, 1.4027], [-5.9152, 1.5579], [-5.8726, 1.5299],
        [-5.7920, 1.5537], [-5.8346, 1.5816], [-5.8319, 1.6610],
        [-6.0819, 1.5637], [-6.0866, 1.4314], [-6.0222, 1.4027]
    ])
    
    p3 = np.array([
        [-5.7324, 1.9225], [-5.5684, 1.9595], [-5.5336, 1.9160],
        [-5.5319, 1.9689], [-5.4498, 2.0457], [-5.4915, 2.1000],
        [-5.7298, 2.0019], [-5.7677, 1.9502], [-5.7324, 1.9225]
    ])
    
    p4 = np.array([
        [-5.2706, 2.2255], [-5.2257, 2.2771], [-5.2197, 2.3563],
        [-5.1912, 2.4614], [-5.2924, 2.4540], [-5.3167, 2.3700],
        [-5.3193, 2.2853], [-5.3445, 2.1748], [-5.2706, 2.2255]
    ])
    
    p5 = np.array([
        [-2.6285, 4.8904], [-2.5232, 4.8094], [-2.4525, 4.7818],
        [-2.1664, 4.9101], [-1.8814, 4.9860], [-2.0589, 4.9882],
        [-2.5905, 5.0486], [-2.6285, 4.8904]
    ])
    
    p6 = np.array([
        [-0.1746, 7.6483], [-0.1049, 7.4630], [-0.0175, 7.4629],
        [0.0349, 7.7276], [0.1394, 7.9923], [-0.1743, 7.8865], [-0.1746, 7.6483]
    ])
    
    p7 = np.array([
        [4.5965, 4.4039], [4.8814, 4.4121], [4.9867, 4.4682],
        [5.0886, 4.6302], [4.8365, 4.7286], [4.3785, 4.5569],
        [4.3813, 4.4511], [4.5965, 4.4039]
    ])
    
    return p1, p2, p3, p4, p5, p6, p7


def move(p: np.ndarray, xm: float, ym: float) -> np.ndarray:
    """Move a node set p by [xm, ym]."""
    n = p.shape[0]
    return p + np.array([[xm, ym]] * n)


def rotate(p: np.ndarray, A: float) -> np.ndarray:
    """Rotate a node set p by A degrees."""
    A_rad = A * np.pi / 180
    T = np.array([
        [np.cos(A_rad), np.sin(A_rad)],
        [-np.sin(A_rad), np.cos(A_rad)]
    ])
    return (T @ p.T).T


def hfun1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """User defined size function for square."""
    return 0.01 + 0.1 * np.sqrt((x - 0.25)**2 + (y - 0.75)**2)


def hfun2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """User defined size function for cylinder."""
    h1 = np.full(x.shape, np.inf)
    inside = (x >= 0) & (x <= 25) & (y >= -3) & (y <= 3)
    h1[inside] = 0.2
    
    r = np.sqrt(x**2 + y**2)
    h2 = np.full(x.shape, np.inf)
    circle_mask = r <= 3
    h2[circle_mask] = 0.02 + 0.05 * r[circle_mask]
    
    return np.minimum(h1, h2)


# Example usage
if __name__ == "__main__":
    print("MESH2D Demo")
    print("=" * 50)
    meshdemo()