import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def particlecursor(p, ind=None):
    """
    PARTICLECURSOR - Find surface elements of discretized particle surface.
    
    Parameters:
    -----------
    p : object
        Discretized particle surface with pos, verts, faces attributes
    ind : int or list, optional
        Index to face elements to be highlighted
        
    Output:
    -------
    Displays selected boundary elements (ind) or allows the user to select
    faces of the discretized particle surface via interactive selection.
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if ind is None:
        # Interactive mode - plot particle and enable clicking
        _plot_particle_interactive(ax, p)
        
        # Set up click event handler
        def on_click(event):
            if event.inaxes == ax:
                _handle_click(p, event, ax)
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.title("Click on red dots to select face elements")
        
    else:
        # Highlight specific indices
        if isinstance(ind, int):
            ind = [ind, ind]  # Convert single index to range
        elif len(ind) == 1:
            ind = [ind[0], ind[0]]
        
        _plot_particle_with_selection(ax, p, ind)
        plt.title(f"Highlighted face elements: {ind}")
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    
    plt.show()


def _plot_particle_interactive(ax, p):
    """Plot particle with interactive elements."""
    # Plot particle surface with blue edges
    _plot_particle_surface(ax, p, edge_color='blue', face_color='lightblue', alpha=0.6)
    
    # Plot centroid positions as red dots
    if hasattr(p, 'pos') and p.pos is not None:
        ax.scatter(p.pos[:, 0], p.pos[:, 1], p.pos[:, 2], 
                  c='red', s=50, picker=True)


def _plot_particle_with_selection(ax, p, selected_indices):
    """Plot particle with specific faces highlighted."""
    # Plot all faces with blue edges
    _plot_particle_surface(ax, p, edge_color='blue', face_color='lightblue', alpha=0.6)
    
    # Highlight selected faces in red
    if hasattr(p, 'faces') and hasattr(p, 'verts'):
        for idx in selected_indices:
            if 0 <= idx < len(p.faces):
                face = p.faces[idx]
                # Remove NaN values for triangular faces
                valid_vertices = face[~np.isnan(face)].astype(int)
                
                if len(valid_vertices) >= 3:
                    # Get vertex coordinates
                    face_verts = p.verts[valid_vertices]
                    
                    # Create face polygon
                    if len(valid_vertices) == 3:
                        # Triangle
                        triangle = np.vstack([face_verts, face_verts[0:1]])  # Close the triangle
                        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                               'r-', linewidth=3)
                    else:
                        # Quad or polygon
                        poly = np.vstack([face_verts, face_verts[0:1]])  # Close the polygon
                        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], 
                               'r-', linewidth=3)


def _plot_particle_surface(ax, p, edge_color='blue', face_color='lightblue', alpha=0.6):
    """Plot the particle surface."""
    if not (hasattr(p, 'faces') and hasattr(p, 'verts')):
        return
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    polygons = []
    for face in p.faces:
        # Remove NaN values for mixed triangle/quad meshes
        valid_indices = face[~np.isnan(face)].astype(int)
        
        if len(valid_indices) >= 3:
            # Get vertex coordinates for this face
            face_vertices = p.verts[valid_indices]
            polygons.append(face_vertices)
    
    if polygons:
        # Create 3D polygon collection
        collection = Poly3DCollection(polygons, alpha=alpha, 
                                    facecolors=face_color, 
                                    edgecolors=edge_color,
                                    linewidths=0.5)
        ax.add_collection3d(collection)


def _handle_click(p, event, ax):
    """Handle mouse click events for face selection."""
    if not (hasattr(p, 'pos') and event.xdata is not None and event.ydata is not None):
        return
    
    # Get click position (this is approximate for 3D)
    click_pos = np.array([event.xdata, event.ydata])
    
    # Project 3D positions to 2D screen coordinates (simplified approach)
    # In a full implementation, you'd use the actual projection matrix
    pos_2d = p.pos[:, :2]  # Simple projection to XY plane
    
    # Find closest point
    distances = np.sqrt(np.sum((pos_2d - click_pos)**2, axis=1))
    closest_idx = np.argmin(distances)
    
    # Check if click is close enough (within some threshold)
    if distances[closest_idx] < 0.1:  # Adjust threshold as needed
        print(f"Selected face element: {closest_idx}")
        
        # Highlight the selected face
        _highlight_face(ax, p, closest_idx)
        
        # Refresh plot
        plt.draw()


def _highlight_face(ax, p, face_idx):
    """Highlight a specific face element."""
    if not (hasattr(p, 'faces') and hasattr(p, 'verts')):
        return
    
    if 0 <= face_idx < len(p.faces):
        face = p.faces[face_idx]
        valid_vertices = face[~np.isnan(face)].astype(int)
        
        if len(valid_vertices) >= 3:
            face_verts = p.verts[valid_vertices]
            
            # Plot highlighted edges
            if len(valid_vertices) == 3:
                # Triangle
                triangle = np.vstack([face_verts, face_verts[0:1]])
                ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                       'r-', linewidth=4, label=f'Face {face_idx}')
            else:
                # Polygon
                poly = np.vstack([face_verts, face_verts[0:1]])
                ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], 
                       'r-', linewidth=4, label=f'Face {face_idx}')


# Example usage function
def demo_particlecursor():
    """Demonstrate particlecursor functionality with synthetic data."""
    # Create a simple cube particle for demonstration
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])
    
    faces = np.array([
        [0, 1, 2, 3],  # bottom
        [4, 7, 6, 5],  # top
        [0, 4, 5, 1],  # front
        [2, 6, 7, 3],  # back
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2]   # right
    ])
    
    # Calculate face centroids
    centroids = []
    for face in faces:
        face_verts = vertices[face]
        centroid = np.mean(face_verts, axis=0)
        centroids.append(centroid)
    
    # Create mock particle object
    class MockParticle:
        def __init__(self, verts, faces, pos):
            self.verts = verts
            self.faces = faces
            self.pos = np.array(pos)
            self.n = len(faces)
    
    p = MockParticle(vertices, faces, centroids)
    
    # Demo with highlighted face
    print("Demo: Highlighting face 0 and 1")
    particlecursor(p, [0, 1])


if __name__ == "__main__":
    demo_particlecursor()