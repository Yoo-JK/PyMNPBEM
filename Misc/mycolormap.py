import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def mycolormap(key, n=100):
    """
    MYCOLORMAP - Load custom colormaps.
    Taken from http://hem.bredband.net/aditus/chunkhtml/ch22s08.html
    
    Parameters:
    -----------
    key : str
        Colormap key:
        - 'std:1' to 'std:10': standard maps
        - 'cen:1' to 'cen:5': centered maps  
        - 'con:1' to 'con:7': continuous maps
        - 'demo:std': show all standard maps
        - 'demo:cen': show all centered maps
        - 'demo:con': show all continuous maps
    n : int, optional
        Number of colors (default: 100)
        
    Returns:
    --------
    np.ndarray or None
        Colormap array (n x 3) or None if demo mode
    """
    
    # Define colormap data (RGB values 0-255)
    colormaps = {
        # Standard maps
        'std:1': np.array([
            [0, 0, 0], [69, 0, 0], [139, 0, 0], [197, 82, 0],
            [247, 154, 0], [255, 204, 0], [255, 249, 0], [255, 255, 111],
            [255, 255, 239]
        ]),
        'std:2': np.array([
            [0, 0, 0], [47, 47, 47], [95, 95, 95], [142, 142, 142],
            [184, 184, 184], [204, 204, 204], [220, 220, 220],
            [236, 236, 236], [252, 252, 252]
        ]),
        'std:3': np.array([
            [139, 0, 0], [255, 0, 0], [255, 165, 0], [255, 255, 0],
            [31, 255, 0], [0, 31, 223], [0, 0, 153], [65, 0, 131],
            [217, 113, 224]
        ]),
        'std:4': np.array([
            [0, 0, 128], [0, 0, 191], [0, 0, 255], [0, 0, 127], [0, 0, 15],
            [111, 0, 0], [239, 0, 0], [204, 0, 0], [146, 0, 0]
        ]),
        'std:5': np.array([
            [0, 0, 128], [0, 0, 191], [0, 0, 255], [127, 127, 127],
            [239, 239, 15], [255, 143, 0], [255, 15, 0], [204, 0, 0],
            [146, 0, 0]
        ]),
        'std:6': np.array([
            [0, 100, 0], [0, 177, 0], [0, 255, 0], [0, 127, 0], [0, 15, 0],
            [111, 0, 0], [239, 0, 0], [204, 0, 0], [146, 0, 0]
        ]),
        'std:7': np.array([
            [0, 100, 0], [0, 177, 0], [0, 255, 0], [127, 255, 0],
            [239, 255, 0], [255, 143, 0], [255, 15, 0], [204, 0, 0],
            [146, 0, 0]
        ]),
        'std:8': np.array([
            [0, 0, 139], [0, 0, 197], [0, 0, 255], [0, 0, 127], [0, 0, 15],
            [0, 111, 0], [0, 239, 0], [0, 187, 0], [0, 109, 0]
        ]),
        'std:9': np.array([
            [0, 0, 139], [0, 0, 197], [0, 0, 255], [127, 127, 127],
            [239, 239, 15], [143, 255, 0], [15, 255, 0], [0, 187, 0],
            [0, 109, 0]
        ]),
        'std:10': np.array([
            [0, 0, 255], [0, 50, 127], [0, 100, 0], [0, 177, 0],
            [0, 245, 0], [111, 143, 0], [239, 15, 0], [204, 0, 0],
            [146, 0, 0]
        ]),
        
        # Centered maps
        'cen:1': np.array([
            [179, 88, 6], [224, 130, 20], [253, 184, 99], [254, 224, 182],
            [247, 247, 247], [219, 221, 236], [182, 176, 213], [134, 122, 176],
            [89, 48, 140]
        ]),
        'cen:2': np.array([
            [140, 81, 10], [191, 129, 45], [223, 194, 125], [246, 232, 195],
            [245, 245, 245], [204, 235, 231], [136, 208, 197], [62, 157, 149],
            [7, 108, 100]
        ]),
        'cen:3': np.array([
            [178, 24, 43], [214, 96, 77], [244, 165, 130], [253, 219, 199],
            [247, 247, 247], [213, 231, 240], [153, 201, 224], [76, 153, 198],
            [37, 107, 174]
        ]),
        'cen:4': np.array([
            [178, 24, 43], [214, 96, 77], [244, 165, 130], [253, 219, 199],
            [255, 255, 255], [227, 227, 227], [190, 190, 190], [141, 141, 141],
            [84, 84, 84]
        ]),
        'cen:5': np.array([
            [215, 48, 39], [244, 109, 67], [253, 174, 97], [254, 224, 139],
            [255, 255, 223], [221, 241, 149], [172, 219, 110], [110, 192, 99],
            [35, 156, 82]
        ]),
        
        # Continuous maps
        'con:1': np.array([
            [255, 247, 251], [236, 231, 242], [208, 209, 230], [166, 189, 219],
            [122, 171, 208], [61, 147, 193], [11, 116, 178], [4, 92, 145],
            [2, 60, 94]
        ]),
        'con:2': np.array([
            [247, 252, 253], [229, 245, 249], [204, 236, 230], [153, 216, 201],
            [108, 196, 168], [69, 176, 123], [38, 143, 75], [4, 112, 47],
            [0, 73, 29]
        ]),
        'con:3': np.array([
            [255, 255, 217], [237, 248, 176], [199, 233, 180], [127, 205, 187],
            [72, 184, 194], [33, 149, 192], [33, 100, 171], [36, 57, 150],
            [11, 31, 95]
        ]),
        'con:4': np.array([
            [255, 247, 236], [254, 232, 200], [253, 212, 158], [253, 187, 132],
            [252, 146, 94], [240, 106, 74], [218, 54, 36], [183, 6, 3],
            [133, 0, 0]
        ]),
        'con:5': np.array([
            [255, 255, 204], [255, 237, 160], [254, 217, 118], [254, 178, 76],
            [253, 145, 62], [252, 85, 44], [230, 32, 29], [193, 3, 36],
            [177, 0, 38]
        ]),
        'con:6': np.array([
            [255, 255, 229], [255, 247, 188], [254, 227, 145], [254, 196, 79],
            [254, 158, 45], [238, 117, 22], [208, 80, 4], [159, 55, 3],
            [108, 38, 5]
        ]),
        'con:7': np.array([
            [247, 252, 253], [224, 236, 244], [191, 211, 230], [158, 188, 218],
            [142, 154, 200], [140, 112, 179], [136, 70, 159], [129, 21, 128],
            [83, 1, 81]
        ])
    }
    
    # Handle demo cases
    if key.startswith('demo:'):
        _show_demo(key, colormaps)
        return None
    
    # Get the colormap
    if key not in colormaps:
        raise ValueError(f"Colormap '{key}' not known")
    
    map_data = colormaps[key]
    
    # Interpolate to desired number of colors
    # Create interpolation points (0 to 1)
    x_orig = np.linspace(0, 1, len(map_data))
    x_new = np.linspace(0, 1, n)
    
    # Interpolate each color channel
    map_interp = np.zeros((n, 3))
    for i in range(3):
        f = interp1d(x_orig, map_data[:, i], kind='linear')
        map_interp[:, i] = f(x_new)
    
    # Convert to 0-1 range
    map_interp = map_interp / 255.0
    
    return map_interp


def _show_demo(demo_key, colormaps):
    """Show demo of colormap categories."""
    category = demo_key.split(':')[1]
    
    if category == 'std':
        keys = [f'std:{i}' for i in range(1, 11)]
        title = 'Standard Colormaps'
    elif category == 'cen':
        keys = [f'cen:{i}' for i in range(1, 6)]
        title = 'Centered Colormaps'
    elif category == 'con':
        keys = [f'con:{i}' for i in range(1, 8)]
        title = 'Continuous Colormaps'
    else:
        raise ValueError(f"Unknown demo category: {category}")
    
    # Create figure
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 10))
    fig.suptitle(title, fontsize=16)
    
    if len(keys) == 1:
        axes = [axes]
    
    for i, key in enumerate(keys):
        if key in colormaps:
            # Get colormap
            cmap = mycolormap(key, 100)
            
            # Create colorbar-like display
            gradient = np.linspace(0, 1, 100).reshape(1, -1)
            
            # Display the colormap
            im = axes[i].imshow(gradient, aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(cmap))
            axes[i].set_title(key)
            axes[i].set_yticks([])
            axes[i].set_xticks([])
    
    plt.tight_layout()
    plt.show()


# Convenience function to apply colormap to current plot
def apply_colormap(key, n=100):
    """
    Apply custom colormap to current matplotlib plot.
    
    Parameters:
    -----------
    key : str
        Colormap key
    n : int, optional
        Number of colors
    """
    cmap_data = mycolormap(key, n)
    if cmap_data is not None:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_data)
        plt.colormap(cmap)


# Example usage
if __name__ == "__main__":
    # Show demos
    mycolormap('demo:std')
    mycolormap('demo:cen') 
    mycolormap('demo:con')
    
    # Get a specific colormap
    cmap = mycolormap('std:1', 256)
    print(f"Colormap shape: {cmap.shape}")
    
    # Example usage with matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.random.randn(50, 50)
    im = ax.imshow(data, cmap=plt.matplotlib.colors.ListedColormap(mycolormap('con:4')))
    plt.colorbar(im)
    plt.title('Example with con:4 colormap')
    plt.show()