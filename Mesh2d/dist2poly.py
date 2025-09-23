import numpy as np


def dist2poly(p, edgexy, lim=None):
    """
    Find the minimum distance from the points in P to the polygon defined by
    the edges in EDGEXY. LIM is an optional argument that defines an upper
    bound on the distance for each point.
    """
    
    p = np.asarray(p, dtype=float)
    edgexy = np.asarray(edgexy, dtype=float)
    
    np_pts = p.shape[0]
    ne = edgexy.shape[0]
    
    if lim is None:
        lim = np.full(np_pts, np.inf)
    else:
        lim = np.asarray(lim, dtype=float)
    
    # Choose the direction with the biggest range as the "y-coordinate"
    dxy = np.max(p, axis=0) - np.min(p, axis=0)
    if dxy[0] > dxy[1]:
        # Flip co-ords if x range is bigger
        p = p[:, [1, 0]]
        edgexy = edgexy[:, [1, 0, 3, 2]]
    
    # Ensure edgexy[:,[0,1]] contains the lower y value
    swap = edgexy[:, 3] < edgexy[:, 1]
    edgexy[swap, :] = edgexy[swap, [2, 3, 0, 1]]
    
    # Sort edges
    sort_idx = np.argsort(edgexy[:, 1])  # Sort by lower y value
    edgexy_lower = edgexy[sort_idx, :]
    sort_idx = np.argsort(edgexy[:, 3])  # Sort by upper y value
    edgexy_upper = edgexy[sort_idx, :]
    
    # Mean edge y value
    ymean = 0.5 * np.sum(edgexy[:, [1, 3]]) / ne
    
    # Alloc output
    L = np.zeros(np_pts)
    
    # Tolerance
    tol = 1000.0 * np.finfo(float).eps * np.max(dxy)
    
    # Loop through points
    for k in range(np_pts):
        x = p[k, 0]
        y = p[k, 1]
        d = lim[k]
        
        if y < ymean:
            # Loop through edges bottom up
            for j in range(ne):
                y2 = edgexy_lower[j, 3]
                if y2 >= (y - d):
                    y1 = edgexy_lower[j, 1]
                    if y1 <= (y + d):
                        x1 = edgexy_lower[j, 0]
                        x2 = edgexy_lower[j, 2]
                        
                        xmin = min(x1, x2)
                        xmax = max(x1, x2)
                        
                        if xmin <= (x + d) and xmax >= (x - d):
                            # Calculate distance along normal projection
                            x2mx1 = x2 - x1
                            y2my1 = y2 - y1
                            
                            r = ((x - x1) * x2mx1 + (y - y1) * y2my1) / (x2mx1**2 + y2my1**2)
                            r = np.clip(r, 0.0, 1.0)  # Limit to wall endpoints
                            
                            dj = (x1 + r * x2mx1 - x)**2 + (y1 + r * y2my1 - y)**2
                            if (dj < d**2) and (dj > tol):
                                d = np.sqrt(dj)
                    else:
                        break
        else:
            # Loop through edges top down
            for j in range(ne - 1, -1, -1):
                y1 = edgexy_upper[j, 1]
                if y1 <= (y + d):
                    y2 = edgexy_upper[j, 3]
                    if y2 >= (y - d):
                        x1 = edgexy_upper[j, 0]
                        x2 = edgexy_upper[j, 2]
                        
                        xmin = min(x1, x2)
                        xmax = max(x1, x2)
                        
                        if xmin <= (x + d) and xmax >= (x - d):
                            # Calculate distance along normal projection
                            x2mx1 = x2 - x1
                            y2my1 = y2 - y1
                            
                            r = ((x - x1) * x2mx1 + (y - y1) * y2my1) / (x2mx1**2 + y2my1**2)
                            r = np.clip(r, 0.0, 1.0)  # Limit to wall endpoints
                            
                            dj = (x1 + r * x2mx1 - x)**2 + (y1 + r * y2my1 - y)**2
                            if (dj < d**2) and (dj > tol):
                                d = np.sqrt(dj)
                    else:
                        break
        
        L[k] = d
    
    return L