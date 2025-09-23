import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from ..bemoptions import get_bemoptions
from ..misc import ipart


class ClusterTree:
    """
    Build cluster tree through bisection.
    
    See S. Boerm et al., Eng. Analysis with Bound. Elem. 27, 405 (2003).
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize cluster tree.
        
        Parameters
        ----------
        p : object
            Compound of particles
        *args, **kwargs : optional
            Options and property pairs
            
        Properties
        ----------
        cleaf : int, optional
            Threshold parameter for bisection
        """
        self.p = None
        self.son = None
        self.mid = None
        self.rad = None
        self.ind = None
        self.cind = None
        self.ipart = None
        self._init(*args, **kwargs)
    
    def __str__(self):
        """Command window display."""
        info = {
            'p': self.p,
            'son': self.son
        }
        return f"ClusterTree:\n{info}"
    
    def _init(self, p, *args, **kwargs):
        """Build cluster tree through bisection."""
        # Extract input
        op = get_bemoptions(['iter', 'hoptions'], *args, **kwargs)
        
        # Get values
        cleaf = getattr(op, 'cleaf', 32)
        
        # Save particle
        self.p = p
        
        # Set up tree
        # Index to particle positions and cluster index
        ind = np.arange(p.n)
        cind = np.arange(p.n)
        
        # Bounding box for particle
        pos_min = np.min(p.pos, axis=0)
        pos_max = np.max(p.pos, axis=0)
        box = np.vstack([pos_min, pos_max])
        
        # Center position and radius
        box_dict = {
            'mid': np.mean(box, axis=0),
            'rad': 0.5 * np.linalg.norm(box[1, :] - box[0, :])
        }
        
        # Empty tree
        tree = [{'cind': [0, p.n - 1], 'box': box_dict}]
        
        # Split tree by bisection
        tree = self._bisection(p, tree, 0, ind, cind, cleaf)
        
        # Extract information from tree
        self.son = np.zeros((len(tree), 2), dtype=int)
        
        # Sons of tree
        for i, node in enumerate(tree):
            if 'son1' in node:
                self.son[i, :] = [node['son1'], node['son2']]
        
        # Cluster index
        self.cind = np.array([node['cind'] for node in tree])
        
        # Extract bounding boxes
        self.mid = np.array([node['box']['mid'] for node in tree])
        self.rad = np.array([node['box']['rad'] for node in tree])
        
        # Conversion between particle index and cluster index
        # Find leaves of tree
        leaf_mask = ['ind' in node for node in tree]
        
        # Index and cluster index for leaves
        ind_leaves = [tree[i]['ind'] for i in range(len(tree)) if leaf_mask[i]]
        cind_leaves = [np.arange(tree[i]['cind'][0], tree[i]['cind'][1] + 1) 
                      for i in range(len(tree)) if leaf_mask[i]]
        
        # Convert to arrays
        ind_flat = np.concatenate(ind_leaves)
        cind_flat = np.concatenate(cind_leaves)
        
        # Sort indices
        i1 = np.argsort(ind_flat)
        i2 = np.argsort(cind_flat)
        
        # Conversion arrays
        self.ind = np.column_stack([
            ind_flat[i2].reshape(-1),
            cind_flat[i1].reshape(-1)
        ])
        
        # Particle index
        ip1 = ipart(p, self.ind[self.cind[:, 0], 0])
        ip2 = ipart(p, self.ind[self.cind[:, 1], 0])
        
        # Zero for composite particles
        ip1[ip1 != ip2] = 0
        
        # Save particle index
        self.ipart = ip1
    
    def _bisection(self, p, tree, ic, ind, cind, cleaf):
        """Construction of cluster tree by bisection."""
        siz = len(tree)
        
        # Add sons to parent node
        tree[ic]['son1'] = siz
        tree[ic]['son2'] = siz + 1
        
        # Split cluster
        son1, son2, ind1, ind2, cind1, cind2, psplit = self._split(p, ind, cind)
        
        # Add new nodes to tree
        tree.append(son1)
        tree.append(son2)
        
        # Further splitting of cluster 1?
        if len(ind1) > cleaf or psplit:
            tree = self._bisection(p, tree, siz, ind1, cind1, cleaf)
        else:
            tree[siz]['ind'] = ind1
        
        # Further splitting of cluster 2?
        if len(ind2) > cleaf or psplit:
            tree = self._bisection(p, tree, siz + 1, ind2, cind2, cleaf)
        else:
            tree[siz + 1]['ind'] = ind2
        
        return tree
    
    def _split(self, p, ind, cind):
        """Split cluster."""
        # Try particle split
        ind1, ind2 = self._partsplit(p, ind)
        psplit = len(ind1) > 0
        
        # Bisection split?
        if len(ind1) == 0:
            ind1, ind2 = self._bisplit(p, ind)
        
        # Cluster index
        cind1 = cind[:len(ind1)]
        cind2 = cind[len(ind1):]
        
        # Sons
        son1 = {
            'cind': [cind1[0], cind1[-1]],
            'box': self._sph_boundary(p, ind1)
        }
        son2 = {
            'cind': [cind2[0], cind2[-1]], 
            'box': self._sph_boundary(p, ind2)
        }
        
        return son1, son2, ind1, ind2, cind1, cind2, psplit
    
    def _partsplit(self, p, ind):
        """Split cluster into different particle boundaries."""
        ip = ipart(p, ind)
        unique_parts = np.unique(ip)
        
        if len(unique_parts) == 1:
            return np.array([]), np.array([])
        else:
            ind1 = ind[ip == unique_parts[0]]
            ind2 = np.setdiff1d(ind, ind1)
            return ind1, ind2
    
    def _bisplit(self, p, ind):
        """Split cluster by bisection."""
        # Bounding box
        box = self._boundary(p, ind)
        
        # Split direction
        k = np.argmax(box[1, :] - box[0, :])
        
        # Split position
        mid = box[0, k] + 0.5 * (box[1, k] - box[0, k])
        
        # Split cluster
        ind1 = ind[p.pos[ind, k] < mid]
        ind2 = np.setdiff1d(ind, ind1)
        
        return ind1, ind2
    
    def _boundary(self, p, ind):
        """Boundary box for cluster."""
        pos_min = np.min(p.pos[ind, :], axis=0)
        pos_max = np.max(p.pos[ind, :], axis=0)
        return np.vstack([pos_min, pos_max])
    
    def _sph_boundary(self, p, ind):
        """Sphere boundary for cluster."""
        box = self._boundary(p, ind)
        return {
            'mid': np.mean(box, axis=0),
            'rad': 0.5 * np.linalg.norm(box[1, :] - box[0, :])
        }
    
    def admissibility(self, obj2, *args, **kwargs):
        """Admissibility matrix."""
        op = get_bemoptions(*args, hoptions=True, **kwargs)
        
        # Function for admissibility condition
        if hasattr(op, 'fadmiss'):
            fadmiss = op.fadmiss
        else:
            fadmiss = lambda rad1, rad2, dist: 2.5 * min(rad1, rad2) < dist
        
        # Allocate matrix
        mat = csr_matrix((self.son.shape[0], obj2.son.shape[0]))
        
        # Build block tree
        mat = self._blocktree(mat, self, obj2, 0, 0, fadmiss)
        
        return mat
    
    def _admissible(self, obj2, i1, i2, fadmiss):
        """Check for admissibility."""
        if self.son[i1, 0] == 0 and obj2.son[i2, 0] == 0:
            # Leaf
            return 2
        else:
            # Particle indices
            ip1 = ipart(self.p, self.ind[self.cind[i1, 0]:self.cind[i1, 1]+1, 1])
            ip2 = ipart(obj2.p, obj2.ind[obj2.cind[i2, 0]:obj2.cind[i2, 1]+1, 1])
            
            # Particle indices unique?
            if len(np.unique(ip1)) > 1 or len(np.unique(ip2)) > 1:
                return 0
            else:
                # Check for admissibility
                dist = np.linalg.norm(self.mid[i1, :] - obj2.mid[i2, :])
                return int(fadmiss(self.rad[i1], obj2.rad[i2], dist))
    
    def _index(self, i):
        """Cluster index for leaves."""
        if self.son[i, 0] == 0:
            return [i]
        else:
            return self.son[i, :].tolist()
    
    def _blocktree(self, mat, obj1, obj2, i1, i2, fadmiss):
        """Build block tree for matrix."""
        ad = obj1._admissible(obj2, i1, i2, fadmiss)
        
        # Check for admissibility
        if ad:
            mat[i1, i2] = ad
        else:
            ind1 = obj1._index(i1)
            ind2 = obj2._index(i2)
            
            # Loop over sons
            for ii1 in ind1:
                for ii2 in ind2:
                    mat = self._blocktree(mat, obj1, obj2, ii1, ii2, fadmiss)
        
        return mat
    
    def matindex(self, obj2, i1, i2):
        """Index for cluster tree sub-matrix."""
        # Cluster indices
        row_range = np.arange(self.cind[i1, 0], self.cind[i1, 1] + 1)
        col_range = np.arange(obj2.cind[i2, 0], obj2.cind[i2, 1] + 1)
        
        row, col = np.meshgrid(self.ind[row_range, 0], obj2.ind[col_range, 0], indexing='ij')
        
        # Matrix index
        ind = np.ravel_multi_index((row.ravel(), col.ravel()), self.matsize(obj2))
        
        # Size and number of elements
        siz = row.shape
        n = row.size
        
        return ind, siz, n
    
    def matsize(self, obj2):
        """Size for cluster tree matrix."""
        return [self.ind.shape[0], obj2.ind.shape[0]]
    
    def cluster2part(self, v):
        """Conversion from cluster index to particle index."""
        return v[self.ind[:, 1], :].reshape(v.shape)
    
    def part2cluster(self, v):
        """Conversion from particle index to cluster index."""
        return v[self.ind[:, 0], :].reshape(v.shape)
    
    def plot(self):
        """Plot cluster tree."""
        # This would implement tree plotting functionality
        # For now, just a placeholder
        print("Tree plotting not implemented in Python version")
    
    def plotadmiss(self, *args, **kwargs):
        """Plot admissibility matrix."""
        # Allocate array
        mat = np.zeros(self.matsize(self), dtype=np.uint16)
        
        # Find full and low-rank matrices
        ad = self.admissibility(self, *args, **kwargs)
        
        # This would implement the plotting functionality
        # For now, just a placeholder
        print("Admissibility plotting not implemented in Python version")
        
        return mat
    
    def plotcluster(self, key='single'):
        """Plot clusters of cluster tree."""
        # This would implement cluster plotting functionality
        # For now, just a placeholder
        print("Cluster plotting not implemented in Python version")