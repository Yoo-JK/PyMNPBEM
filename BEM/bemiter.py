"""
Iterative BEM Solver - Python Implementation

Base class for iterative BEM solvers.
Translated from MATLAB MNPBEM toolbox @bemiter class.
"""

import numpy as np
import scipy.sparse.linalg as spsolve
import time
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple, Dict, Any, List, Union
from ..Base.bembase import BemBase


class BemIter(BemBase):
    """
    Base class for iterative BEM solvers.
    
    This class provides iterative solvers for BEM equations using various
    iterative methods like GMRES, CGS, and BiCGSTAB.
    """
    
    # BemBase abstract properties
    name = 'bemiter'
    needs = ['iter']
    
    def __init__(self, *args, **kwargs):
        """
        Initialize BEM solver.
        
        Parameters:
        -----------
        *args : tuple
            Variable arguments
        **kwargs : dict
            PropertyName, PropertyValue pairs:
            - 'solver' : 'gmres', 'cgs', or 'bicgstab'
            - 'tol' : tolerance for iterative solver
            - 'maxit' : maximum number of iterations
            - 'restart' : restart for GMRES solver
            - 'precond' : None or 'hmat'
            - 'output' : intermediate output for iterative solver
        """
        # Public properties
        self.solver = 'gmres'      # iterative solver
        self.tol = 1e-4           # tolerance
        self.maxit = 200          # maximum number of iterations
        self.restart = None       # restart for GMRES solver
        self.precond = 'hmat'     # preconditioner for iterative solver
        self.output = 0           # intermediate output for iterative solver
        
        # Protected properties
        self._flag = []           # flag from iterative routine
        self._relres = []         # relative residual error
        self._iter = []           # number of iterations
        self._eneisav = []        # previously computed wavelengths
        self._stat = {}           # statistics for H-matrices
        self._timer = {}          # timer statistics
        
        # Initialize
        self._init(*args, **kwargs)
    
    def __str__(self):
        """String representation."""
        return f"BemIter(solver='{self.solver}', tol={self.tol}, maxit={self.maxit})"
    
    def __repr__(self):
        """Command window display."""
        info_dict = {
            'solver': self.solver,
            'tol': self.tol,
            'maxit': self.maxit,
            'restart': self.restart,
            'precond': self.precond,
            'output': self.output
        }
        return f"bemiter:\n{info_dict}"
    
    def _init(self, *args, **kwargs):
        """
        Initialize iterative BEM solver.
        
        Parameters:
        -----------
        *args : tuple
            Variable arguments (may include enei)
        **kwargs : dict
            Option fields
        """
        # Remove ENEI entry if present
        varargin = list(args)
        if varargin and isinstance(varargin[0], (int, float, complex)):
            varargin = varargin[1:]
        
        # Extract input options
        op = self._getbemoptions(['iter', 'bemiter'], *varargin, **kwargs)
        
        # Set options for iterative solver
        if 'solver' in op:
            self.solver = op['solver']
        if 'tol' in op:
            self.tol = op['tol']
        if 'maxit' in op:
            self.maxit = op['maxit']
        if 'restart' in op:
            self.restart = op['restart']
        if 'precond' in op:
            self.precond = op['precond']
        if 'output' in op:
            self.output = op['output']
        if 'hmode' in op:
            self.hmode = op['hmode']
    
    def _getbemoptions(self, default_keys: List[str], *args, **kwargs) -> Dict[str, Any]:
        """
        Extract BEM options from arguments.
        Simplified version - in real implementation this would be more complex.
        """
        # Combine all arguments into options dictionary
        op = {}
        
        # Add any dictionary arguments
        for arg in args:
            if isinstance(arg, dict):
                op.update(arg)
        
        # Add keyword arguments
        op.update(kwargs)
        
        return op
    
    @classmethod
    def options(cls, **kwargs) -> Dict[str, Any]:
        """
        Set options for iterative BEM solver.
        
        Parameters:
        -----------
        **kwargs : dict
            PropertyName, PropertyValue pairs:
            - 'solver' : 'gmres', 'cgs', or 'bicgstab'
            - 'tol' : tolerance for iterative solver
            - 'maxit' : maximum number of iterations
            - 'restart' : restart for GMRES solver
            - 'precond' : None or 'hmat'
            - 'output' : intermediate output for iterative solver
            - 'cleaf' : threshold parameter for bisection
            - 'fadmiss' : function for admissibility
            - 'htol' : tolerance for termination of aca loop
            - 'kmax' : maximum rank for low-rank matrices
            
        Returns:
        --------
        op : dict
            Options dictionary
        """
        # Set default values
        op = {
            'solver': 'gmres',
            'tol': 1e-6,
            'maxit': 100,
            'restart': None,
            'precond': 'hmat',
            'output': 0,
            'cleaf': 200,
            'htol': 1e-6,
            'kmax': [4, 100],
            'fadmiss': lambda rad1, rad2, dist: 2.5 * min(rad1, rad2) < dist
        }
        
        # Update with input options
        op.update(kwargs)
        
        return op
    
    def solve(self, x0: np.ndarray, b: np.ndarray, 
              afun: Callable, mfun: Optional[Callable] = None) -> Tuple[np.ndarray, 'BemIter']:
        """
        Iterative solution of BEM equations.
        
        Parameters:
        -----------
        x0 : ndarray
            Initial guess for solution vector
        b : ndarray
            Inhomogeneity, A * x = b
        afun : callable
            Function to evaluate A * x
        mfun : callable, optional
            Preconditioner, function to evaluate M * x
            
        Returns:
        --------
        x : ndarray
            Solution vector
        obj : BemIter
            Updated solver object with statistics
        """
        # Use only preconditioner?
        if self.maxit == 0 or not self.solver:
            # Make sure that preconditioner is set
            assert self.precond == 'hmat', "Preconditioner must be 'hmat'"
            # Compute solution vector
            x = mfun(b) if mfun else b
            return x, self
        
        # Iterative solution through SciPy functions
        if self.solver == 'cgs':
            # Conjugate gradient solver
            x, info = spsolve.cgs(
                afun, b, x0=x0, tol=self.tol, maxiter=self.maxit, M=mfun
            )
            flag = info
            # Calculate relative residual (simplified)
            relres = np.linalg.norm(afun(x) - b) / np.linalg.norm(b) if callable(afun) else 0
            iter_count = [self.maxit, 0]  # Simplified iteration count
            
        elif self.solver == 'bicgstab':
            # Biconjugate gradients stabilized method
            x, info = spsolve.bicgstab(
                afun, b, x0=x0, tol=self.tol, maxiter=self.maxit, M=mfun
            )
            flag = info
            relres = np.linalg.norm(afun(x) - b) / np.linalg.norm(b) if callable(afun) else 0
            iter_count = [self.maxit, 0]
            
        elif self.solver == 'gmres':
            # Generalized minimum residual method (with restarts)
            restart = self.restart if self.restart is not None else min(20, len(b))
            x, info = spsolve.gmres(
                afun, b, x0=x0, tol=self.tol, restart=restart, 
                maxiter=self.maxit, M=mfun
            )
            flag = info
            relres = np.linalg.norm(afun(x) - b) / np.linalg.norm(b) if callable(afun) else 0
            iter_count = [self.maxit, restart]
            
        else:
            raise ValueError(f"Iterative solver '{self.solver}' not known")
        
        # Save statistics
        self._setiter(flag, relres, iter_count)
        
        # Print statistics
        if self.output:
            self._printstat(flag, relres, iter_count)
        
        return x, self
    
    def _setiter(self, flag: int, relres: float, iter_count: List[int]):
        """
        Set statistics for iterative solver.
        
        Parameters:
        -----------
        flag : int
            Convergence flag
        relres : float
            Relative residual norm
        iter_count : list
            Outer and inner iteration numbers
        """
        if hasattr(self, 'enei'):
            if not (np.all(np.diff(np.append(self._eneisav, self.enei)) >= 0) or
                    np.all(np.diff(np.append(self._eneisav, self.enei)) <= 0)):
                # Reset statistics
                self._flag, self._relres, self._iter, self._eneisav = [], [], [], []
        
        # Make iter_count 2D
        if len(iter_count) == 1:
            iter_count = [iter_count[0], 0]
        
        # Save statistics
        self._flag.append(flag)
        self._relres.append(relres)
        self._iter.append(iter_count)
        if hasattr(self, 'enei'):
            self._eneisav.append(self.enei)
    
    def _printstat(self, flag: int, relres: float, iter_count: List[int]):
        """Print statistics for iterative solver."""
        if self.solver == 'cgs':
            print(f"cgs({self.maxit}), it={iter_count[0]:3d}, "
                  f"res={relres:10.4g}, flag={flag}")
        elif self.solver == 'bicgstab':
            print(f"bicgstab({self.maxit}), it={iter_count[0]:5.1f}, "
                  f"res={relres:10.4g}, flag={flag}")
        elif self.solver == 'gmres':
            print(f"gmres({self.maxit}), it={iter_count[1]:3d}({iter_count[0]}), "
                  f"res={relres:10.4g}, flag={flag}")
    
    def info(self) -> Union[None, Tuple[List, List, List]]:
        """
        Get information for iterative solver.
        
        Returns:
        --------
        flag : list
            Convergence flags
        relres : list
            Relative residual norms
        iter : list
            Outer and inner iteration numbers
            
        If no return values requested, prints statistics instead.
        """
        flag, relres, iter_vals = self._flag, self._relres, self._iter
        
        # Print statistics for each iteration
        for i in range(len(flag)):
            self._printstat(flag[i], relres[i], iter_vals[i])
        
        return flag, relres, iter_vals
    
    def hinfo(self):
        """Print information about H-matrices."""
        stat = self._stat
        
        # Empty statistics
        if 'compression' not in stat:
            return
        
        # Compression factors
        eta1, eta2 = [], []
        
        # Loop over fieldnames
        for name in stat['compression'].keys():
            if name in ['G', 'F', 'G1', 'H1', 'G2', 'H2']:
                eta1.extend(stat['compression'][name] if isinstance(stat['compression'][name], list) 
                           else [stat['compression'][name]])
            else:
                eta2.extend(stat['compression'][name] if isinstance(stat['compression'][name], list) 
                           else [stat['compression'][name]])
        
        if eta1:
            print(f"\nCompression Green functions        : {np.mean(eta1):8.6f}")
        if eta2:
            print(f"Compression auxiliary matrices     : {np.mean(eta2):8.6f}\n")
        
        # Timing
        if 'main' not in stat:
            return
        
        # Total time
        tall = np.sum(stat['main'])
        tsum = 0
        
        # Print total time
        print(f"Total time for H-matrix operations : {tall:9.3f} sec")
        
        for name in stat.keys():
            if name not in ['main', 'compression']:
                # Print percentage of time
                time_percent = 100 * np.sum(stat[name]) / tall
                print(f"  {name:12s} : {time_percent:6.2f} %")
                # Add time
                tsum += np.sum(stat[name])
        
        # Print rest percentage
        rest_percent = 100 * (1 - tsum / tall)
        print(f"  {'rest':12s} : {rest_percent:6.2f} %")
    
    def setstat(self, name: str, hmat):
        """
        Set statistics for H-matrices.
        
        Parameters:
        -----------
        name : str
            Name of H-matrix
        hmat : object
            Corresponding H-matrix object
        """
        stat = self._stat.copy()
        
        if not stat:
            stat = {'compression': {}}
        elif (hasattr(self, 'enei') and self._eneisav and
              not (np.all(np.diff(np.append(self._eneisav, self.enei)) >= 0) or
                   np.all(np.diff(np.append(self._eneisav, self.enei)) <= 0))):
            # Reset statistics
            stat = {'compression': {}}
        
        # Compression for H-matrix
        if name not in stat['compression']:
            stat['compression'][name] = self._get_compression(hmat)
        else:
            current = stat['compression'][name]
            if not isinstance(current, list):
                current = [current]
            current.append(self._get_compression(hmat))
            stat['compression'][name] = current
        
        # Timing
        if hasattr(hmat, 'stat') and hmat.stat:
            for stat_name, stat_value in hmat.stat.items():
                if stat_name not in stat:
                    stat[stat_name] = stat_value
                else:
                    if not isinstance(stat[stat_name], list):
                        stat[stat_name] = [stat[stat_name]]
                    if not isinstance(stat_value, list):
                        stat_value = [stat_value]
                    stat[stat_name].extend(stat_value)
        
        # Save statistics
        self._stat = stat
    
    def _get_compression(self, hmat) -> float:
        """
        Get compression factor from H-matrix.
        Placeholder implementation - would need actual H-matrix interface.
        """
        if hasattr(hmat, 'compression'):
            return hmat.compression()
        else:
            return 1.0  # Default compression factor
    
    def tocout(self, key: str, *varargin):
        """
        Intermediate output for BEM step.
        
        Parameters:
        -----------
        key : str
            Operation key ('init', 'close', or operation name)
        *varargin : tuple
            Additional arguments for initialization
        """
        if not self.output or not self.precond:
            return
        
        timer = self._timer
        
        if key == 'init':
            # Initialize structure (if needed)
            if not timer:
                timer = {
                    'names': list(varargin),
                    'toc': [],
                    'start_time': time.time()
                }
            timer['toc'].append([0] * len(timer['names']))
            
            # Initialize plot
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(range(len(timer['names'])), timer['toc'][-1], width=0.6)
            ax.set_xlim(-0.5, len(timer['names']) - 0.5)
            ax.set_ylim(0, max(max(timer['toc'][-1]) if timer['toc'][-1] else [0], 0.1))
            
            # Annotate plot
            ax.set_xticks(range(len(timer['names'])))
            ax.set_xticklabels(timer['names'], rotation=90)
            ax.set_ylabel('Elapsed time (sec)')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)
            
            # Save plot handles
            timer.update({
                'fig': fig,
                'ax': ax,
                'bars': bars,
                'last_time': time.time()
            })
            
        elif key == 'close':
            # Save total time
            total_time = time.time() - timer['start_time']
            timer['toc'][-1][-1] = total_time
            
            # Update plot
            if 'bars' in timer:
                for i, bar in enumerate(timer['bars']):
                    if i < len(timer['toc'][-1]):
                        bar.set_height(timer['toc'][-1][i])
                timer['ax'].set_ylim(0, max(timer['toc'][-1]))
                plt.draw()
            
            # Close figure
            if 'fig' in timer:
                plt.close(timer['fig'])
                
        else:
            # Save time for specific operation
            current_time = time.time()
            if key in timer['names']:
                idx = timer['names'].index(key)
                timer['toc'][-1][idx] = current_time - timer['last_time']
                
                # Update plot
                if 'bars' in timer and idx < len(timer['bars']):
                    timer['bars'][idx].set_height(timer['toc'][-1][idx])
                    timer['ax'].set_ylim(0, max(timer['toc'][-1]))
                    plt.draw()
                    plt.pause(0.01)
            
            timer['last_time'] = current_time
        
        # Save timer
        self._timer = timer