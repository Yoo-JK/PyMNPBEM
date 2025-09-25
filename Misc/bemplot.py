import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import copy

class BemPlot:
    """
    Plotting value arrays and vector functions within PyMNPBEM.
    
    BemPlot allows to plot value and vector functions, and to page through
    multidimensional arrays. It provides context menu functionality and
    interactive plotting capabilities.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize BemPlot object.
        
        Parameters:
        -----------
        fun : callable, optional
            Plot function (default: real part)
        scale : float, optional  
            Scale factor for vector array (default: 1)
        sfun : callable, optional
            Scale function for vector array (default: identity)
        """
        self.var = []  # Value and vector arrays for plotting
        self.siz = None  # Size for paging plots
        self.opt = {
            'ind': None,  # Index for paging plots
            'fun': lambda x: np.real(x),  # Plot function
            'scale': 1,  # Scale factor for vector array
            'sfun': lambda x: x  # Scale function for vector array
        }
        
        # Current figure and axes
        self.fig = None
        self.ax = None
        self.toolbar = None
        
        # Initialize with provided options
        self._init(**kwargs)
    
    def _init(self, **kwargs):
        """Initialize BemPlot object with options."""
        # Set default values if not provided
        if 'fun' in kwargs:
            self.opt['fun'] = kwargs['fun']
        if 'scale' in kwargs:
            self.opt['scale'] = kwargs['scale'] 
        if 'sfun' in kwargs:
            self.opt['sfun'] = kwargs['sfun']
            
        # Set up figure if needed
        if plt.get_fignums() == [] or self.fig is None:
            self._setup_figure()
        
        self._set_figure_name()
    
    def _setup_figure(self):
        """Set up figure with proper axes and context menu."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis properties
        self.ax.set_aspect('equal')
        self.ax.view_init(elev=40, azim=1)
        
        # Add context menu functionality
        self._setup_context_menu()
        
        plt.tight_layout()
    
    def _setup_context_menu(self):
        """Set up context menu for the figure."""
        def on_right_click(event):
            if event.button == 3:  # Right click
                self._show_context_menu(event)
        
        self.fig.canvas.mpl_connect('button_press_event', on_right_click)
    
    def _show_context_menu(self, event):
        """Show context menu with plot options."""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Real", command=lambda: self._set_function(np.real))
        menu.add_command(label="Imaginary", command=lambda: self._set_function(np.imag))
        menu.add_command(label="Absolute", command=lambda: self._set_function(np.abs))
        menu.add_separator()
        menu.add_command(label="Plot Options", command=self._show_options_dialog)
        menu.add_command(label="Tight Color Axis", command=self._set_tight_caxis)
        
        try:
            menu.tk_popup(event.x, event.y)
        finally:
            menu.grab_release()
    
    def _set_function(self, func):
        """Set the plot function."""
        self.opt['fun'] = func
        self.refresh('fun')
    
    def _show_options_dialog(self):
        """Show dialog for plot options."""
        root = tk.Tk()
        root.withdraw()
        
        # Get current values
        ind_str = str(self.opt['ind']) if self.opt['ind'] is not None else ""
        fun_str = self.opt['fun'].__name__ if hasattr(self.opt['fun'], '__name__') else "custom"
        scale_str = str(self.opt['scale'])
        sfun_str = self.opt['sfun'].__name__ if hasattr(self.opt['sfun'], '__name__') else "custom"
        
        # Input dialogs
        try:
            new_ind = simpledialog.askstring("Index", "Enter the index:", initialvalue=ind_str)
            new_scale = simpledialog.askstring("Scale", "Enter the scale parameter:", initialvalue=scale_str)
            
            kwargs = {}
            if new_ind and new_ind != ind_str:
                kwargs['ind'] = eval(new_ind) if new_ind else None
            if new_scale and new_scale != scale_str:
                kwargs['scale'] = float(new_scale)
                
            if kwargs:
                self.set(**kwargs)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        finally:
            root.destroy()
    
    def _set_tight_caxis(self):
        """Set tight color axis based on data."""
        if not self.var:
            return
            
        # Get min/max values from all variables
        all_mins = []
        all_maxs = []
        
        for var in self.var:
            if hasattr(var, 'get_minmax'):
                vmin, vmax = var.get_minmax(self.opt)
                all_mins.append(vmin)
                all_maxs.append(vmax)
        
        if all_mins and all_maxs:
            vmin = min(all_mins)
            vmax = max(all_maxs)
            
            # Find all images/collections and set color limits
            for child in self.ax.get_children():
                if hasattr(child, 'set_clim'):
                    child.set_clim(vmin, vmax)
            
            plt.draw()
    
    def _set_figure_name(self):
        """Set name of figure based on current options."""
        func = self.opt['fun']
        
        if func == np.real:
            name = "(real)"
        elif func == np.imag:
            name = "(imag)"
        elif func == np.abs:
            name = "(abs)"
        else:
            name = "(fun)"
        
        if self.siz is not None and self.opt['ind'] is not None:
            # Convert linear index to subscript
            if np.isscalar(self.opt['ind']):
                indices = np.unravel_index(self.opt['ind'] - 1, self.siz)  # MATLAB uses 1-based indexing
                name = f"Element {list(indices)} of {list(self.siz)} {name}"
        
        if self.fig:
            self.fig.canvas.manager.set_window_title(name)
    
    def _init_paging(self):
        """Set up figure for paging through multidimensional arrays."""
        if self.siz is None:
            return
            
        # Create navigation buttons
        if hasattr(self.fig.canvas, 'toolbar'):
            # Add custom buttons to toolbar
            self._add_paging_buttons()
        
        self._set_figure_name()
    
    def _add_paging_buttons(self):
        """Add previous/next buttons for paging."""
        # This would be implementation-specific based on matplotlib backend
        # For now, we'll use keyboard shortcuts
        def on_key(event):
            if event.key == 'left':
                self._page_down()
            elif event.key == 'right':
                self._page_up()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
    
    def _page_down(self):
        """Page down through arrays."""
        if self.opt['ind'] is None or self.opt['ind'] <= 1:
            return
        
        self.opt['ind'] -= 1
        self.refresh('ind')
        self._set_figure_name()
    
    def _page_up(self):
        """Page up through arrays."""
        if self.opt['ind'] is None or self.siz is None:
            return
        
        if self.opt['ind'] >= np.prod(self.siz):
            return
        
        self.opt['ind'] += 1
        self.refresh('ind')
        self._set_figure_name()
    
    @classmethod
    def get(cls, **kwargs):
        """Get BemPlot object from current figure or create new one."""
        # Check if current figure has BemPlot object
        if plt.get_fignums() and hasattr(plt.gcf(), '_bemplot_obj'):
            obj = plt.gcf()._bemplot_obj
            obj._init(**kwargs)
            return obj
        else:
            obj = cls(**kwargs)
            if obj.fig:
                obj.fig._bemplot_obj = obj
            return obj
    
    def set(self, **kwargs):
        """
        Set properties of BemPlot object.
        
        Parameters:
        -----------
        ind : int or tuple
            Index for paging plots
        fun : callable
            Plot function
        scale : float
            Scale factor for vector function
        sfun : callable
            Scale function for vector functions
        """
        refresh_keys = []
        
        # Handle index
        if 'ind' in kwargs:
            if isinstance(kwargs['ind'], (list, tuple)) and self.siz is not None:
                # Convert multidimensional index to linear index
                self.opt['ind'] = np.ravel_multi_index(kwargs['ind'], self.siz) + 1  # MATLAB 1-based
            else:
                self.opt['ind'] = kwargs['ind']
            refresh_keys.append('ind')
        
        # Handle other options
        for key in ['fun', 'scale', 'sfun']:
            if key in kwargs:
                self.opt[key] = kwargs[key]
                refresh_keys.append(key)
        
        try:
            # Refresh affected plots
            if refresh_keys:
                self.refresh(*refresh_keys)
        except Exception as e:
            if self.fig:
                messagebox.showerror("Error", "Error in plot options, no update of figure")
            raise e
    
    def plot(self, p, inifun, inifun2, **kwargs):
        """
        Plot value or vector array.
        
        Parameters:
        -----------
        p : object
            Particle surface or vector positions
        inifun : callable
            First initialization of value or vector array
        inifun2 : callable
            Re-initialization of value or vector array
        **kwargs : dict
            Additional arguments passed to plot function
        """
        # Initialize value array
        var = inifun(p)
        
        # Handle size argument of value array
        if hasattr(var, 'is_page') and var.is_page() and self.siz is not None:
            assert var.page_size() == self.siz, "Page sizes must match"
        
        # Check if particle has been plotted before
        ind = None
        for i, existing_var in enumerate(self.var):
            if hasattr(existing_var, 'is_base') and existing_var.is_base(p):
                ind = i
                break
        
        if ind is None:
            # Save value array
            self.var.append(var)
            ind = len(self.var) - 1
        else:
            # Update existing value array
            self.var[ind] = inifun2(self.var[ind])
        
        # Handle paging setup
        if hasattr(self.var[ind], 'is_page') and self.var[ind].is_page() and self.siz is None:
            self.siz = self.var[ind].page_size()
            self.opt['ind'] = 1
            self._init_paging()
        
        # Plot the array
        if hasattr(self.var[ind], 'plot'):
            self.var[ind] = self.var[ind].plot(self.opt, **kwargs)
        
        # Set lighting and shading
        if self.ax and hasattr(self.ax, 'zaxis'):
            # Add lighting effects if supported
            pass
        
        # Update figure
        if self.fig:
            self.fig._bemplot_obj = self
            plt.draw()
    
    def plotval(self, p, val, **kwargs):
        """Plot value array on surface."""
        from .valarray import ValArray  # Import here to avoid circular imports
        
        inifun = lambda p: ValArray(p, val)
        inifun2 = lambda var: var.init2(val)
        
        self.plot(p, inifun, inifun2, **kwargs)
    
    def plottrue(self, p, val=None, **kwargs):
        """Plot value array with true colors on surface."""
        from .valarray import ValArray
        
        inifun = lambda p: ValArray(p, val, mode='truecolor')
        inifun2 = lambda var: var.init2(val, mode='truecolor')
        
        self.plot(p, inifun, inifun2, **kwargs)
    
    def plotarrow(self, pos, vec, **kwargs):
        """Plot vector array with arrows."""
        from .vecarray import VecArray
        
        inifun = lambda pos: VecArray(pos, vec, mode='arrow')
        inifun2 = lambda var: var.init2(vec, mode='arrow')
        
        self.plot(pos, inifun, inifun2, **kwargs)
    
    def plotcone(self, pos, vec, **kwargs):
        """Plot vector array with cones."""
        from .vecarray import VecArray
        
        inifun = lambda pos: VecArray(pos, vec, mode='cone')
        inifun2 = lambda var: var.init2(vec, mode='cone')
        
        self.plot(pos, inifun, inifun2, **kwargs)
    
    def refresh(self, *keys):
        """
        Refresh value and vector plots.
        
        Parameters:
        -----------
        *keys : str
            Keys to refresh ('ind', 'fun', 'scale', 'sfun')
        """
        # Find arrays that depend on the given keys
        affected_indices = []
        for i, var in enumerate(self.var):
            if hasattr(var, 'depends') and var.depends(*keys):
                affected_indices.append(i)
        
        # Update affected arrays
        for i in affected_indices:
            if hasattr(self.var[i], 'plot'):
                self.var[i] = self.var[i].plot(self.opt)
        
        # Update figure name
        self._set_figure_name()
        
        # Redraw
        if self.fig:
            plt.draw()
    
    def __str__(self):
        """String representation."""
        return f"BemPlot(var={len(self.var)} arrays, siz={self.siz}, opt={self.opt})"
    
    def __repr__(self):
        """String representation."""
        return self.__str__()