import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any
import gc
import time

class Mem:
    """
    Memory information monitoring class (used for testing).
    
    This class provides functionality to monitor Python memory usage
    similar to MATLAB's memory monitoring capabilities.
    
    Usage:
        Mem.on()
        Mem.set('key1')
        # ... some operations ...
        Mem.set('key2')
        # ... more operations ...
        Mem.off()
        print(Mem())
    """
    
    # Class variables to store global state
    _flag = False
    _report = []
    
    def __init__(self):
        """Initialize Mem object."""
        pass
    
    def __str__(self) -> str:
        """String representation showing memory report."""
        return self._format_report()
    
    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()
    
    @classmethod
    def _format_report(cls) -> str:
        """Format memory report for display."""
        if not cls._report:
            return "Memory monitoring: No data collected"
        
        lines = ["Memory Report:"]
        lines.append("-" * 50)
        lines.append(f"{'ID':<20} {'Memory (MB)':<15} {'Memory (GB)':<15}")
        lines.append("-" * 50)
        
        for entry in cls._report:
            id_str, memory_bytes = entry
            memory_mb = memory_bytes / (1024 * 1024)
            memory_gb = memory_bytes / (1024 * 1024 * 1024)
            lines.append(f"{id_str:<20} {memory_mb:<15.2f} {memory_gb:<15.4f}")
        
        return "\n".join(lines)
    
    @classmethod
    def on(cls):
        """Start reporting memory information."""
        try:
            # Test if psutil is available and working
            psutil.virtual_memory()
            cls._flag = True
            cls._report = []
            print("Memory monitoring started")
        except Exception:
            cls._flag = False
            print("Warning: Memory monitoring not available (psutil required)")
    
    @classmethod
    def off(cls):
        """Stop reporting memory information."""
        cls._flag = False
        print("Memory monitoring stopped")
    
    @classmethod
    def clear(cls):
        """Clear memory report."""
        cls._report = []
        print("Memory report cleared")
    
    @classmethod
    def set(cls, identifier: str = ""):
        """
        Set memory information with string identifier.
        
        Parameters:
        -----------
        identifier : str
            String identifier for this memory checkpoint
        """
        if cls._flag:
            try:
                # Force garbage collection for more accurate measurement
                gc.collect()
                
                # Get current memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                
                # Use RSS (Resident Set Size) as it's closest to MATLAB's MemAvailableAllArrays
                memory_used = memory_info.rss
                
                cls._report.append([identifier if identifier else f"checkpoint_{len(cls._report)}", 
                                  memory_used])
                
            except Exception as e:
                print(f"Warning: Could not collect memory information: {e}")
    
    @classmethod
    def flag(cls) -> bool:
        """Get monitoring flag status."""
        return cls._flag
    
    @classmethod
    def report(cls) -> List[List]:
        """Get report data as list of [identifier, memory] pairs."""
        return cls._report.copy()
    
    @classmethod
    def plot(cls):
        """Plot memory information."""
        if not cls._flag:
            print("Memory monitoring is not active")
            return
        
        if not cls._report:
            print("No memory data to plot")
            return
        
        # Extract identifiers and memory values
        identifiers = [entry[0] for entry in cls._report]
        memory_values = np.array([entry[1] for entry in cls._report])
        
        # Convert to GB and compute relative memory usage
        memory_gb = memory_values / (1024**3)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(memory_gb)), memory_gb)
        
        # Customize plot
        ax.set_xlim(-0.5, len(memory_gb) - 0.5)
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Memory Usage Report')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(identifiers)))
        ax.set_xticklabels(identifiers, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, mem_val) in enumerate(zip(bars, memory_gb)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem_val:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Add custom cursor functionality
        def on_hover(event):
            if event.inaxes == ax:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        ax.set_title(f'Memory Usage Report - {identifiers[i]}: {memory_gb[i]:.3f} GB')
                        fig.canvas.draw_idle()
                        break
                else:
                    ax.set_title('Memory Usage Report')
                    fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        plt.show()
    
    @classmethod
    def get_system_memory(cls) -> dict:
        """
        Get comprehensive system memory information.
        
        Returns:
        --------
        dict
            Dictionary with memory information including total, available, 
            used, and percentage
        """
        try:
            virtual_mem = psutil.virtual_memory()
            process = psutil.Process()
            process_mem = process.memory_info()
            
            return {
                'total_system_memory_gb': virtual_mem.total / (1024**3),
                'available_system_memory_gb': virtual_mem.available / (1024**3),
                'system_memory_percent': virtual_mem.percent,
                'process_memory_mb': process_mem.rss / (1024**2),
                'process_memory_gb': process_mem.rss / (1024**3),
                'virtual_memory_mb': process_mem.vms / (1024**2),
                'virtual_memory_gb': process_mem.vms / (1024**3)
            }
        except Exception as e:
            return {'error': f'Could not retrieve memory information: {e}'}
    
    @classmethod
    def memory_diff(cls, start_idx: int = 0, end_idx: int = -1) -> dict:
        """
        Calculate memory difference between two checkpoints.
        
        Parameters:
        -----------
        start_idx : int
            Index of starting checkpoint
        end_idx : int  
            Index of ending checkpoint (-1 for last)
            
        Returns:
        --------
        dict
            Dictionary with memory difference information
        """
        if len(cls._report) < 2:
            return {'error': 'Need at least 2 checkpoints for comparison'}
        
        if end_idx == -1:
            end_idx = len(cls._report) - 1
        
        if start_idx >= len(cls._report) or end_idx >= len(cls._report):
            return {'error': 'Index out of range'}
        
        start_mem = cls._report[start_idx][1]
        end_mem = cls._report[end_idx][1]
        diff_bytes = end_mem - start_mem
        
        return {
            'start_checkpoint': cls._report[start_idx][0],
            'end_checkpoint': cls._report[end_idx][0],
            'start_memory_mb': start_mem / (1024**2),
            'end_memory_mb': end_mem / (1024**2),
            'difference_mb': diff_bytes / (1024**2),
            'difference_gb': diff_bytes / (1024**3),
            'percent_change': (diff_bytes / start_mem) * 100 if start_mem > 0 else 0
        }
    
    @classmethod
    def peak_memory(cls) -> dict:
        """
        Find peak memory usage from recorded checkpoints.
        
        Returns:
        --------
        dict
            Dictionary with peak memory information
        """
        if not cls._report:
            return {'error': 'No memory data available'}
        
        memory_values = [entry[1] for entry in cls._report]
        peak_idx = np.argmax(memory_values)
        peak_memory = memory_values[peak_idx]
        
        return {
            'peak_checkpoint': cls._report[peak_idx][0],
            'peak_memory_mb': peak_memory / (1024**2),
            'peak_memory_gb': peak_memory / (1024**3),
            'peak_index': peak_idx
        }


# Convenience function for quick memory check
def memory_snapshot(label: str = "snapshot") -> dict:
    """
    Take a quick memory snapshot without using the monitoring system.
    
    Parameters:
    -----------
    label : str
        Label for this snapshot
        
    Returns:
    --------
    dict
        Memory information
    """
    try:
        gc.collect()
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'label': label,
            'timestamp': time.time(),
            'process_memory_mb': memory_info.rss / (1024**2),
            'process_memory_gb': memory_info.rss / (1024**3),
            'system_available_gb': virtual_mem.available / (1024**3),
            'system_percent': virtual_mem.percent
        }
    except Exception as e:
        return {'error': f'Could not take memory snapshot: {e}'}