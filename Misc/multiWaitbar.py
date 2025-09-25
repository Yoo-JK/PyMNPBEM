import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Dict, Optional, Callable, Union, Any
import numpy as np

class MultiWaitbar:
    """
    MultiWaitbar: add, remove or update an entry on the multi waitbar
    
    This class provides a progress bar system with multiple entries that can be
    updated independently, similar to MATLAB's multiWaitbar function.
    """
    
    _instance = None
    _window = None
    _entries = {}
    _busy_timer = None
    _is_running = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def multiwaitbar(cls, label: str, *args, **kwargs) -> bool:
        """
        Add, remove or update an entry on the multi waitbar.
        
        Parameters:
        -----------
        label : str
            Label for the progress bar entry
        *args : various
            Can be a value (0-1) or commands
        **kwargs : dict
            Command-value pairs for changing the waitbar entry
            
        Supported commands:
        - 'Value': Set value (0-1)
        - 'Increment': Increment value
        - 'Color': Change color (RGB tuple or color name)
        - 'Relabel': Change label
        - 'Reset': Reset to zero
        - 'CanCancel': Enable/disable cancel button
        - 'CancelFcn': Set cancel callback function
        - 'ResetCancel': Reset cancelled flag
        - 'Close': Remove entry
        - 'Busy': Put in busy mode
        
        Returns:
        --------
        bool
            Whether the entry has been cancelled
        """
        instance = cls()
        return instance._process_command(label, *args, **kwargs)
    
    def _process_command(self, label: str, *args, **kwargs) -> bool:
        """Process the waitbar command."""
        # Handle special commands
        if label.upper() in ['CLOSEALL', 'CLOSE ALL']:
            self._close_all()
            return False
        
        # Ensure window exists
        if self._window is None:
            self._create_window()
        
        # Parse arguments
        if len(args) == 1 and isinstance(args[0], (int, float)):
            # Simple value update
            kwargs['Value'] = args[0]
        
        # Get or create entry
        if label not in self._entries:
            self._entries[label] = self._create_entry(label)
        
        entry = self._entries[label]
        cancel_status = entry.get('Cancel', False)
        
        # Process commands
        for key, value in kwargs.items():
            self._process_single_command(label, key, value)
        
        # Update display
        self._update_display()
        
        return cancel_status
    
    def _create_window(self):
        """Create the main progress window."""
        self._window = tk.Toplevel()
        self._window.title("Progress")
        self._window.geometry("360x42")
        self._window.resizable(True, True)
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Create main frame
        self.main_frame = ttk.Frame(self._window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Start busy timer
        self._start_busy_timer()
    
    def _create_entry(self, label: str) -> Dict[str, Any]:
        """Create a new progress entry."""
        entry = {
            'Label': label,
            'Value': 0.0,
            'LastValue': float('inf'),
            'Color': '#CC0011',  # Default red color
            'Created': time.time(),
            'CanCancel': False,
            'Cancel': False,
            'CancelFcn': None,
            'Busy': False,
            'widgets': {}
        }
        
        # Create GUI elements
        frame = ttk.Frame(self.main_frame)
        
        # Label
        label_var = tk.StringVar(value=label)
        label_widget = ttk.Label(frame, textvariable=label_var)
        label_widget.grid(row=0, column=0, sticky=tk.W, padx=2)
        
        # ETA label
        eta_var = tk.StringVar(value="")
        eta_widget = ttk.Label(frame, textvariable=eta_var)
        eta_widget.grid(row=0, column=1, sticky=tk.E, padx=2)
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_widget = ttk.Progressbar(frame, variable=progress_var, 
                                        maximum=1.0, length=300)
        progress_widget.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)
        
        # Cancel button (initially hidden)
        cancel_button = ttk.Button(frame, text="✖", width=3, 
                                 command=lambda: self._cancel_entry(label))
        
        # Configure grid weights
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        
        entry['widgets'] = {
            'frame': frame,
            'label_var': label_var,
            'label_widget': label_widget,
            'eta_var': eta_var,
            'eta_widget': eta_widget,
            'progress_var': progress_var,
            'progress_widget': progress_widget,
            'cancel_button': cancel_button
        }
        
        return entry
    
    def _process_single_command(self, label: str, command: str, value: Any):
        """Process a single command for an entry."""
        entry = self._entries[label]
        command_upper = command.upper()
        
        if command_upper == 'VALUE':
            entry['LastValue'] = entry['Value']
            entry['Value'] = max(0.0, min(1.0, float(value)))
            entry['Busy'] = False
            
        elif command_upper in ['INCREMENT', 'INC']:
            entry['LastValue'] = entry['Value']
            entry['Value'] = max(0.0, min(1.0, entry['Value'] + float(value)))
            entry['Busy'] = False
            
        elif command_upper in ['COLOR', 'COLOUR']:
            entry['Color'] = self._parse_color(value)
            
        elif command_upper in ['RELABEL', 'UPDATELABEL']:
            if not isinstance(value, str):
                raise ValueError("Value for 'Relabel' must be a string")
            if value in self._entries and value != label:
                raise ValueError("Cannot relabel to existing label")
            entry['Label'] = value
            entry['widgets']['label_var'].set(value)
            
        elif command_upper == 'CANCANCEL':
            if not isinstance(value, str) or value.lower() not in ['on', 'off']:
                raise ValueError("Parameter 'CanCancel' must be 'on' or 'off'")
            entry['CanCancel'] = (value.lower() == 'on')
            entry['Cancel'] = False
            
        elif command_upper == 'RESETCANCEL':
            entry['Cancel'] = False
            
        elif command_upper == 'CANCELFCN':
            if not callable(value):
                raise ValueError("Parameter 'CancelFcn' must be callable")
            entry['CancelFcn'] = value
            entry['CanCancel'] = True
            
        elif command_upper == 'RESET':
            entry['Value'] = 0.0
            entry['LastValue'] = float('inf')
            entry['Created'] = time.time()
            entry['Busy'] = False
            
        elif command_upper == 'BUSY':
            entry['Busy'] = True
            entry['Value'] = 0.0
            
        elif command_upper in ['CLOSE', 'DONE']:
            self._remove_entry(label)
            
    def _parse_color(self, color: Union[str, tuple, list]) -> str:
        """Parse color specification into hex string."""
        if isinstance(color, str):
            color_map = {
                'r': '#CC0000', 'g': '#00CC00', 'b': '#0000CC',
                'c': '#00CCCC', 'm': '#CC00CC', 'y': '#CCCC00',
                'k': '#000000', 'w': '#FFFFFF'
            }
            return color_map.get(color.lower(), color)
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            r, g, b = [int(c * 255) if c <= 1 else int(c) for c in color]
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            raise ValueError("Invalid color specification")
    
    def _cancel_entry(self, label: str):
        """Handle cancel button press."""
        if label in self._entries:
            entry = self._entries[label]
            entry['Cancel'] = True
            
            # Call user cancel function if provided
            if entry['CancelFcn']:
                try:
                    entry['CancelFcn'](label, 'Cancelled')
                except Exception as e:
                    print(f"Error in cancel function: {e}")
    
    def _remove_entry(self, label: str):
        """Remove an entry from the display."""
        if label in self._entries:
            entry = self._entries[label]
            entry['widgets']['frame'].destroy()
            del self._entries[label]
        
        # Close window if no entries left
        if not self._entries and self._window:
            self._close_all()
    
    def _update_display(self):
        """Update the visual display of all entries."""
        if not self._window:
            return
        
        row = 0
        for label, entry in self._entries.items():
            widgets = entry['widgets']
            frame = widgets['frame']
            
            # Pack frame
            frame.grid(row=row, column=0, sticky=tk.EW, pady=2)
            row += 1
            
            # Update progress bar
            if entry['Busy']:
                widgets['progress_widget'].config(mode='indeterminate')
                if not widgets['progress_widget'].cget('mode') == 'indeterminate':
                    widgets['progress_widget'].start(10)
            else:
                widgets['progress_widget'].config(mode='determinate')
                widgets['progress_widget'].stop()
                widgets['progress_var'].set(entry['Value'])
            
            # Update label with percentage
            if not entry['Busy']:
                percentage = int(entry['Value'] * 100)
                label_text = f"{entry['Label']} ({percentage}%)"
            else:
                label_text = entry['Label']
            widgets['label_var'].set(label_text)
            
            # Update ETA
            eta_text = self._calculate_eta(entry)
            widgets['eta_var'].set(eta_text)
            
            # Handle cancel button
            if entry['CanCancel']:
                widgets['cancel_button'].grid(row=1, column=2, padx=2)
            else:
                widgets['cancel_button'].grid_remove()
        
        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        
        # Adjust window size
        self._adjust_window_size()
    
    def _calculate_eta(self, entry: Dict) -> str:
        """Calculate estimated time of arrival."""
        if entry['Busy'] or entry['Value'] <= 0:
            return ""
        
        elapsed = time.time() - entry['Created']
        if elapsed < 3:  # Don't show ETA for first 3 seconds
            return ""
        
        remaining = elapsed * (1 - entry['Value']) / entry['Value']
        
        if remaining > 172800:  # > 2 days
            return f"{int(remaining/86400)} days"
        elif remaining > 7200:  # > 2 hours
            return f"{int(remaining/3600)} hours"
        elif remaining > 120:   # > 2 minutes
            return f"{int(remaining/60)} mins"
        elif remaining > 1:
            return f"{int(remaining)} secs"
        elif remaining >= 1:
            return "1 sec"
        else:
            return ""
    
    def _adjust_window_size(self):
        """Adjust window size based on number of entries."""
        if not self._window:
            return
        
        num_entries = len(self._entries)
        if num_entries == 0:
            return
        
        # Calculate required height
        height_per_entry = 60  # Approximate height per entry
        border = 20
        min_height = 60
        
        required_height = max(min_height, border + num_entries * height_per_entry)
        
        # Get current size
        current_width = self._window.winfo_width()
        if current_width < 100:  # Window not yet realized
            current_width = 360
        
        # Set new size
        self._window.geometry(f"{current_width}x{required_height}")
    
    def _start_busy_timer(self):
        """Start the timer for busy animations."""
        if not self._is_running:
            self._is_running = True
            self._busy_timer_thread = threading.Thread(target=self._busy_timer_loop, daemon=True)
            self._busy_timer_thread.start()
    
    def _busy_timer_loop(self):
        """Timer loop for updating busy animations."""
        while self._is_running and self._window:
            try:
                # Check if any entries are busy
                has_busy = any(entry.get('Busy', False) for entry in self._entries.values())
                
                if has_busy:
                    # Update busy entries
                    for entry in self._entries.values():
                        if entry.get('Busy', False):
                            widgets = entry['widgets']
                            progress = widgets['progress_widget']
                            if progress.cget('mode') != 'indeterminate':
                                progress.config(mode='indeterminate')
                                progress.start(10)
                
                time.sleep(0.02)  # 50 FPS
            except Exception:
                break
    
    def _on_close(self):
        """Handle window close event."""
        self._window.withdraw()  # Hide instead of destroy
    
    def _close_all(self):
        """Close all entries and destroy window."""
        self._is_running = False
        
        # Clear all entries
        for entry in list(self._entries.values()):
            entry['widgets']['frame'].destroy()
        self._entries.clear()
        
        # Destroy window
        if self._window:
            self._window.destroy()
            self._window = None


# Convenience function that matches MATLAB interface
def multiwaitbar(label: str, *args, **kwargs) -> bool:
    """
    MATLAB-style interface for multiWaitbar.
    
    Examples:
    --------
    multiwaitbar('CloseAll')
    multiwaitbar('Task 1', 0)
    multiwaitbar('Task 2', 0.5, Color='b')
    multiwaitbar('Task 1', Value=0.1)
    multiwaitbar('Task 2', Increment=0.2)
    multiwaitbar('Task 1', 'Close')
    """
    return MultiWaitbar.multiwaitbar(label, *args, **kwargs)


# Example usage
if __name__ == "__main__":
    import tkinter as tk
    
    # Create root window (required for tkinter)
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    # Example usage
    multiwaitbar('CloseAll')
    multiwaitbar('Task 1', 0)
    multiwaitbar('Task 2', Value=0.5, Color='b')
    multiwaitbar('Task 3', Busy=True)
    
    # Simulate progress
    for i in range(100):
        multiwaitbar('Task 1', Value=i/100)
        multiwaitbar('Task 2', Increment=0.01)
        time.sleep(0.05)
    
    multiwaitbar('Task 1', 'Close')
    multiwaitbar('Task 2', 'Close')
    multiwaitbar('Task 3', 'Close')