import os
import sys
import inspect
import importlib
import pkgutil
from typing import List

def subclasses(classname: str, str_filter: str = 'MNPBEM') -> List[str]:
    """
    SUBCLASSES - Find names of subclasses for superclass CLASSNAME.
    
    Parameters:
    -----------
    classname : str
        Name of superclass
    str_filter : str, optional
        Keep only modules with this string in path (default: 'MNPBEM')
        
    Returns:
    --------
    list
        Names of derived classes
        
    Note:
    -----
    This function scans through Python modules in the path and finds
    classes that inherit from the specified classname.
    """
    sub = []
    
    # Get all modules in sys.path
    paths_to_search = []
    for path in sys.path:
        if str_filter in path and os.path.isdir(path):
            paths_to_search.append(path)
    
    # Also search current working directory and its subdirectories
    for root, dirs, files in os.walk('.'):
        if str_filter in root:
            paths_to_search.append(root)
    
    # Search through modules
    for search_path in paths_to_search:
        try:
            # Walk through directory structure
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        try:
                            # Construct module name
                            module_path = os.path.join(root, file)
                            rel_path = os.path.relpath(module_path, search_path)
                            module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                            
                            # Try to import and inspect the module
                            spec = importlib.util.spec_from_file_location(module_name, module_path)
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                # Find classes in module
                                for name, obj in inspect.getmembers(module, inspect.isclass):
                                    if obj.__module__ == module_name:
                                        # Check if it's a subclass of classname
                                        try:
                                            # Get the base class by name
                                            if hasattr(obj, '__bases__'):
                                                for base in obj.__mro__[1:]:  # Skip self
                                                    if base.__name__ == classname:
                                                        if name not in sub:
                                                            sub.append(name)
                                                        break
                                        except (AttributeError, TypeError):
                                            continue
                        except (ImportError, SyntaxError, AttributeError, OSError):
                            # Skip modules that can't be imported or inspected
                            continue
        except (OSError, PermissionError):
            # Skip paths that can't be accessed
            continue
    
    return sub


def find_subclasses_by_inspection(base_class_name: str) -> List[str]:
    """
    Alternative implementation using runtime inspection.
    
    This version looks for already imported classes that inherit from
    the specified base class.
    """
    subclass_names = []
    
    # Get all currently loaded modules
    for module_name, module in sys.modules.items():
        if module is None:
            continue
            
        try:
            # Inspect all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's defined in this module (not imported)
                if hasattr(obj, '__module__') and obj.__module__ == module_name:
                    # Check inheritance
                    for base in obj.__mro__[1:]:  # Skip the class itself
                        if base.__name__ == base_class_name:
                            if name not in subclass_names:
                                subclass_names.append(name)
                            break
        except (AttributeError, TypeError):
            continue
    
    return subclass_names