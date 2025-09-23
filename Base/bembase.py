"""
MNPBEM Base Classes - Python Implementation

Abstract base class for MNPBEM classes.
By providing a task name and a list of attributes, one can access
different classes by setting option parameters.

Example:
    name = 'bemsolver'
    needs = [{'sim': 'stat'}, 'nev']

Translated from MATLAB MNPBEM toolbox.
"""

from abc import ABC, abstractmethod
import inspect
import sys
from typing import List, Union, Dict, Any, Optional, Type


class BemBase(ABC):
    """
    Abstract base class for MNPBEM classes.
    
    By providing a task name and a list of attributes, one can access
    different classes by setting option parameters.
    
    A possible problem of the present implementation is that it is not
    possible to use BEMBASE classes as superclasses for other BEMBASE
    classes, since the class attributes cannot be easily changed by the derived
    class in the same way as MATLAB's Constant properties.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task name - must be implemented by subclasses"""
        pass
    
    @property
    @abstractmethod
    def needs(self) -> List[Union[str, Dict[str, Any]]]:
        """
        List of fieldnames that must be provided or set in the option structure.
        Can contain strings (field names) or dictionaries (field name: required value pairs).
        """
        pass
    
    @classmethod
    def find(cls, name: str, op: Dict[str, Any], **kwargs) -> Optional[Type['BemBase']]:
        """
        Select classname from subclasses of BEMBASE using option structure.
        
        Parameters:
        -----------
        name : str
            Task name
        op : dict
            Option structure/dictionary
        **kwargs : dict
            Additional fields added to op
            
        Returns:
        --------
        class_type : Type[BemBase] or None
            Class type among subclasses of BEMBASE where all needs are fulfilled
            Returns None if no suitable class found
        """
        # Get list of all BEMBASE subclasses
        subclass_list = cls._get_subclasses()
        
        # Number of NEEDS agreements with options
        n = [0] * len(subclass_list)
        
        # Add keyword arguments to option dictionary
        op_copy = op.copy()
        op_copy.update(kwargs)
        
        # Loop through subclass list
        for i, subclass in enumerate(subclass_list):
            # Check if class name matches requested name
            if hasattr(subclass, 'name') and subclass.name == name:
                # Get needs of class
                if hasattr(subclass, 'needs'):
                    needs = subclass.needs.copy()  # Make a copy to avoid modifying original
                    
                    # Loop over NEEDS
                    agreements = []
                    for j, need in enumerate(needs):
                        if isinstance(need, str):
                            # Simple field existence check
                            agreement = (need in op_copy and 
                                       op_copy[need] is not None and
                                       op_copy[need] != '')
                        else:
                            # Dictionary - check field name and value match
                            if isinstance(need, dict) and len(need) == 1:
                                fname = list(need.keys())[0]  # Get fieldname from dictionary
                                required_value = need[fname]
                                # Compare with op entries
                                agreement = (fname in op_copy and 
                                           op_copy[fname] == required_value)
                            else:
                                agreement = False
                        
                        agreements.append(agreement)
                    
                    # Count agreements
                    if all(agreements):
                        n[i] = len(needs)
        
        # Find maximum number of agreements
        if not n or max(n) == 0:
            return None
        
        max_idx = n.index(max(n))
        return subclass_list[max_idx]
    
    @classmethod
    def _get_subclasses(cls) -> List[Type['BemBase']]:
        """
        Get all subclasses of BemBase recursively.
        
        Returns:
        --------
        subclasses : List[Type[BemBase]]
            List of all subclass types
        """
        def get_all_subclasses(base_class):
            subclasses = []
            for subclass in base_class.__subclasses__():
                subclasses.append(subclass)
                subclasses.extend(get_all_subclasses(subclass))
            return subclasses
        
        return get_all_subclasses(cls)
    



# Example usage and testing
if __name__ == "__main__":
    # Example subclass for testing
    class ExampleSolver(BemBase):
        name = 'bemsolver'
        needs = [{'sim': 'stat'}, 'nev']
    
    class AnotherSolver(BemBase):
        name = 'bemsolver'
        needs = ['method', 'tolerance']
    
    # Test the find method
    op1 = {'sim': 'stat', 'nev': 100}
    result1 = BemBase.find('bemsolver', op1)
    print(f"Found class: {result1}")
    
    op2 = {'method': 'direct', 'tolerance': 1e-6}
    result2 = BemBase.find('bemsolver', op2)
    print(f"Found class: {result2}")
    
    op3 = {'sim': 'dynamic'}  # This should not match ExampleSolver
    result3 = BemBase.find('bemsolver', op3)
    print(f"Found class: {result3}")