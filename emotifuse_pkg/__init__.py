"""Python packaging namespace for the compiled `emotifuse` extension.

This module imports the compiled C++ extension and makes the classes
available for use.
"""

# This package provides access to the compiled C++ extension 'emotifuse'
# The extension is built as 'emotifuse.so' in the parent directory

import sys
import os
from pathlib import Path

# Add the parent directory (where the .so file is) to the path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import the actual compiled module
try:
    # Import the compiled extension directly 
    import emotifuse as _ext
    
    # Re-export all classes at package level
    EmotiFuse = _ext.EmotiFuse
    AudioIO = _ext.AudioIO  
    FeatureExtractor = _ext.FeatureExtractor
    Fusion = _ext.Fusion
    BranchONNX = _ext.BranchONNX
    FrameBuffer = _ext.FrameBuffer
    
    __all__ = ['EmotiFuse', 'AudioIO', 'FeatureExtractor', 'Fusion', 'BranchONNX', 'FrameBuffer']
    
finally:
    # Clean up sys.path
    if _parent_dir in sys.path:
        sys.path.remove(_parent_dir) 