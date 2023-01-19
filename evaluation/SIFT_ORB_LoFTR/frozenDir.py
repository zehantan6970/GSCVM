"""
冻结路径
"""
import sys
import os
def relativePath():
    """Returns the relative path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

if __name__=='__main__':
    print(relativePath())
