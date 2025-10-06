# WHAT:  verify all libraries installed correctly
# WHY: Catch issues before week 1 starts
# When: Run immediately after installation
# File: test_setup.py

import sys

def test_imports():
    """Test all critical imports"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import scipy
        
        print("✅ All core libraries installed successfully!")
        print(f"✅ Python version: {sys.version}")
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ Pandas version: {pd.__version__}")
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
        
        # Test basic functionality
        arr = np.array([1, 2, 3])
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        print("✅ Basic operations work!")
        print("\n🚀 Environment ready for bootcamp!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    test_imports()