import pickle
import pandas as pd
import numpy as np
import os

def load_csv(filepath):
    """Loads a CSV file into a pandas DataFrame."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

def save_pickle(obj, filepath):
    """Saves a Python object as a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """Loads a pickle file."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

def save_numpy_array(arr, filepath):
    """Saves a NumPy array to a file."""
    np.save(filepath, arr)

def load_numpy_array(filepath):
    """Loads a NumPy array from a file."""
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")
