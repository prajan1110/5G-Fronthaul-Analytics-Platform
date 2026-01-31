"""
Data Loader Module
==================
Nokia Fronthaul Network Analysis - Base Implementation

Loads slot-level packet loss and throughput data from CSV files.
"""

import pandas as pd
from pathlib import Path


def load_packet_loss(cell_id, data_dir):
    """
    Load packet loss timeline for a specific cell.
    
    Args:
        cell_id: Cell identifier (1-24)
        data_dir: Path to directory containing CSV files
        
    Returns:
        pandas.Series: Binary loss flags indexed by slot
    """
    filepath = Path(data_dir) / f"cell_{cell_id}_packet_loss_per_slot.csv"
    df = pd.read_csv(filepath)
    return pd.Series(data=df['loss_flag'].values, index=df['slot'].values, name=f'cell_{cell_id}')


def load_throughput(cell_id, data_dir):
    """
    Load throughput timeline for a specific cell.
    
    Args:
        cell_id: Cell identifier (1-24)
        data_dir: Path to directory containing CSV files
        
    Returns:
        pandas.Series: Throughput values indexed by slot
    """
    filepath = Path(data_dir) / f"cell_{cell_id}_throughput_per_slot.csv"
    df = pd.read_csv(filepath)
    return pd.Series(data=df['throughput'].values, index=df['slot'].values, name=f'cell_{cell_id}')


def load_all_cells_data(data_type, cell_ids, data_dir):
    """
    Load data for multiple cells and align on common slots.
    
    Args:
        data_type: 'loss' or 'throughput'
        cell_ids: List of cell identifiers
        data_dir: Path to directory containing CSV files
        
    Returns:
        pandas.DataFrame: Aligned data with cells as columns, slots as index
    """
    loader = load_packet_loss if data_type == 'loss' else load_throughput
    
    data = {}
    for cell_id in cell_ids:
        data[cell_id] = loader(cell_id, data_dir)
    
    df = pd.DataFrame(data)
    df = df.dropna()  # Keep only common slots
    
    return df
