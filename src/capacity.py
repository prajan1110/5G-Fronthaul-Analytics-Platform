"""
Capacity Estimation Module
===========================
Nokia Fronthaul Network Analysis - Base Implementation

Estimates required link capacity based on aggregated cell throughput.
Includes Nokia base requirements:
- Without buffer case
- With buffer case (4 symbols = 142.8 μs)
- Packet loss ≤ 1% validation
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Nokia Buffer Specification
SYMBOLS_PER_BUFFER = 4
SYMBOL_DURATION_US = 35.7
BUFFER_TIME_US = SYMBOLS_PER_BUFFER * SYMBOL_DURATION_US  # 142.8 μs

# Capacity planning
SAFETY_BUFFER = 0.15  # 15% margin
BYTES_TO_MBPS = 8 / 1_000_000


def load_topology(topology_file):
    """
    Load cell-to-link mapping from CSV.
    
    Returns:
        dict: {link_id: [cell_ids]}
    """
    df = pd.read_csv(topology_file)
    
    link_groups = {}
    for _, row in df.iterrows():
        link_id = row['Inferred_Link_ID']
        cell_id = row['Cell_ID']
        
        if link_id not in link_groups:
            link_groups[link_id] = []
        link_groups[link_id].append(cell_id)
    
    return link_groups


def compute_aggregate_throughput(throughput_df):
    """
    Sum throughput across all cells per slot.
    
    Args:
        throughput_df: DataFrame with cells as columns
        
    Returns:
        pandas.Series: Aggregated throughput per slot
    """
    return throughput_df.sum(axis=1)


def compute_capacity_metrics(aggregate_throughput_mbps, link_id):
    """
    Compute capacity statistics and Nokia requirements.
    
    Returns:
        dict: Capacity metrics including Nokia base requirements
    """
    metrics = {
        'link_id': link_id,
        'mean_mbps': aggregate_throughput_mbps.mean(),
        'p95_mbps': aggregate_throughput_mbps.quantile(0.95),
        'p99_mbps': aggregate_throughput_mbps.quantile(0.99),
    }
    
    # Recommended capacity (P99 + 15%)
    metrics['recommended_capacity_mbps'] = metrics['p99_mbps'] * (1 + SAFETY_BUFFER)
    
    # Nokia Requirement 1: Without Buffer Case
    metrics['capacity_without_buffer_mbps'] = aggregate_throughput_mbps.max()
    
    # Nokia Requirement 2: With Buffer Case
    metrics['buffer_time_us'] = BUFFER_TIME_US
    
    # Nokia Requirement 3: Packet Loss ≤ 1% Validation
    overload_slots = (aggregate_throughput_mbps > metrics['recommended_capacity_mbps']).sum()
    total_slots = len(aggregate_throughput_mbps)
    metrics['overload_slot_percentage'] = (overload_slots / total_slots) * 100
    metrics['nokia_constraint_status'] = 'PASS' if metrics['overload_slot_percentage'] <= 1.0 else 'FAIL'
    
    return metrics


def save_capacity_summary(all_metrics, output_path):
    """
    Save capacity summary to CSV.
    
    Args:
        all_metrics: List of metric dicts
        output_path: Path to save CSV
    """
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns
    columns = [
        'link_id', 'mean_mbps', 'p95_mbps', 'p99_mbps',
        'recommended_capacity_mbps', 'capacity_without_buffer_mbps',
        'buffer_time_us', 'overload_slot_percentage', 'nokia_constraint_status'
    ]
    df = df[columns]
    
    # Rename for clarity
    df = df.rename(columns={
        'link_id': 'Link_ID',
        'mean_mbps': 'Mean_Mbps',
        'p95_mbps': 'P95_Mbps',
        'p99_mbps': 'P99_Mbps',
        'recommended_capacity_mbps': 'Recommended_Capacity_Mbps',
        'capacity_without_buffer_mbps': 'Capacity_Without_Buffer_Mbps',
        'buffer_time_us': 'Buffer_Time_Microseconds',
        'overload_slot_percentage': 'Overload_Slot_Percentage',
        'nokia_constraint_status': 'Nokia_Constraint_Status'
    })
    
    # Round values
    numeric_cols = ['Mean_Mbps', 'P95_Mbps', 'P99_Mbps', 'Recommended_Capacity_Mbps', 'Capacity_Without_Buffer_Mbps']
    df[numeric_cols] = df[numeric_cols].round(2)
    df['Buffer_Time_Microseconds'] = df['Buffer_Time_Microseconds'].round(1)
    df['Overload_Slot_Percentage'] = df['Overload_Slot_Percentage'].round(3)
    
    df.to_csv(output_path, index=False)
    
    return df


def main():
    """Main capacity estimation pipeline."""
    from data_loader import load_all_cells_data
    
    print("\n" + "="*80)
    print("CAPACITY ESTIMATION")
    print("="*80 + "\n")
    
    # Configuration
    DATA_DIR = Path("phase1_slot_level_csvs")
    TOPOLOGY_FILE = Path("outputs/topology/cell_to_link_mapping.csv")
    OUTPUT_DIR = Path("outputs/capacity")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load topology
    print("Loading topology...")
    link_groups = load_topology(TOPOLOGY_FILE)
    print(f"  ✓ Loaded {len(link_groups)} links\n")
    
    all_metrics = []
    
    for link_id in sorted(link_groups.keys()):
        cells = link_groups[link_id]
        print(f"Processing {link_id} ({len(cells)} cells)...")
        
        # Load throughput data
        throughput_df = load_all_cells_data('throughput', cells, DATA_DIR)
        
        # Aggregate throughput
        aggregate_throughput = compute_aggregate_throughput(throughput_df)
        aggregate_throughput_mbps = aggregate_throughput * BYTES_TO_MBPS
        
        # Compute metrics
        metrics = compute_capacity_metrics(aggregate_throughput_mbps, link_id)
        metrics['num_cells'] = len(cells)
        all_metrics.append(metrics)
        
        print(f"  Mean: {metrics['mean_mbps']:.2f} Mbps")
        print(f"  P99: {metrics['p99_mbps']:.2f} Mbps")
        print(f"  Recommended: {metrics['recommended_capacity_mbps']:.2f} Mbps")
        print(f"  Without Buffer: {metrics['capacity_without_buffer_mbps']:.2f} Mbps")
        print(f"  Overload: {metrics['overload_slot_percentage']:.3f}% [{metrics['nokia_constraint_status']}]\n")
    
    # Save summary
    print("Saving capacity summary...")
    summary_df = save_capacity_summary(all_metrics, OUTPUT_DIR / "capacity_summary.csv")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'capacity_summary.csv'}\n")
    
    print(summary_df.to_string(index=False))
    print()
    
    print("="*80)
    print("CAPACITY ESTIMATION COMPLETE")
    print("="*80 + "\n")
    
    return all_metrics


if __name__ == "__main__":
    main()
