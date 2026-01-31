"""
Visualization Module
====================
Nokia Fronthaul Network Analysis - Base Implementation

Generates required Nokia-style figures:
- Figure 1: Packet loss pattern snapshot
- Figure 2: Fronthaul topology diagram
- Figure 3: Link traffic with capacity line (per link)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


BYTES_TO_MBPS = 8 / 1_000_000


def generate_figure1_packet_loss_pattern(loss_df, output_path, time_window=60000):
    """
    Figure 1: Packet loss pattern snapshot (first 60 seconds).
    
    Args:
        loss_df: DataFrame with cells as columns, loss flags as values
        output_path: Path to save figure
        time_window: Number of slots to display (default 60000 = 60 seconds)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Select subset of cells and time window
    cells_to_plot = sorted(loss_df.columns)[:12]  # First 12 cells
    data_subset = loss_df[cells_to_plot].iloc[:time_window]
    
    # Create heatmap-style visualization
    for i, cell in enumerate(cells_to_plot):
        loss_events = data_subset[cell].values
        time_slots = np.arange(len(loss_events))
        
        # Plot loss events as vertical lines
        loss_indices = time_slots[loss_events == 1]
        ax.scatter(loss_indices, [i] * len(loss_indices), 
                  marker='|', s=10, color='red', alpha=0.6)
    
    ax.set_xlabel('Time Slot (1ms per slot)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell ID', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(cells_to_plot)))
    ax.set_yticklabels([f'Cell {c}' for c in cells_to_plot])
    ax.set_title('Figure 1: Packet Loss Pattern (First 60 seconds)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def generate_figure2_topology_diagram(link_groups, capacity_metrics, output_path):
    """
    Figure 2: Clean fronthaul topology diagram.
    
    Args:
        link_groups: Dict {link_id: [cell_ids]}
        capacity_metrics: List of capacity dicts
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Create capacity lookup
    capacity_lookup = {m['link_id']: m['recommended_capacity_mbps'] for m in capacity_metrics}
    
    # DU (top center)
    du_rect = mpatches.FancyBboxPatch((4, 8.5), 2, 0.8, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(du_rect)
    ax.text(5, 8.9, 'DU', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Links and cells
    link_positions = {'Link_1': 2, 'Link_2': 5, 'Link_3': 8}
    
    for link_id in sorted(link_groups.keys()):
        x_pos = link_positions[link_id]
        cells = link_groups[link_id]
        capacity = capacity_lookup.get(link_id, 0)
        
        # Draw link line from DU
        ax.plot([5, x_pos], [8.5, 7], 'k-', linewidth=2)
        
        # Link box
        link_rect = mpatches.FancyBboxPatch((x_pos - 0.6, 6.5), 1.2, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(link_rect)
        ax.text(x_pos, 6.95, link_id, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x_pos, 6.65, f'{capacity:.1f} Mbps', ha='center', va='center', fontsize=8)
        
        # Draw cells
        n_cells = len(cells)
        cell_spacing = min(0.4, 4.5 / max(n_cells, 1))
        start_y = 5.5 - (n_cells * cell_spacing) / 2
        
        for i, cell_id in enumerate(cells):
            y_pos = start_y + i * cell_spacing
            
            # Line from link to cell
            ax.plot([x_pos, x_pos], [6.5, y_pos + 0.15], 'k-', linewidth=0.5, alpha=0.5)
            
            # Cell circle
            cell_circle = mpatches.Circle((x_pos, y_pos), 0.12, 
                                         facecolor='orange', edgecolor='black', linewidth=1)
            ax.add_patch(cell_circle)
            ax.text(x_pos, y_pos, str(cell_id), ha='center', va='center', 
                   fontsize=7, fontweight='bold')
    
    ax.text(5, 9.5, 'Figure 2: 5G Fronthaul Network Topology', 
           ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def generate_figure3_link_traffic(link_id, cells, capacity, output_dir, 
                                  data_dir, time_window=60000):
    """
    Figure 3: Slot-level aggregated traffic for a link (60 seconds).
    
    Args:
        link_id: Link identifier
        cells: List of cell IDs on this link
        capacity: Recommended capacity in Mbps
        output_dir: Directory to save figure
        data_dir: Directory containing input CSV files
        time_window: Number of slots to plot
    """
    from data_loader import load_all_cells_data
    
    # Load throughput data
    throughput_df = load_all_cells_data('throughput', cells, data_dir)
    
    # Aggregate and convert to Mbps
    aggregate_throughput = throughput_df.sum(axis=1)
    aggregate_mbps = aggregate_throughput * BYTES_TO_MBPS
    
    # Plot subset
    data_subset = aggregate_mbps.iloc[:time_window]
    time_slots = np.arange(len(data_subset))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot traffic
    ax.plot(time_slots, data_subset.values, linewidth=0.8, color='#2E86AB', 
           label='Aggregated Traffic', alpha=0.8)
    
    # Capacity line
    ax.axhline(y=capacity, color='red', linestyle='--', linewidth=2, 
              label=f'Recommended Capacity: {capacity:.2f} Mbps')
    
    ax.set_xlabel('Time Slot (1ms per slot)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 3: {link_id} Traffic (60 seconds, {len(cells)} cells)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    output_path = output_dir / f"figure3_{link_id.lower()}_traffic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main visualization pipeline."""
    from data_loader import load_all_cells_data
    from capacity import load_topology
    
    print("\n" + "="*80)
    print("GENERATING NOKIA FIGURES")
    print("="*80 + "\n")
    
    # Configuration
    DATA_DIR = Path("phase1_slot_level_csvs")
    TOPOLOGY_FILE = Path("outputs/topology/cell_to_link_mapping.csv")
    CAPACITY_FILE = Path("outputs/capacity/capacity_summary.csv")
    OUTPUT_DIR = Path("outputs/figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    link_groups = load_topology(TOPOLOGY_FILE)
    capacity_df = pd.read_csv(CAPACITY_FILE)
    capacity_metrics = capacity_df.to_dict('records')
    
    # Convert column names back to lowercase for processing
    capacity_metrics = [
        {
            'link_id': m['Link_ID'],
            'recommended_capacity_mbps': m['Recommended_Capacity_Mbps']
        }
        for m in capacity_metrics
    ]
    
    all_cells = list(range(1, 25))
    loss_df = load_all_cells_data('loss', all_cells, DATA_DIR)
    print(f"  ✓ Loaded data\n")
    
    # Generate Figure 1: Packet loss pattern
    print("Generating Figure 1: Packet loss pattern...")
    generate_figure1_packet_loss_pattern(
        loss_df,
        OUTPUT_DIR / "figure1_packet_loss_pattern.png"
    )
    print()
    
    # Generate Figure 2: Topology diagram
    print("Generating Figure 2: Topology diagram...")
    generate_figure2_topology_diagram(
        link_groups,
        capacity_metrics,
        OUTPUT_DIR / "figure2_fronthaul_topology.png"
    )
    print()
    
    # Generate Figure 3: Link traffic (per link)
    print("Generating Figure 3: Link traffic plots...")
    for link_id in sorted(link_groups.keys()):
        cells = link_groups[link_id]
        capacity = next(m['recommended_capacity_mbps'] for m in capacity_metrics 
                       if m['link_id'] == link_id)
        
        generate_figure3_link_traffic(
            link_id, cells, capacity, OUTPUT_DIR, DATA_DIR
        )
    print()
    
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
