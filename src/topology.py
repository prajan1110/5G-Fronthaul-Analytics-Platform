"""
Topology Identification Module
===============================
Nokia Fronthaul Network Analysis - Base Implementation

Identifies which cells share the same fronthaul link based on
correlated packet loss patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_jaccard_similarity(loss_a, loss_b):
    """
    Compute Jaccard similarity between two binary loss timelines.
    
    Formula: |A ∩ B| / |A ∪ B|
    
    Args:
        loss_a, loss_b: Binary arrays of packet loss flags
        
    Returns:
        float: Similarity score [0, 1]
    """
    intersection = np.sum((loss_a == 1) & (loss_b == 1))
    union = np.sum((loss_a == 1) | (loss_b == 1))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def build_similarity_matrix(loss_df):
    """
    Build pairwise similarity matrix for all cells.
    
    Args:
        loss_df: DataFrame with cells as columns, loss flags as values
        
    Returns:
        pandas.DataFrame: Symmetric similarity matrix
    """
    n_cells = len(loss_df.columns)
    similarity = np.zeros((n_cells, n_cells))
    
    for i, cell_i in enumerate(loss_df.columns):
        for j, cell_j in enumerate(loss_df.columns):
            if i == j:
                similarity[i, j] = 1.0
            elif i < j:
                sim = compute_jaccard_similarity(
                    loss_df[cell_i].values,
                    loss_df[cell_j].values
                )
                similarity[i, j] = sim
                similarity[j, i] = sim
    
    return pd.DataFrame(similarity, index=loss_df.columns, columns=loss_df.columns)


def assign_cells_to_links(similarity_matrix, anchor_cells={'Link_1': [], 'Link_2': [1], 'Link_3': [2]}, threshold=0.4):
    """
    Assign cells to links using anchor-seeded grouping.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        anchor_cells: Dict mapping link IDs to anchor cell lists
        threshold: Minimum similarity for group membership
        
    Returns:
        dict: {link_id: [cell_ids]}
    """
    all_cells = list(similarity_matrix.index)
    assigned_cells = set()
    link_groups = {link: list(cells) for link, cells in anchor_cells.items()}
    
    # Add anchor cells to assigned set
    for cells in anchor_cells.values():
        assigned_cells.update(cells)
    
    # Assign remaining cells
    unassigned = [c for c in all_cells if c not in assigned_cells]
    
    for cell in unassigned:
        best_link = None
        best_score = -1
        
        for link, anchors in link_groups.items():
            if not anchors:
                continue
            
            avg_sim = similarity_matrix.loc[cell, anchors].mean()
            
            if avg_sim > threshold and avg_sim > best_score:
                best_score = avg_sim
                best_link = link
        
        if best_link:
            link_groups[best_link].append(cell)
        else:
            # Assign to Link_1 by default (no anchor group)
            link_groups['Link_1'].append(cell)
    
    # Sort cells in each link
    for link in link_groups:
        link_groups[link] = sorted(link_groups[link])
    
    return link_groups


def save_topology(link_groups, output_path):
    """
    Save cell-to-link mapping as CSV.
    
    Args:
        link_groups: Dict {link_id: [cell_ids]}
        output_path: Path to save CSV file
    """
    rows = []
    for link_id, cells in sorted(link_groups.items()):
        for cell_id in cells:
            rows.append({'Cell_ID': cell_id, 'Inferred_Link_ID': link_id})
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Cell_ID')
    df.to_csv(output_path, index=False)
    
    return df


def main():
    """Main topology identification pipeline."""
    from data_loader import load_all_cells_data
    
    print("\n" + "="*80)
    print("TOPOLOGY IDENTIFICATION")
    print("="*80 + "\n")
    
    # Configuration
    DATA_DIR = Path("phase1_slot_level_csvs")
    OUTPUT_DIR = Path("outputs/topology")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    CELL_IDS = list(range(1, 25))
    ANCHORS = {'Link_1': [], 'Link_2': [1], 'Link_3': [2]}
    
    # Load packet loss data
    print("Loading packet loss data...")
    loss_df = load_all_cells_data('loss', CELL_IDS, DATA_DIR)
    print(f"  ✓ Loaded {len(loss_df)} common slots for {len(CELL_IDS)} cells\n")
    
    # Build similarity matrix
    print("Computing pairwise similarity...")
    similarity_matrix = build_similarity_matrix(loss_df)
    print(f"  ✓ Built {len(similarity_matrix)}×{len(similarity_matrix)} similarity matrix\n")
    
    # Assign cells to links
    print("Assigning cells to links...")
    link_groups = assign_cells_to_links(similarity_matrix, ANCHORS)
    
    for link_id in sorted(link_groups.keys()):
        cells = link_groups[link_id]
        print(f"  {link_id}: {len(cells)} cells → {cells}")
    print()
    
    # Save results
    print("Saving topology mapping...")
    topology_df = save_topology(link_groups, OUTPUT_DIR / "cell_to_link_mapping.csv")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'cell_to_link_mapping.csv'}\n")
    
    print("="*80)
    print("TOPOLOGY IDENTIFICATION COMPLETE")
    print("="*80 + "\n")
    
    return link_groups, similarity_matrix


if __name__ == "__main__":
    main()
