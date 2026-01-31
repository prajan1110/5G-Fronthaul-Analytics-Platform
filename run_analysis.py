"""
Nokia Fronthaul Network Analysis
=================================
Main Execution Pipeline

Runs all phases:
1. Topology Identification
2. Capacity Estimation  
3. Visualization Generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import topology, capacity, visualization


def main():
    """Execute complete Nokia fronthaul analysis pipeline."""
    
    print("\n" + "="*80)
    print(" "*20 + "NOKIA FRONTHAUL NETWORK ANALYSIS")
    print(" "*25 + "Base Implementation")
    print("="*80 + "\n")
    
    try:
        # Phase 1: Topology Identification
        print("PHASE 1: TOPOLOGY IDENTIFICATION")
        print("-"*80)
        topology.main()
        
        # Phase 2: Capacity Estimation
        print("\nPHASE 2: CAPACITY ESTIMATION")
        print("-"*80)
        capacity.main()
        
        # Phase 3: Visualization
        print("\nPHASE 3: VISUALIZATION")
        print("-"*80)
        visualization.main()
        
        # Summary
        print("\n" + "="*80)
        print(" "*30 + "ANALYSIS COMPLETE")
        print("="*80)
        print("\nOutput Files:")
        print("  • outputs/topology/cell_to_link_mapping.csv")
        print("  • outputs/capacity/capacity_summary.csv")
        print("  • outputs/figures/figure1_packet_loss_pattern.png")
        print("  • outputs/figures/figure2_fronthaul_topology.png")
        print("  • outputs/figures/figure3_link1_traffic.png")
        print("  • outputs/figures/figure3_link2_traffic.png")
        print("  • outputs/figures/figure3_link3_traffic.png")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
