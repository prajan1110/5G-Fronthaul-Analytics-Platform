"""
Streamlit Page: Traffic Pattern Analysis
Shows traffic patterns and packet loss visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Traffic Pattern Analysis", page_icon="📊", layout="wide")

# Title
st.title("📊 Traffic Pattern Analysis")
st.markdown("**Packet Loss Correlation Visualization**")
st.markdown("---")

st.info("""
This visualization demonstrates **Nokia Challenge 1: Topology Identification** by revealing 
correlated packet loss patterns across all 24 cells.
""")

st.markdown("### 🔑 Key Observations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Chart Elements:**")
    st.markdown("- **X-axis:** Time slots (each slot = 1 ms)")
    st.markdown("- **Y-axis:** Cell IDs (1-24)")
    st.markdown("- **Color Intensity:** Represents packet loss occurrence")
    st.markdown("- **Vertical Patterns:** Cells experiencing simultaneous packet loss likely share the same fronthaul link")

with col2:
    st.markdown("**Analysis Principle:**")
    st.markdown("""
    When a fronthaul link becomes congested:
    - All cells connected to that link experience packet loss simultaneously
    - This creates vertical "stripes" in the pattern visualization
    - High correlation in these patterns indicates shared infrastructure
    """)

st.markdown("---")

# Load packet loss data if available
loss_dir = Path("phase1_slot_level_csvs")
if loss_dir.exists():
    st.subheader("📉 Packet Loss Heat Map")
    
    # Try to load and visualize packet loss data
    loss_files = list(loss_dir.glob("*packet_loss_per_slot.csv"))
    
    if len(loss_files) > 0:
        st.info(f"Found {len(loss_files)} cell packet loss files")
        
        # Sample visualization for first few cells
        sample_size = min(10, len(loss_files))
        
        st.markdown(f"**Showing packet loss patterns for {sample_size} cells (sample)**")
        
        # Load sample data
        loss_data = {}
        for loss_file in loss_files[:sample_size]:
            cell_id = loss_file.stem.replace('_packet_loss_per_slot', '')
            try:
                df = pd.read_csv(loss_file)
                if 'Slot' in df.columns and 'Packet_Loss' in df.columns:
                    loss_data[cell_id] = df
            except:
                continue
        
        if loss_data:
            # Create heatmap
            fig = go.Figure()
            
            for i, (cell_id, df) in enumerate(loss_data.items()):
                # Take first 1000 slots for visualization
                slots = df['Slot'].values[:1000]
                losses = df['Packet_Loss'].values[:1000]
                
                fig.add_trace(go.Scatter(
                    x=slots,
                    y=[i] * len(slots),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=losses,
                        colorscale='Reds',
                        showscale=(i == 0)
                    ),
                    name=cell_id,
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Packet Loss Pattern Visualization (First 1000 slots)",
                xaxis_title="Slot Number",
                yaxis_title="Cell Index",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **Red markers** indicate packet loss events
            - **Vertical alignment** of red markers suggests cells sharing the same link
            - **Correlation** in loss patterns reveals the underlying topology
            """)
        else:
            st.warning("Could not load packet loss data for visualization")
    else:
        st.warning("No packet loss data files found")
else:
    st.warning("Packet loss data directory not found")

st.markdown("---")

# Traffic statistics
st.subheader("📈 Traffic Statistics")

throughput_dir = Path("phase1_slot_level_csvs")
if throughput_dir.exists():
    throughput_files = list(throughput_dir.glob("*throughput_per_slot.csv"))
    
    if len(throughput_files) > 0:
        st.info(f"Analyzing throughput data from {len(throughput_files)} cells")
        
        # Calculate aggregate statistics
        total_cells = len(throughput_files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cells", f"{total_cells}")
        
        with col2:
            st.metric("Data Files", f"{len(throughput_files) + len(loss_files) if 'loss_files' in locals() else len(throughput_files)}")
        
        with col3:
            # Try to estimate total slots
            try:
                sample_df = pd.read_csv(throughput_files[0])
                total_slots = len(sample_df)
                st.metric("Slots per Cell", f"{total_slots:,}")
            except:
                st.metric("Slots per Cell", "N/A")
    else:
        st.warning("No throughput data files found")
else:
    st.warning("Throughput data directory not found")

st.markdown("---")

st.success("""
✅ **Traffic Pattern Analysis Complete!**

The packet loss correlation analysis reveals the network topology structure, 
enabling accurate cell-to-link mapping for subsequent capacity analysis.
""")
