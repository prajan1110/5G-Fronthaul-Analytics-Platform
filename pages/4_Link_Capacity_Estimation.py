"""
Streamlit Page: Link Capacity Estimation
Shows capacity analysis and buffer modeling results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Link Capacity Estimation", page_icon="📈", layout="wide")

# Title
st.title("📈 Link Capacity Estimation")
st.markdown("**Nokia Challenge 2: Capacity Analysis with Buffer Modeling**")
st.markdown("---")

# Load capacity results
capacity_file = Path("outputs/capacity/capacity_analysis_results.csv")

if not capacity_file.exists():
    st.error("⚠️ Capacity results not found. Please run analysis first: `python run_analysis.py`")
    st.stop()

capacity_df = pd.read_csv(capacity_file)

st.markdown("## 🎯 Slot-Level Traffic Aggregation")

st.info("""
This analysis addresses **Nokia Challenge 2: Link Capacity Estimation** by analyzing 
aggregated traffic patterns and buffer effects.
""")

st.markdown("### 📋 Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔹 Traffic Aggregation:**")
    st.markdown("For each link, sum throughput from all assigned cells at each 1 ms time slot")
    
    st.markdown("**🔹 Distribution Analysis:**")
    st.markdown("Examine the distribution of aggregated traffic to identify peak loads and typical operating points")
    
    st.markdown("**🔹 Buffer Modeling:**")
    st.markdown("Account for buffer capacity (142.8 μs = 4 symbols) which allows temporary traffic bursts above link capacity")

with col2:
    st.markdown("**🔹 Capacity Calculation:**")
    st.markdown("Determine minimum capacity to ensure packet loss ≤ 1%")
    
    st.markdown("**📊 Two Capacity Scenarios:**")
    st.markdown("- **Without Buffer:** Instantaneous capacity needed to avoid any packet loss")
    st.markdown("- **With Buffer:** Reduced capacity needed when 142.8 μs buffer absorbs bursts")

st.markdown("---")

# Capacity Summary
st.markdown("## 📊 Capacity Summary")

for idx, row in capacity_df.iterrows():
    link_id = row['Link_ID']
    
    with st.expander(f"**{link_id}** - Analysis Results", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cells", f"{int(row['Num_Cells'])}")
        
        with col2:
            st.metric("Capacity (No Buffer)", f"{row['Capacity_No_Buffer_Mbps']:.2f} Mbps")
        
        with col3:
            st.metric("Capacity (With Buffer)", f"{row['Capacity_With_Buffer_Mbps']:.2f} Mbps")
        
        with col4:
            overload_pct = row['Overload_Percentage']
            status = "✅ PASS" if overload_pct < 1.0 else "⚠️ FAIL"
            st.metric("Overload %", f"{overload_pct:.2f}%", delta=status)
        
        # Detailed metrics
        st.markdown("**📈 Detailed Metrics:**")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"- **Mean Throughput:** {row['Mean_Throughput_Mbps']:.2f} Mbps")
            st.markdown(f"- **Median Throughput:** {row['Median_Throughput_Mbps']:.2f} Mbps")
        
        with col_b:
            st.markdown(f"- **95th Percentile:** {row['P95_Throughput_Mbps']:.2f} Mbps")
            st.markdown(f"- **99th Percentile:** {row['P99_Throughput_Mbps']:.2f} Mbps")
        
        with col_c:
            st.markdown(f"- **Max Throughput:** {row['Max_Throughput_Mbps']:.2f} Mbps")
            st.markdown(f"- **Std Dev:** {row['Std_Throughput_Mbps']:.2f} Mbps")
        
        # Validation status
        if overload_pct < 1.0:
            st.success(f"✅ **{link_id} meets Nokia requirement:** Overload < 1% ({overload_pct:.2f}%)")
        else:
            st.error(f"⚠️ **{link_id} exceeds limit:** Overload = {overload_pct:.2f}% (requirement: < 1%)")

st.markdown("---")

# Visualizations
st.markdown("## 📊 Capacity Visualization")

# Bar chart comparison
fig = go.Figure()

fig.add_trace(go.Bar(
    x=capacity_df['Link_ID'],
    y=capacity_df['Capacity_No_Buffer_Mbps'],
    name='Without Buffer',
    marker_color='#ef4444'
))

fig.add_trace(go.Bar(
    x=capacity_df['Link_ID'],
    y=capacity_df['Capacity_With_Buffer_Mbps'],
    name='With Buffer (142.8 μs)',
    marker_color='#10b981'
))

fig.update_layout(
    title="Link Capacity Comparison: With vs Without Buffer",
    xaxis_title="Link",
    yaxis_title="Capacity (Mbps)",
    barmode='group',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("**Interpretation:**")
st.markdown("- **Red bars (Without Buffer):** Peak capacity needed for instantaneous traffic")
st.markdown("- **Green bars (With Buffer):** Reduced capacity when buffer smooths out bursts")
st.markdown("- **Buffer benefit:** Allows lower capacity links while maintaining QoS")

st.markdown("---")

# Overload percentage
st.markdown("## 🎯 Overload Percentage Analysis")

fig2 = go.Figure()

colors = ['#10b981' if x < 1.0 else '#ef4444' for x in capacity_df['Overload_Percentage']]

fig2.add_trace(go.Bar(
    x=capacity_df['Link_ID'],
    y=capacity_df['Overload_Percentage'],
    marker_color=colors,
    text=[f'{x:.2f}%' for x in capacity_df['Overload_Percentage']],
    textposition='outside'
))

# Add reference line at 1%
fig2.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="red",
    annotation_text="1% Threshold (Nokia Requirement)",
    annotation_position="right"
)

fig2.update_layout(
    title="Overload Percentage per Link (Nokia Requirement: < 1%)",
    xaxis_title="Link",
    yaxis_title="Overload Percentage (%)",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

# Overall validation
st.markdown("---")

all_pass = all(capacity_df['Overload_Percentage'] < 1.0)

if all_pass:
    st.success("""
    ✅ **All Links Pass Nokia Requirements!**
    
    All fronthaul links maintain overload percentage below 1%, demonstrating sufficient capacity 
    with the 142.8 μs buffer to handle traffic bursts while meeting QoS requirements.
    """)
else:
    failing_links = capacity_df[capacity_df['Overload_Percentage'] >= 1.0]['Link_ID'].tolist()
    st.error(f"""
    ⚠️ **Capacity Issues Detected**
    
    The following links exceed the 1% overload threshold: {', '.join(failing_links)}
    
    **Recommended Actions:**
    - Increase link capacity
    - Implement traffic shaping
    - Redistribute cells across links
    """)

# Download results
st.markdown("---")

csv = capacity_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Capacity Analysis Results",
    data=csv,
    file_name="capacity_analysis_results.csv",
    mime="text/csv"
)
