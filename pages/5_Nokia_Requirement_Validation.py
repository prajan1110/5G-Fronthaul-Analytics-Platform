"""
Streamlit Page: Nokia Requirement Validation
Shows validation against Nokia's specific requirements.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Nokia Requirement Validation", page_icon="✅", layout="wide")

# Title
st.title("✅ Nokia Requirement Validation")
st.markdown("**Validation Against Nokia's Fronthaul Network Requirements**")
st.markdown("---")

# Load capacity results
capacity_file = Path("outputs/capacity/capacity_analysis_results.csv")

if not capacity_file.exists():
    st.error("⚠️ Validation data not found. Please run analysis first: `python run_analysis.py`")
    st.stop()

capacity_df = pd.read_csv(capacity_file)

st.markdown("## 🎯 Nokia Requirements")

st.info("""
**Requirement:** Link capacity must be sufficient such that the percentage of time 
slots with overload (when buffer would overflow) is **less than 1%**.

**Buffer Specification:** 142.8 μs = 4 symbols
""")

st.markdown("---")

# Validation Results
st.markdown("## 📊 Validation Results")

# Overall status
all_pass = all(capacity_df['Overload_Percentage'] < 1.0)
pass_count = sum(capacity_df['Overload_Percentage'] < 1.0)
total_links = len(capacity_df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Links", f"{total_links}")

with col2:
    st.metric("Passing Links", f"{pass_count}")

with col3:
    pass_rate = (pass_count / total_links) * 100 if total_links > 0 else 0
    st.metric("Pass Rate", f"{pass_rate:.1f}%")

if all_pass:
    st.success("🎉 **All links meet Nokia requirements!**")
else:
    st.warning(f"⚠️ **{total_links - pass_count} link(s) exceed the threshold**")

st.markdown("---")

# Detailed validation table
st.markdown("## 📋 Detailed Validation Table")

# Create validation dataframe
validation_df = capacity_df[['Link_ID', 'Num_Cells', 'Capacity_With_Buffer_Mbps', 
                              'Overload_Percentage']].copy()
validation_df['Status'] = validation_df['Overload_Percentage'].apply(
    lambda x: '✅ PASS' if x < 1.0 else '❌ FAIL'
)
validation_df['Margin'] = 1.0 - validation_df['Overload_Percentage']

# Rename columns for display
validation_df.columns = ['Link', 'Cells', 'Required Capacity (Mbps)', 
                         'Overload %', 'Status', 'Margin (%)']

st.dataframe(
    validation_df,
    use_container_width=True,
    height=300
)

st.markdown("---")

# Visual comparison
st.markdown("## 📊 Visual Comparison Against Threshold")

fig = go.Figure()

colors = ['#10b981' if x < 1.0 else '#ef4444' for x in capacity_df['Overload_Percentage']]

fig.add_trace(go.Bar(
    x=capacity_df['Link_ID'],
    y=capacity_df['Overload_Percentage'],
    marker_color=colors,
    text=[f'{x:.3f}%' for x in capacity_df['Overload_Percentage']],
    textposition='outside',
    name='Actual Overload %'
))

# Add 1% threshold line
fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="red",
    line_width=3,
    annotation_text="Nokia Requirement: < 1%",
    annotation_position="right"
)

fig.update_layout(
    title="Overload Percentage vs Nokia Requirement (< 1%)",
    xaxis_title="Link",
    yaxis_title="Overload Percentage (%)",
    height=500,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Link-by-link analysis
st.markdown("## 🔍 Link-by-Link Analysis")

for idx, row in capacity_df.iterrows():
    link_id = row['Link_ID']
    overload_pct = row['Overload_Percentage']
    margin = 1.0 - overload_pct
    
    status = overload_pct < 1.0
    
    with st.expander(f"**{link_id}** - {'✅ PASS' if status else '❌ FAIL'}", expanded=not status):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Performance Metrics:**")
            st.markdown(f"- **Cells on link:** {int(row['Num_Cells'])}")
            st.markdown(f"- **Required capacity:** {row['Capacity_With_Buffer_Mbps']:.2f} Mbps")
            st.markdown(f"- **Mean throughput:** {row['Mean_Throughput_Mbps']:.2f} Mbps")
            st.markdown(f"- **Peak throughput:** {row['Max_Throughput_Mbps']:.2f} Mbps")
        
        with col2:
            st.markdown("**✅ Validation Results:**")
            st.markdown(f"- **Overload percentage:** {overload_pct:.4f}%")
            st.markdown(f"- **Nokia requirement:** < 1.0%")
            st.markdown(f"- **Margin:** {margin:.4f}%")
            st.markdown(f"- **Status:** {'PASS ✅' if status else 'FAIL ❌'}")
        
        # Progress bar for overload
        progress_value = min(overload_pct / 1.0, 1.0)  # Cap at 100%
        st.progress(progress_value, text=f"Overload: {overload_pct:.4f}% / 1.0%")
        
        if status:
            st.success(f"✅ {link_id} meets Nokia requirement with {margin:.4f}% margin")
        else:
            excess = overload_pct - 1.0
            st.error(f"❌ {link_id} exceeds limit by {excess:.4f}%")
            st.markdown(f"""
            **Recommended Actions:**
            - Increase link capacity by {excess * row['Capacity_With_Buffer_Mbps'] / 100:.2f} Mbps
            - Redistribute some cells to other links
            - Implement traffic shaping policies
            """)

st.markdown("---")

# Summary statistics
st.markdown("## 📈 Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Overload Statistics:**")
    st.markdown(f"- **Min:** {capacity_df['Overload_Percentage'].min():.4f}%")
    st.markdown(f"- **Max:** {capacity_df['Overload_Percentage'].max():.4f}%")
    st.markdown(f"- **Mean:** {capacity_df['Overload_Percentage'].mean():.4f}%")

with col2:
    st.markdown("**Capacity Statistics:**")
    st.markdown(f"- **Total capacity:** {capacity_df['Capacity_With_Buffer_Mbps'].sum():.2f} Mbps")
    st.markdown(f"- **Avg per link:** {capacity_df['Capacity_With_Buffer_Mbps'].mean():.2f} Mbps")
    st.markdown(f"- **Max link capacity:** {capacity_df['Capacity_With_Buffer_Mbps'].max():.2f} Mbps")

with col3:
    st.markdown("**Cell Distribution:**")
    st.markdown(f"- **Total cells:** {int(capacity_df['Num_Cells'].sum())}")
    st.markdown(f"- **Avg per link:** {capacity_df['Num_Cells'].mean():.1f}")
    st.markdown(f"- **Max on one link:** {int(capacity_df['Num_Cells'].max())}")

st.markdown("---")

# Final verdict
if all_pass:
    st.success("""
    ## 🎉 Validation Complete - All Requirements Met!
    
    **Summary:**
    - ✅ All {total_links} fronthaul links meet Nokia's overload requirement
    - ✅ All links maintain overload percentage < 1%
    - ✅ Buffer capacity (142.8 μs) is sufficient to handle traffic bursts
    - ✅ Network capacity is adequate for current traffic demands
    
    **Conclusion:**
    The fronthaul network is properly dimensioned and meets all Nokia requirements for Quality of Service.
    """.format(total_links=total_links))
else:
    failing_links = capacity_df[capacity_df['Overload_Percentage'] >= 1.0]
    st.error(f"""
    ## ⚠️ Validation Issues Detected
    
    **Failed Links:** {', '.join(failing_links['Link_ID'].tolist())}
    
    **Issues:**
    - {len(failing_links)} link(s) exceed the 1% overload threshold
    - These links require capacity upgrades or traffic redistribution
    
    **Next Steps:**
    1. Review capacity planning for failing links
    2. Consider redistributing cells across links
    3. Implement QoS policies and traffic shaping
    4. Monitor links closely for potential congestion
    """)

# Download validation report
st.markdown("---")

csv = validation_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Validation Report",
    data=csv,
    file_name="nokia_validation_report.csv",
    mime="text/csv"
)
