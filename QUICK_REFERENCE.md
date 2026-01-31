# QUICK REFERENCE - Nokia Fronthaul Analysis

**Clean Base Implementation - Fast Access Guide**

---

## 🚀 Quick Start

```bash
# Run everything
python run_analysis.py

# Run individual phases
python -m src.topology
python -m src.capacity  
python -m src.visualization
```

---

## 📁 File Locations

### Input
```
phase1_slot_level_csvs/          # 48 CSV files (24 cells × 2 types)
```

### Outputs
```
outputs/topology/cell_to_link_mapping.csv          # Cell → Link mapping
outputs/capacity/capacity_summary.csv              # Capacity recommendations
outputs/figures/figure1_packet_loss_pattern.png    # Loss pattern
outputs/figures/figure2_fronthaul_topology.png     # Topology diagram
outputs/figures/figure3_link_*_traffic.png         # Traffic plots (×3)
```

---

## 📊 Results Summary

### Topology
- **Link-1**: 11 cells → [3,4,5,9,11,12,14,17,20,21,22]
- **Link-2**: 1 cell → [1]
- **Link-3**: 12 cells → [2,6,7,8,10,13,15,16,18,19,23,24]

### Capacity
| Link | Recommended | Without Buffer | Status |
|------|-------------|----------------|--------|
| Link-1 | 5.36 Mbps | 9.31 Mbps | ✅ PASS (0.077%) |
| Link-2 | 0.06 Mbps | 4.65 Mbps | ✅ PASS (0.919%) |
| Link-3 | 5.58 Mbps | 11.68 Mbps | ✅ PASS (0.865%) |

### Nokia Requirements
- ✅ Buffer: 4 symbols × 35.7 μs = **142.8 μs**
- ✅ Packet loss ≤ 1% validated for all links
- ✅ Without buffer case: max instantaneous throughput

---

## 🔧 Module Functions

### src/data_loader.py
```python
load_packet_loss(cell_id, data_dir)           # Load loss timeline
load_throughput(cell_id, data_dir)            # Load throughput timeline
load_all_cells_data(type, cells, data_dir)    # Load & align multiple cells
```

### src/topology.py
```python
compute_jaccard_similarity(loss_a, loss_b)    # Similarity score
build_similarity_matrix(loss_df)              # 24×24 matrix
assign_cells_to_links(matrix, anchors)        # Group cells
save_topology(link_groups, path)              # Save CSV
```

### src/capacity.py
```python
load_topology(file)                           # Load cell→link mapping
compute_aggregate_throughput(df)              # Sum across cells
compute_capacity_metrics(agg, link)           # Compute statistics
save_capacity_summary(metrics, path)          # Save CSV
```

### src/visualization.py
```python
generate_figure1_packet_loss_pattern()        # Loss pattern plot
generate_figure2_topology_diagram()           # Network diagram
generate_figure3_link_traffic()               # Traffic + capacity line
```

---

## 📐 Key Formulas

### Topology - Jaccard Similarity
```
Similarity = |A ∩ B| / |A ∪ B|
```
Where A, B are sets of loss event slots

### Capacity - Recommended
```
Recommended = P99 × 1.15
```
99th percentile + 15% safety buffer

### Nokia Buffer
```
Buffer_Time = 4 symbols × 35.7 μs = 142.8 μs
Buffer_Bits = Buffer_Time × Link_Rate
```
Example: @ 1 GbE → 142,800 bits

### Validation
```
Overload% = (slots > capacity) / total_slots × 100
Status = PASS if Overload% ≤ 1%
```

---

## 🎯 Quality Metrics

### Topology Validation
- ✅ Anchor accuracy: 100% (Cell-1→Link-2, Cell-2→Link-3)
- ✅ Intra-link similarity: 0.384
- ✅ Inter-link similarity: 0.293
- ✅ Separation: 9.1% (clear distinction)

### Capacity Validation
- ✅ P99 > P95 (percentile consistency)
- ✅ Recommended > P99 (buffer applied)
- ✅ Peak >> Mean (realistic traffic)
- ✅ Overload ≤ 1% (Nokia constraint)

---

## 📚 Documentation Files

- `README_CLEAN.md` - Main documentation
- `REFACTORING_SUMMARY.md` - Refactoring report
- `MIGRATION_CHECKLIST.md` - Migration guide
- `QUICK_REFERENCE.md` - This file

---

## 🔄 Typical Workflow

1. **Run Analysis**
   ```bash
   python run_analysis.py
   ```

2. **Check Outputs**
   ```bash
   cat outputs/topology/cell_to_link_mapping.csv
   cat outputs/capacity/capacity_summary.csv
   ```

3. **View Figures**
   - Open `outputs/figures/*.png` in image viewer

4. **Verify Results**
   - Check all 3 links show PASS status
   - Verify topology matches expected (11+1+12)
   - Confirm figures generated correctly

---

## 🛠️ Troubleshooting

### Issue: Module not found
```bash
# Solution: Run from Nokia/ directory
cd c:\Users\hamsa\Documents\Nokia
python run_analysis.py
```

### Issue: CSV file not found
```bash
# Solution: Ensure input CSVs exist
dir phase1_slot_level_csvs\*.csv
```

### Issue: Import errors
```bash
# Solution: Install dependencies
pip install pandas numpy matplotlib
```

---

## 📊 Performance

- **Phase 1 (Topology)**: ~30 seconds
- **Phase 2 (Capacity)**: ~15 seconds
- **Phase 3 (Visualization)**: ~10 seconds
- **Total Runtime**: ~1 minute

**Resource Usage**:
- Memory: ~200 MB peak
- Disk: 4 MB outputs
- CPU: Single-threaded

---

## 🎨 Output Specifications

### CSV Format
- UTF-8 encoding
- Comma-separated
- Header row included
- Numeric precision: 2-3 decimals

### Figure Format
- PNG format
- 300 DPI resolution
- Size: 12-14 inches wide
- Professional styling

---

## ✅ Verification Commands

```bash
# Count output files
(Get-ChildItem -Path outputs -Recurse -File).Count  # Should be 7

# Check CSV rows
(Import-Csv outputs/topology/cell_to_link_mapping.csv).Count    # 24
(Import-Csv outputs/capacity/capacity_summary.csv).Count        # 3

# Check figures exist
Test-Path outputs/figures/figure*.png  # All should be True
```

---

## 🚀 Next Steps

### Ready for Enhancement
- [ ] Add ML models (Phase 4)
- [ ] Add automation (Phase 5)
- [ ] Add dashboard (Phase 6)

### Code Extensions
- [ ] Real-time data ingestion
- [ ] Predictive analytics
- [ ] Alert system
- [ ] Interactive visualizations

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 31, 2026  
**Status**: Production Ready

**For full documentation, see `README_CLEAN.md`**
