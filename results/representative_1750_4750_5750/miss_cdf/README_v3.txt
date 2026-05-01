[final_samplewise_cdf_v3]
- Main figure keeps only miss-rate CDF: FN / (TP + FN)
- Y-axis is normalized to 0~1 (true CDF)
- Thresholds: [-70.0, -65.0, -60.0]
- Main frequencies: [1750, 4750, 5750]
- Zoom frequencies: [1750]
- IMPORTANT: if the requested thresholds are not present in the input CSV,
  you must first regenerate per_sample_metrics.csv with those fixed thresholds.